"""Masked GPT-2 policy used for SynGenMol molecular assembly.

This implementation is reconstructed from the retained multistep SynGenMol
model pathway.  It intentionally excludes legacy two-component, three-component,
soft-prompt, and unsupported external-oracle code paths.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from rdkit import Chem
from transformers import GPT2Config, GPT2LMHeadModel

from syngenmol.chemistry.masking import mask_logits, policy_mask
from syngenmol.chemistry.search_space import PreparedSearchSpace


@dataclass
class GeneratedTrajectory:
    """One valid route branch emitted by :class:`SynGenMolGPT2Model`."""

    token_ids: torch.Tensor
    product: Chem.Mol
    reaction_ids: tuple[int, ...]
    mean_entropy: torch.Tensor


class SynGenMolGPT2Model(GPT2LMHeadModel):
    """Decoder-only building-block policy with dynamic reaction compatibility.

    At each building-block selection, compatibility is evaluated through the
    reaction SMARTS associated with the current intermediate.  The policy mask
    excludes invalid building-block tokens while preserving grammar-level special
    tokens such as EOS whenever the route state permits termination.
    """

    def __init__(
        self,
        config: GPT2Config,
        *,
        search_space: PreparedSearchSpace,
        initial_building_block_ids: Iterable[int] | None = None,
    ) -> None:
        super().__init__(config)
        self.search_space = search_space
        self.templates = {template.identifier: template for template in search_space.templates}
        self.compatibility = search_space.compatibility
        self.bb_token_ids = frozenset(search_space.building_block_ids)
        self.bb_mols = search_space.token_to_mol

        # The next-action mask after the first selected building block depends
        # only on its compatible reaction templates.  Cache these masks lazily:
        # PPO repeatedly evaluates the same first-token prefixes, while a fully
        # materialized table would be unnecessarily large for production-scale
        # building-block vocabularies.
        self._initial_reaction_bits_by_token: dict[int, int] = {}
        for reaction_id, by_position in self.compatibility.items():
            bit = 1 << int(reaction_id)
            for token_id in by_position[0]:
                token_id = int(token_id)
                self._initial_reaction_bits_by_token[token_id] = (
                    self._initial_reaction_bits_by_token.get(token_id, 0) | bit
                )
        self._first_step_mask_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._first_step_mask_cache_size = 256
        # TRL evaluates logits through the final response token even though that
        # logit is not used to score any response action.  PPO temporarily sets
        # this flag to False and therefore avoids unnecessary route expansion.
        self._mask_final_logit = True

        if config.vocab_size != search_space.tokenizer.vocab_size:
            raise ValueError(
                "GPT-2 config vocab_size must equal the prepared search-space tokenizer size "
                f"({config.vocab_size} != {search_space.tokenizer.vocab_size})"
            )
        if config.bos_token_id != search_space.tokenizer.tokenizer.bos_token_id:
            raise ValueError("GPT-2 BOS token does not match the prepared tokenizer")
        if config.eos_token_id != search_space.tokenizer.tokenizer.eos_token_id:
            raise ValueError("GPT-2 EOS token does not match the prepared tokenizer")

        start_ids = set(self._union_compatible_ids(self.templates, position=0))
        if initial_building_block_ids is not None:
            initial_building_block_ids = {int(token_id) for token_id in initial_building_block_ids}
            if not initial_building_block_ids.issubset(self.bb_token_ids):
                raise ValueError("Initial constraint contains token IDs that are not building blocks")
            start_ids &= initial_building_block_ids
        if not start_ids:
            raise ValueError("No building blocks can start a valid reaction under the current constraints")
        self._initial_building_block_ids = frozenset(start_ids)
        self.is_peft_model = False  # Required by supported TRL value-head wrappers.

    @classmethod
    def from_search_space(
        cls,
        search_space: PreparedSearchSpace,
        *,
        n_embd: int = 256,
        n_layer: int = 4,
        n_head: int = 4,
        initial_building_block_ids: Iterable[int] | None = None,
        **config_kwargs: object,
    ) -> "SynGenMolGPT2Model":
        tokenizer = search_space.tokenizer.tokenizer
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **config_kwargs,
        )
        return cls(
            config,
            search_space=search_space,
            initial_building_block_ids=initial_building_block_ids,
        )

    def _union_compatible_ids(
        self, reaction_ids: Iterable[int], *, position: int
    ) -> set[int]:
        ids: set[int] = set()
        for reaction_id in reaction_ids:
            ids.update(self.compatibility[reaction_id][position])
        return ids

    def _reactions_accepting_intermediate(self, intermediate: Chem.Mol) -> list[int]:
        return [
            reaction_id
            for reaction_id, template in self.templates.items()
            if template.matches_reactant(intermediate, position=0)
        ]

    def _first_step_action_mask(self, token_id: int, *, device: torch.device) -> torch.Tensor:
        """Return the partner-building-block mask after an initial token.

        Compatibility for an unreacted building block is already available in
        the prepared cache, so this path does not need to rerun SMARTS matching
        for every PPO forward pass.  CPU masks are retained in a bounded LRU
        cache and copied only when a non-CPU policy device is used.
        """
        token_id = int(token_id)
        cached = self._first_step_mask_cache.get(token_id)
        if cached is None:
            reaction_bits = self._initial_reaction_bits_by_token.get(token_id, 0)
            compatible_ids: set[int] = set()
            for reaction_id, by_position in self.compatibility.items():
                if reaction_bits & (1 << int(reaction_id)):
                    compatible_ids.update(int(partner_id) for partner_id in by_position[1])
            cached = policy_mask(
                vocab_size=self.config.vocab_size,
                building_block_ids=self.bb_token_ids,
                compatible_building_block_ids=compatible_ids,
                device=torch.device("cpu"),
            )
            self._first_step_mask_cache[token_id] = cached
            self._first_step_mask_cache.move_to_end(token_id)
            while len(self._first_step_mask_cache) > self._first_step_mask_cache_size:
                self._first_step_mask_cache.popitem(last=False)
        else:
            self._first_step_mask_cache.move_to_end(token_id)
        return cached.to(device=device, non_blocking=True)

    def allowed_token_mask(
        self,
        *,
        intermediate: Chem.Mol | None,
        completed_reactions: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the complete policy mask for the current route state.

        Compatibility only gates building-block tokens. EOS is explicitly added
        after one completed reaction, so special tokens can never disappear merely
        because a reaction template has no matching building-block token.
        """
        if intermediate is None:
            compatible_ids = self._initial_building_block_ids
            special_ids: set[int] = set()
        else:
            compatible_reactions = self._reactions_accepting_intermediate(intermediate)
            compatible_ids = self._union_compatible_ids(compatible_reactions, position=1)
            special_ids = (
                {int(self.config.eos_token_id)} if completed_reactions >= 1 else set()
            )

        return policy_mask(
            vocab_size=self.config.vocab_size,
            building_block_ids=self.bb_token_ids,
            compatible_building_block_ids=compatible_ids,
            allowed_special_ids=special_ids,
            device=device,
        )

    @staticmethod
    def sample_masked_token(
        logits: torch.Tensor,
        allowed: torch.Tensor,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        epsilon: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one token from the normalized masked policy and return entropy."""
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("Masked autoregressive sampling currently expects logits with shape (1, vocab)")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("top_p must lie in [0, 1]")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")

        constrained_logits = mask_logits(logits / temperature, allowed)
        keep = allowed.to(device=logits.device, dtype=torch.bool).view(1, -1)

        if 0.0 < top_p < 1.0:
            permitted_logits = constrained_logits[keep]
            sorted_logits, sort_order = torch.sort(permitted_logits, descending=True)
            cumulative_probability = F.softmax(sorted_logits, dim=0).cumsum(dim=0)
            selected = cumulative_probability <= top_p
            selected[0] = True
            permitted_keep = torch.zeros_like(permitted_logits, dtype=torch.bool)
            permitted_keep[sort_order[selected]] = True
            keep = torch.zeros_like(keep)
            keep[allowed.view(1, -1)] = permitted_keep
            constrained_logits = mask_logits(constrained_logits, keep.view(-1))

        probabilities = F.softmax(constrained_logits, dim=-1)
        if epsilon:
            uniform = keep.to(dtype=probabilities.dtype)
            uniform = uniform / uniform.sum(dim=-1, keepdim=True)
            probabilities = (1.0 - epsilon) * probabilities + epsilon * uniform
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(dim=-1)
        return torch.multinomial(probabilities, num_samples=1), entropy

    def _apply_reaction(self, intermediate: Chem.Mol, partner_token_id: int, reaction_id: int) -> list[Chem.Mol]:
        return self.templates[reaction_id].run(intermediate, self.bb_mols[partner_token_id])

    def _advance_states(
        self,
        states: list[tuple[Chem.Mol | None, int]],
        token_id: int,
    ) -> list[tuple[Chem.Mol | None, int]]:
        """Advance route states after a sampled token while preserving branches.

        A state is ``(intermediate, completed_reactions)``.  The first building
        block creates an intermediate without applying a template.  A later
        building block is applied to every compatible template branch.  EOS
        terminates all active states and therefore contributes no next-state
        action mask.
        """
        if token_id == int(self.config.eos_token_id):
            return []
        if token_id not in self.bb_token_ids:
            return []

        next_states: list[tuple[Chem.Mol | None, int]] = []
        seen: set[tuple[str, int]] = set()
        for intermediate, completed_reactions in states:
            if intermediate is None:
                candidate_states = [(Chem.Mol(self.bb_mols[token_id]), completed_reactions)]
            else:
                candidate_states = []
                for reaction_id in self._reactions_accepting_intermediate(intermediate):
                    if token_id not in self.compatibility[reaction_id][1]:
                        continue
                    candidate_states.extend(
                        (product, completed_reactions + 1)
                        for product in self._apply_reaction(intermediate, token_id, reaction_id)
                    )
            for product, new_depth in candidate_states:
                key = (Chem.MolToSmiles(product, canonical=True, isomericSmiles=True), new_depth)
                if key not in seen:
                    seen.add(key)
                    next_states.append((product, new_depth))
        return next_states

    def _allowed_mask_for_states(
        self,
        states: list[tuple[Chem.Mol | None, int]],
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return the union policy mask over all currently feasible route branches."""
        if not states:
            return None
        allowed = torch.zeros(self.config.vocab_size, dtype=torch.bool, device=device)
        for intermediate, completed_reactions in states:
            allowed |= self.allowed_token_mask(
                intermediate=intermediate,
                completed_reactions=completed_reactions,
                device=device,
            )
        return allowed

    def _mask_logits_for_input_ids(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        mask_final_logit: bool,
    ) -> torch.Tensor:
        """Apply dynamic masks to every next-token distribution in a batch.

        This makes the masked, normalized distribution the policy used by both
        sampling and PPO likelihood calculations.  It reconstructs all feasible
        reaction branches from each prefix, so ambiguous template applications
        are represented by the union of their valid next building blocks.
        """
        if input_ids.ndim != 2:
            return logits
        masked = logits.clone()
        batch_size, sequence_length = input_ids.shape
        for batch_index in range(batch_size):
            valid_length = sequence_length
            if attention_mask is not None:
                valid_length = int(attention_mask[batch_index].to(dtype=torch.long).sum().item())
            if valid_length <= 0:
                continue
            tokens = input_ids[batch_index, :valid_length].tolist()
            # A causal logit at position i predicts token i+1.  PPO uses logits
            # only through the final response token, so it can omit the final
            # next-token mask without changing any action log probability.
            positions_to_mask = valid_length if mask_final_logit else max(valid_length - 1, 0)
            states: list[tuple[Chem.Mol | None, int]] = [(None, 0)]
            for position, token_id in enumerate(tokens[:positions_to_mask]):
                if position == 0:
                    if token_id != int(self.config.bos_token_id):
                        # This policy expects the standard BOS-prefixed
                        # trajectories used throughout the public workflow.
                        states = []
                    allowed = self._allowed_mask_for_states(states, device=logits.device)
                elif position == 1 and len(states) == 1 and states[0][0] is None:
                    # The first building block has not yet undergone a reaction.
                    # Its compatible partners are completely determined by the
                    # prepared first-reactant compatibility cache.
                    states = self._advance_states(states, int(token_id))
                    allowed = self._first_step_action_mask(int(token_id), device=logits.device)
                else:
                    states = self._advance_states(states, int(token_id))
                    allowed = self._allowed_mask_for_states(states, device=logits.device)
                if allowed is not None:
                    masked[batch_index, position] = mask_logits(
                        masked[batch_index, position].unsqueeze(0), allowed
                    ).squeeze(0)
        return masked

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *,
        apply_dynamic_mask: bool = True,
        mask_final_logit: bool | None = None,
        **kwargs,
    ):
        """Run GPT-2 and apply state-dependent chemistry masks to its logits.

        Dynamic masking is part of the policy itself, not merely a sampling
        post-processing step.  Passing a complete BOS-prefixed trajectory makes
        it possible to reconstruct the valid action set for every prefix.
        ``mask_final_logit`` is an execution optimization: callers may omit the
        final next-token distribution only when it is provably unused, as in a
        PPO response-likelihood calculation.
        """
        output = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if apply_dynamic_mask and input_ids is not None:
            if mask_final_logit is None:
                mask_final_logit = bool(self._mask_final_logit)
            output.logits = self._mask_logits_for_input_ids(
                output.logits,
                input_ids,
                attention_mask,
                mask_final_logit=bool(mask_final_logit),
            )
            # GPT-2's CausalLM loss is normally evaluated internally before a
            # subclass can replace ``output.logits``.  Recompute it from the
            # masked policy if callers supplied labels, so likelihood-based
            # training and diagnostics use the same constrained distribution.
            labels = kwargs.get("labels")
            if labels is not None:
                shifted_logits = output.logits[..., :-1, :].contiguous()
                shifted_labels = labels[..., 1:].to(shifted_logits.device).contiguous()
                output.loss = F.cross_entropy(
                    shifted_logits.view(-1, shifted_logits.size(-1)),
                    shifted_labels.view(-1),
                    ignore_index=-100,
                )
        return output

    @torch.inference_mode()
    def generate_trajectory_branches(
        self,
        *,
        max_reactions: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        epsilon: float = 0.0,
        device: torch.device | None = None,
    ) -> list[GeneratedTrajectory]:
        """Sample a building-block trajectory and enumerate valid reaction branches.

        The first token selects a compatible starting building block.  Each later
        token is sampled from building blocks compatible with the current
        intermediate, plus EOS after the first successful reaction.  Sampling EOS
        terminates the route without attempting another template application.
        """
        if max_reactions < 1:
            raise ValueError("max_reactions must be at least 1")
        device = device or next(self.parameters()).device
        bos = int(self.config.bos_token_id)
        eos = int(self.config.eos_token_id)
        start_tokens = torch.tensor([[bos]], dtype=torch.long, device=device)
        results: list[GeneratedTrajectory] = []
        self._expand_branch(
            generated=start_tokens,
            intermediate=None,
            reaction_ids=(),
            completed_reactions=0,
            max_reactions=max_reactions,
            temperature=temperature,
            top_p=top_p,
            epsilon=epsilon,
            entropy_sum=torch.zeros((), device=device),
            entropy_count=0,
            eos_token_id=eos,
            results=results,
        )
        return results

    def _expand_branch(
        self,
        *,
        generated: torch.Tensor,
        intermediate: Chem.Mol | None,
        reaction_ids: tuple[int, ...],
        completed_reactions: int,
        max_reactions: int,
        temperature: float,
        top_p: float,
        epsilon: float,
        entropy_sum: torch.Tensor,
        entropy_count: int,
        eos_token_id: int,
        results: list[GeneratedTrajectory],
    ) -> None:
        if completed_reactions >= max_reactions:
            if intermediate is not None:
                results.append(
                    GeneratedTrajectory(
                        token_ids=generated[:, 1:].squeeze(0).detach().clone(),
                        product=Chem.Mol(intermediate),
                        reaction_ids=reaction_ids,
                        mean_entropy=entropy_sum / max(entropy_count, 1),
                    )
                )
            return

        logits = self(
            input_ids=generated,
            use_cache=False,
            mask_final_logit=True,
        ).logits[:, -1, :]
        if intermediate is not None and completed_reactions == 0 and generated.shape[1] == 2:
            allowed = self._first_step_action_mask(int(generated[0, -1].item()), device=generated.device)
        else:
            allowed = self.allowed_token_mask(
                intermediate=intermediate,
                completed_reactions=completed_reactions,
                device=generated.device,
            )
        if not bool(allowed.any()):
            if intermediate is not None:
                results.append(
                    GeneratedTrajectory(
                        token_ids=generated[:, 1:].squeeze(0).detach().clone(),
                        product=Chem.Mol(intermediate),
                        reaction_ids=reaction_ids,
                        mean_entropy=entropy_sum / max(entropy_count, 1),
                    )
                )
            return

        token, entropy = self.sample_masked_token(
            logits,
            allowed,
            temperature=temperature,
            top_p=top_p,
            epsilon=epsilon,
        )
        token_id = int(token.item())
        next_entropy_sum = entropy_sum + entropy.squeeze(0)
        next_entropy_count = entropy_count + 1

        if token_id == eos_token_id:
            if intermediate is not None:
                results.append(
                    GeneratedTrajectory(
                        token_ids=generated[:, 1:].squeeze(0).detach().clone(),
                        product=Chem.Mol(intermediate),
                        reaction_ids=reaction_ids,
                        mean_entropy=next_entropy_sum / next_entropy_count,
                    )
                )
            return

        next_generated = torch.cat([generated, token], dim=1)
        if intermediate is None:
            self._expand_branch(
                generated=next_generated,
                intermediate=Chem.Mol(self.bb_mols[token_id]),
                reaction_ids=reaction_ids,
                completed_reactions=completed_reactions,
                max_reactions=max_reactions,
                temperature=temperature,
                top_p=top_p,
                epsilon=epsilon,
                entropy_sum=next_entropy_sum,
                entropy_count=next_entropy_count,
                eos_token_id=eos_token_id,
                results=results,
            )
            return

        compatible_reactions = self._reactions_accepting_intermediate(intermediate)
        for reaction_id in compatible_reactions:
            if token_id not in self.compatibility[reaction_id][1]:
                continue
            for product in self._apply_reaction(intermediate, token_id, reaction_id):
                self._expand_branch(
                    generated=next_generated.clone(),
                    intermediate=product,
                    reaction_ids=reaction_ids + (reaction_id,),
                    completed_reactions=completed_reactions + 1,
                    max_reactions=max_reactions,
                    temperature=temperature,
                    top_p=top_p,
                    epsilon=epsilon,
                    entropy_sum=next_entropy_sum,
                    entropy_count=next_entropy_count,
                    eos_token_id=eos_token_id,
                    results=results,
                )


# Compatibility alias for downstream code during the migration. New code should
# import SynGenMolGPT2Model.
CustomGPT2ModelMultiStep = SynGenMolGPT2Model
