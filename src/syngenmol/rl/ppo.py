"""PPO optimization helpers for masked SynGenMol trajectories."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from trl import PPOConfig, PPOTrainer

from syngenmol.models.gpt import GeneratedTrajectory
from syngenmol.oracles.base import MolecularOracle, require_aligned_scores


@dataclass(frozen=True)
class PPOSettings:
    """Material PPO settings exposed by the public training workflow."""

    batch_size: int = 256
    mini_batch_size: int = 64
    learning_rate: float = 1e-5
    initial_kl_coefficient: float = 0.05
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_loss_coefficient: float = 0.5
    gamma: float = 1.0
    lam: float = 0.95
    entropy_coefficient: float = 0.1
    ppo_epochs: int = 4
    # Exponential-moving-average schedule for the KL reference policy:
    # every `reference_update_interval` updates, reference <- decay * reference
    # + (1 - decay) * policy.  An interval of 0 keeps the reference frozen.
    reference_ema_decay: float = 0.99
    reference_update_interval: int = 3
    # Products at or above this molecular weight keep a score of 0.0 and are not
    # submitted to the oracle.  Use None to score every product.
    maximum_molecular_weight: float | None = 800.0
    # Seed for the resampling that pads an undersized batch, so a training run
    # is reproducible from its configuration alone.
    resample_seed: int = 42


def build_ppo_trainer(model: Any, tokenizer: Any, settings: PPOSettings) -> PPOTrainer:
    """Build the TRL PPO trainer for the masked policy.

    The generated token sequence has already been sampled under the dynamic
    mask, so the wrapped SynGenMol policy itself is the constrained policy used
    in PPO likelihood ratios and KL regularization.
    """
    config = PPOConfig(
        model_name=None,
        batch_size=settings.batch_size,
        mini_batch_size=settings.mini_batch_size,
        learning_rate=settings.learning_rate,
        kl_penalty="abs",
        adap_kl_ctrl=False,
        init_kl_coef=settings.initial_kl_coefficient,
        cliprange=settings.clip_range,
        cliprange_value=settings.value_clip_range,
        vf_coef=settings.value_loss_coefficient,
        gamma=settings.gamma,
        lam=settings.lam,
        ppo_epochs=settings.ppo_epochs,
        ratio_threshold=10,
    )
    return PPOTrainer(config=config, model=model, tokenizer=tokenizer)


@torch.no_grad()
def ema_update_reference(reference: Any, source: Any, *, decay: float) -> None:
    """Blend the KL reference policy toward the current policy, in place.

    ``reference <- decay * reference + (1 - decay) * source``

    A reference policy frozen at initialization anchors the KL penalty to
    randomly initialized weights, which stops being an informative constraint
    once the policy has moved.  Updating it as a slow exponential moving average
    of the trained policy keeps the anchor near the current policy, so the
    penalty limits how fast the policy may change instead of how far it may
    travel from its starting point.

    Only floating-point parameters and buffers are blended; integer and boolean
    buffers (attention masks, for example) are copied so the reference wrapper
    stays structurally identical to the policy.

    Tied weights appear under several names in ``state_dict`` while sharing one
    storage, so updates are applied per distinct storage.  Blending them once per
    name would raise their effective decay to ``decay ** number_of_names``.
    """
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must lie in [0, 1]")
    reference_model = getattr(reference, "pretrained_model", reference)
    source_model = getattr(source, "pretrained_model", source)
    source_state = source_model.state_dict()
    updated_storages: set[tuple[Any, int]] = set()
    for name, reference_tensor in reference_model.state_dict().items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            continue
        storage = (reference_tensor.device, reference_tensor.data_ptr())
        if storage in updated_storages:
            continue
        updated_storages.add(storage)
        source_tensor = source_tensor.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
        if torch.is_floating_point(reference_tensor):
            reference_tensor.mul_(decay).add_(source_tensor, alpha=1.0 - decay)
        else:
            reference_tensor.copy_(source_tensor)


def _trajectory_key(tokens: torch.Tensor) -> tuple[int, ...]:
    """Return a hashable identity for one sampled building-block sequence."""
    return tuple(int(token) for token in tokens.reshape(-1).tolist())


def _molecular_weight(product: Chem.Mol) -> float:
    """Return the molecular weight, or a large sentinel when it cannot be computed."""
    try:
        weight = float(Descriptors.MolWt(product))
    except Exception:
        return 2000.0
    return weight if weight > 0.0 else 2000.0


def trajectory_table(trajectories: list[GeneratedTrajectory]) -> pd.DataFrame:
    """Convert valid generated branches to a score-ready table."""
    rows: list[dict[str, object]] = []
    for trajectory in trajectories:
        rows.append(
            {
                "tokens": trajectory.token_ids.detach().cpu(),
                "reaction_ids": list(trajectory.reaction_ids),
                "product_smiles": Chem.MolToSmiles(
                    trajectory.product, canonical=True, isomericSmiles=True
                ),
                "molecular_weight": _molecular_weight(trajectory.product),
                "entropy": float(trajectory.mean_entropy.detach().cpu()),
            }
        )
    return pd.DataFrame(rows)


class SynGenMolPPOTrainer:
    """Compact task-specific PPO loop around a masked SynGenMol policy."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        oracle: MolecularOracle,
        settings: PPOSettings = PPOSettings(),
        output_dir: str | Path | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.oracle = oracle
        self.settings = settings
        self.ppo = build_ppo_trainer(model, tokenizer, settings)
        self._completed_updates = 0
        self.output_dir = None if output_dir is None else Path(output_dir)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _skip_unused_terminal_masks(self):
        """Avoid masking the logit after the final PPO response action.

        TRL computes response log probabilities from ``logits[:, :-1]``.  The
        logit at the final input position consequently cannot affect the
        likelihood ratio, KL term, value loss, or policy gradient of a supplied
        response.  Computing its dynamic chemical mask can nevertheless require
        an unnecessary reaction expansion.  Toggle the execution flag on both
        policy and reference wrappers for the duration of one PPO update while
        preserving the masked distributions for every actual response action.
        """
        policies: list[Any] = []
        seen: set[int] = set()
        for candidate in (self.model, getattr(self.ppo, "model", None), getattr(self.ppo, "ref_model", None)):
            policy = getattr(candidate, "pretrained_model", candidate)
            if policy is not None and hasattr(policy, "_mask_final_logit") and id(policy) not in seen:
                policies.append(policy)
                seen.add(id(policy))
        prior = [bool(policy._mask_final_logit) for policy in policies]
        for policy in policies:
            policy._mask_final_logit = False
        try:
            yield
        finally:
            for policy, value in zip(policies, prior):
                policy._mask_final_logit = value

    def _update_reference_policy(self) -> bool:
        """Advance the KL reference policy every ``reference_update_interval`` updates."""
        interval = int(self.settings.reference_update_interval)
        step = self._completed_updates
        self._completed_updates += 1
        if interval <= 0:
            return False
        reference = getattr(self.ppo, "ref_model", None)
        if reference is None:
            return False
        # ``1 % interval`` keeps the original schedule (updates at 1, 1 + K, ...)
        # while still meaning "every update" when the interval is 1.
        if step % interval != 1 % interval:
            return False
        ema_update_reference(reference, self.model, decay=float(self.settings.reference_ema_decay))
        return True

    def _score(self, table: pd.DataFrame) -> pd.DataFrame:
        """Score generated branches and reduce each trajectory to its best product.

        Products heavier than ``maximum_molecular_weight`` keep a score of 0.0
        and are never sent to the oracle, so obviously out-of-range assemblies
        neither consume oracle budget nor reward the policy.
        """
        table = table.copy()
        table["score"] = 0.0
        limit = self.settings.maximum_molecular_weight
        scoreable = table.index if limit is None else table.index[table["molecular_weight"] < float(limit)]
        if len(scoreable) > 0:
            smiles = table.loc[scoreable, "product_smiles"].tolist()
            table.loc[scoreable, "score"] = require_aligned_scores(smiles, self.oracle.score_smiles(smiles))
        return self._reduce_to_best_product(table)

    @staticmethod
    def _reduce_to_best_product(table: pd.DataFrame) -> pd.DataFrame:
        """Represent each sampled trajectory by its highest-scoring product.

        One sampled building-block sequence can reach several products through
        the different reaction templates that accept the same pair.  Those
        branches are alternative interpretations of a single sampled route, not
        independent samples, so the trajectory is scored by the best product it
        can reach and contributes exactly one row to the PPO batch.  Without this
        reduction the same token sequence would receive gradient once per
        product, which optimizes the mean over branches rather than the maximum.
        """
        if table.empty:
            return table
        keyed = table.assign(_trajectory=table["tokens"].map(_trajectory_key))
        best = keyed.loc[keyed.groupby("_trajectory")["score"].idxmax()]
        return best.drop(columns="_trajectory").reset_index(drop=True)

    def step(
        self,
        *,
        max_reactions: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        epsilon: float = 0.0,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Generate, score, and perform one PPO update."""
        policy = self.model.pretrained_model
        device = next(policy.parameters()).device
        candidates: list[GeneratedTrajectory] = []
        while len(candidates) < self.settings.batch_size:
            candidates.extend(
                policy.generate_trajectory_branches(
                    max_reactions=max_reactions,
                    temperature=temperature,
                    top_p=top_p,
                    epsilon=epsilon,
                    device=device,
                )
            )
        table = self._score(trajectory_table(candidates))
        if table.empty:
            raise RuntimeError("No valid, scoreable trajectories were generated for this PPO step")
        if len(table) > self.settings.batch_size:
            table = table.iloc[: self.settings.batch_size].copy()
        # Report the sampled batch before any padding, so repeated rows added to
        # reach the configured batch size cannot skew the reported statistics.
        score_mean = float(table["score"].mean())
        score_max = float(table["score"].max())
        if len(table) < self.settings.batch_size:
            table = pd.concat(
                [
                    table,
                    table.sample(
                        n=self.settings.batch_size - len(table),
                        replace=True,
                        random_state=int(self.settings.resample_seed),
                    ),
                ],
                ignore_index=True,
            )

        queries = [torch.tensor([policy.config.bos_token_id], device=device)] * len(table)
        responses = [tensor.to(device=device).reshape(-1) for tensor in table["tokens"]]
        rewards = [
            torch.tensor(
                float(score) + self.settings.entropy_coefficient * float(entropy),
                dtype=torch.float32,
                device=device,
            )
            for score, entropy in zip(table["score"], table["entropy"])
        ]
        with self._skip_unused_terminal_masks():
            status = self.ppo.step(queries, responses, rewards)
        reference_updated = self._update_reference_policy()
        summary = {key: float(value) for key, value in status.items() if isinstance(value, (int, float))}
        summary["score_mean"] = score_mean
        summary["score_max"] = score_max
        summary["reference_policy_updated"] = float(reference_updated)
        if self.output_dir is not None:
            table.to_csv(self.output_dir / "latest_trajectories.csv", index=False)
        return table, summary
