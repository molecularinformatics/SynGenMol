"""Model construction and portable checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from syngenmol.chemistry.search_space import PreparedSearchSpace
from syngenmol.models.gpt import SynGenMolGPT2Model


def build_policy_model(
    search_space: PreparedSearchSpace,
    *,
    n_embd: int = 256,
    n_layer: int = 4,
    n_head: int = 4,
    initial_building_block_ids: Iterable[int] | None = None,
) -> SynGenMolGPT2Model:
    """Construct a randomly initialized masked SynGenMol GPT-2 policy."""
    return SynGenMolGPT2Model.from_search_space(
        search_space,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        initial_building_block_ids=initial_building_block_ids,
    )


def load_policy_checkpoint(
    model: SynGenMolGPT2Model,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> SynGenMolGPT2Model:
    """Load a state dictionary saved by the public SynGenMol workflow."""
    checkpoint_path = Path(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a model state dictionary")

    # ``train`` serializes the TRL value-head wrapper so that optimization can
    # be resumed externally.  Molecular generation needs only the underlying
    # masked language-model policy.  Accept both a bare policy state dictionary
    # and a TRL wrapper state dictionary, dropping its value-head parameters.
    expected_keys = set(model.state_dict())
    policy_state: dict[str, torch.Tensor] = {}
    unsupported_keys: list[str] = []
    for key, value in state.items():
        normalized_key = key.removeprefix("pretrained_model.")
        if normalized_key.startswith("v_head."):
            continue
        if normalized_key in expected_keys:
            policy_state[normalized_key] = value
        else:
            unsupported_keys.append(key)
    if unsupported_keys:
        preview = ", ".join(unsupported_keys[:5])
        suffix = " ..." if len(unsupported_keys) > 5 else ""
        raise ValueError(
            f"Checkpoint {checkpoint_path} contains unsupported non-value-head keys: {preview}{suffix}"
        )
    model.load_state_dict(policy_state, strict=strict)
    return model
