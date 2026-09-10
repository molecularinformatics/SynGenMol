"""Dynamic token masks for chemistry-constrained decoding.

Compatibility masks are defined only over building-block tokens.  Token-level
trajectory grammar, including EOS availability, is handled separately so a
chemical compatibility check never removes a special token from the policy.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def mask_from_ids(vocab_size: int, token_ids: Iterable[int], *, device: torch.device | None = None) -> torch.Tensor:
    """Return a boolean vocabulary mask that is true exactly at ``token_ids``."""
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    ids = sorted({int(token_id) for token_id in token_ids})
    if ids:
        if ids[0] < 0 or ids[-1] >= vocab_size:
            raise ValueError(f"Token ids must be within [0, {vocab_size}), got {ids[0]}..{ids[-1]}")
        mask[torch.tensor(ids, dtype=torch.long, device=device)] = True
    return mask


def union_masks(masks: Iterable[torch.Tensor], *, vocab_size: int, device: torch.device) -> torch.Tensor:
    """Return the union of compatible-building-block masks."""
    result = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for mask in masks:
        if mask.numel() != vocab_size:
            raise ValueError("All masks must have the same vocabulary size")
        result |= mask.to(device=device, dtype=torch.bool)
    return result


def policy_mask(
    *,
    vocab_size: int,
    building_block_ids: Iterable[int],
    compatible_building_block_ids: Iterable[int],
    allowed_special_ids: Iterable[int] = (),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the complete action mask for one decoding state.

    The compatibility relation applies only to entries in ``building_block_ids``.
    Special tokens are added through ``allowed_special_ids`` and are therefore
    unaffected by chemical compatibility.  For example, callers can enable EOS
    after at least one successful reaction without adding it to every reaction
    compatibility cache.
    """
    building_block_ids = {int(token_id) for token_id in building_block_ids}
    compatible_building_block_ids = {int(token_id) for token_id in compatible_building_block_ids}
    allowed_special_ids = {int(token_id) for token_id in allowed_special_ids}

    if not compatible_building_block_ids.issubset(building_block_ids):
        unexpected = sorted(compatible_building_block_ids - building_block_ids)
        raise ValueError(f"Compatibility map contains non-building-block token ids: {unexpected[:5]}")

    mask = mask_from_ids(vocab_size, compatible_building_block_ids | allowed_special_ids, device=device)
    return mask


def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Set unavailable actions to negative infinity without modifying input logits."""
    if logits.shape[-1] != allowed.numel():
        raise ValueError(
            f"Logit vocabulary size {logits.shape[-1]} does not match mask size {allowed.numel()}"
        )
    if not bool(allowed.any()):
        raise ValueError("Cannot construct a policy distribution with no allowed actions")
    allowed = allowed.to(device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
