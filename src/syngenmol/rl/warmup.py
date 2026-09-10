"""Task-specific supervised warm-up for a SynGenMol value head.

Warm-up jointly optimizes the shared Transformer trunk and value head against
an oracle-scored trajectory pool.  It is not large-scale route pretraining and
is not value-head-only training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast


class _TrajectoryDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, rewards: torch.Tensor) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.rewards = rewards

    def __len__(self) -> int:
        return int(self.rewards.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "reward": self.rewards[index],
        }


@dataclass(frozen=True)
class WarmupResult:
    reward_mean: float
    reward_std: float
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]


def warmup_value_model(
    model: Any,
    tokenizer: PreTrainedTokenizerFast,
    trajectories: pd.DataFrame,
    *,
    score_column: str = "score",
    sequence_column: str = "name",
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 1e-4,
    validation_fraction: float = 0.2,
    split_seed: int | None = None,
    device: str | torch.device | None = None,
) -> WarmupResult:
    """Fit trajectory-return estimates using terminal-value MSE.

    The input sequence column must contain whitespace-separated building-block
    identifiers.  The model is expected to be a TRL value-head wrapper whose
    forward output exposes per-token values at index 2.
    """
    if score_column not in trajectories.columns:
        raise ValueError(f"Warm-up data has no score column {score_column!r}")
    if sequence_column not in trajectories.columns:
        raise ValueError(f"Warm-up data has no sequence column {sequence_column!r}")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    table = trajectories[[sequence_column, score_column]].dropna().copy()
    if table.empty:
        raise ValueError("No complete warm-up trajectory rows are available")
    sequence_strings = [f"{tokenizer.bos_token} {sequence}" for sequence in table[sequence_column].astype(str)]
    rewards = torch.as_tensor(table[score_column].astype(float).to_numpy(), dtype=torch.float32)
    reward_mean = float(rewards.mean().item())
    reward_std = float(rewards.std(unbiased=False).item()) or 1.0
    encoded = tokenizer(sequence_strings, return_tensors="pt", padding=True)
    dataset = _TrajectoryDataset(encoded["input_ids"], encoded["attention_mask"], rewards)

    if len(dataset) < 2:
        raise ValueError("At least two warm-up trajectories are required for a train/validation split")
    generator = None
    if split_seed is not None:
        generator = torch.Generator().manual_seed(int(split_seed))
    permutation = torch.randperm(len(dataset), generator=generator)
    cutoff = max(1, min(len(dataset) - 1, int((1.0 - validation_fraction) * len(dataset))))
    train_set = Subset(dataset, permutation[:cutoff])
    validation_set = Subset(dataset, permutation[cutoff:])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        accumulated = 0.0
        batches = 0
        for batch in tqdm(train_loader, desc=f"Warm-up {epoch + 1}/{epochs}", leave=False):
            inputs = {key: value.to(device) for key, value in batch.items() if key != "reward"}
            target = (batch["reward"].to(device) - reward_mean) / reward_std
            # The value loss consumes hidden states, not language-model logits;
            # skipping post-logit masking leaves value predictions unchanged.
            output = model(**inputs, apply_dynamic_mask=False)
            values = output[2]
            prediction = values[:, -1]
            loss = loss_function(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            accumulated += float(loss.item())
            batches += 1
        train_losses.append(accumulated / max(batches, 1))

        model.eval()
        accumulated = 0.0
        batches = 0
        with torch.no_grad():
            for batch in validation_loader:
                inputs = {key: value.to(device) for key, value in batch.items() if key != "reward"}
                target = (batch["reward"].to(device) - reward_mean) / reward_std
                output = model(**inputs, apply_dynamic_mask=False)
                loss = loss_function(output[2][:, -1], target)
                accumulated += float(loss.item())
                batches += 1
        validation_losses.append(accumulated / max(batches, 1))

    return WarmupResult(
        reward_mean=reward_mean,
        reward_std=reward_std,
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
    )


def warmup_from_csv(
    model: Any,
    tokenizer: PreTrainedTokenizerFast,
    path: str | Path,
    **kwargs: Any,
) -> WarmupResult:
    """Load an oracle-scored trajectory table and call :func:`warmup_value_model`."""
    return warmup_value_model(model, tokenizer, pd.read_csv(path), **kwargs)
