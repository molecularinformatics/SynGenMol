"""Base interfaces shared by SynGenMol scoring oracles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
import math


class MolecularOracle(ABC):
    """Map a batch of SMILES strings to finite, higher-is-better scores."""

    @abstractmethod
    def score_smiles(self, smiles: Sequence[str]) -> list[float]:
        """Return one finite score for each input SMILES, in the same order."""

    def __call__(self, smiles: Sequence[str]) -> list[float]:
        return self.score_smiles(smiles)


def sigmoid_score(value: float, *, high: float, low: float, k: float) -> float:
    """Apply the sigmoid score transform used by the PED study configurations.

    This formulation is equivalent to the transformation used in the original
    molecular-generation scoring configuration.  Parameters are exposed rather
    than fixed because the numerical scale of a score is oracle-dependent.
    """
    if high == low:
        raise ValueError("Sigmoid transformation requires distinct high and low values")
    exponent = 10.0 * float(k) * (float(value) - (float(low) + float(high)) * 0.5)
    exponent /= float(low) - float(high)
    try:
        return float(1.0 / (1.0 + math.pow(10.0, exponent)))
    except OverflowError:
        return 0.0 if exponent > 0 else 1.0


def require_aligned_scores(smiles: Sequence[str], scores: Sequence[float]) -> list[float]:
    """Validate a scorer result before it enters the optimization loop."""
    if len(smiles) != len(scores):
        raise ValueError(f"Oracle returned {len(scores)} scores for {len(smiles)} molecules")
    checked: list[float] = []
    for index, score in enumerate(scores):
        score = float(score)
        if not math.isfinite(score):
            raise ValueError(f"Oracle returned a non-finite score at position {index}: {score!r}")
        checked.append(score)
    return checked
