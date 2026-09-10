"""Lightweight RDKit ECFP/Tanimoto similarity oracle."""

from __future__ import annotations

from collections.abc import Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from .base import MolecularOracle, sigmoid_score


class TanimotoSimilarityOracle(MolecularOracle):
    """Score query molecules by Morgan-fingerprint similarity to one reference."""

    def __init__(
        self,
        reference_smiles: str,
        *,
        radius: int = 3,
        n_bits: int = 2048,
        use_features: bool = True,
        transformation: dict[str, float] | None = None,
        invalid_score: float = 0.0,
    ) -> None:
        reference = Chem.MolFromSmiles(reference_smiles)
        if reference is None:
            raise ValueError(f"Invalid reference SMILES: {reference_smiles!r}")
        self.radius = int(radius)
        self.n_bits = int(n_bits)
        self.use_features = bool(use_features)
        self.transformation = transformation
        self.invalid_score = float(invalid_score)
        self._reference_fingerprint = self._fingerprint(reference)

    def _fingerprint(self, molecule: Chem.Mol):
        return AllChem.GetMorganFingerprintAsBitVect(
            molecule,
            radius=self.radius,
            nBits=self.n_bits,
            useFeatures=self.use_features,
        )

    def score_smiles(self, smiles: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for item in smiles:
            molecule = Chem.MolFromSmiles(str(item))
            if molecule is None:
                scores.append(self.invalid_score)
                continue
            score = float(DataStructs.TanimotoSimilarity(self._reference_fingerprint, self._fingerprint(molecule)))
            if self.transformation:
                score = sigmoid_score(score, **self.transformation)
            scores.append(score)
        return scores
