"""PED-derived MoLFormer embedding-distance oracle.

Adapted from the public PED project maintained by Molecular Informatics:
https://github.com/molecularinformatics/PED/tree/main

This module is an adapted SynGenMol integration layer.  It does not distribute
MoLFormer weights, tokenizer files, Hugging Face caches, or other model assets.
Users must obtain those assets from their original source and comply with the
applicable licenses and terms.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from .base import MolecularOracle, sigmoid_score

LOGGER = logging.getLogger(__name__)


class PedMolFormerOracle(MolecularOracle):
    """Embedding-distance similarity oracle based on the PED MoLFormer setup.

    The model and tokenizer are loaded lazily on the first scoring call.  This
    prevents a core SynGenMol installation from importing Transformers or
    downloading model assets unless the optional oracle is explicitly selected.
    """

    def __init__(
        self,
        reference_smiles: str,
        *,
        model_id: str = "ibm/MoLFormer-XL-both-10pct",
        cache_dir: str | Path | None = None,
        revision: str | None = None,
        distance_metric: str = "euclidean",
        batch_size: int = 64,
        device: str = "auto",
        offline: bool = False,
        transformation: dict[str, float] | None = None,
        invalid_score: float = 0.0,
        trust_remote_code: bool = True,
    ) -> None:
        reference = Chem.MolFromSmiles(reference_smiles)
        if reference is None:
            raise ValueError(f"Invalid reference SMILES: {reference_smiles!r}")
        if distance_metric not in {"euclidean", "cosine"}:
            raise ValueError("distance_metric must be 'euclidean' or 'cosine'")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.reference_smiles = Chem.MolToSmiles(reference, canonical=True, isomericSmiles=True)
        self.model_id = str(model_id)
        self.cache_dir = None if cache_dir is None else str(cache_dir)
        self.revision = revision
        self.distance_metric = distance_metric
        self.batch_size = int(batch_size)
        self.device_spec = str(device)
        self.offline = bool(offline)
        self.transformation = transformation
        self.invalid_score = float(invalid_score)
        self.trust_remote_code = bool(trust_remote_code)
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None
        self._reference_embedding: np.ndarray | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "PED-MolFormer scoring requires the optional dependencies. "
                "Install the SynGenMol PED extra and obtain the MoLFormer model assets as described in the README."
            ) from exc

        device = torch.device("cuda" if self.device_spec == "auto" and torch.cuda.is_available() else self.device_spec)
        if self.device_spec == "auto" and not torch.cuda.is_available():
            device = torch.device("cpu")
        kwargs = {
            "cache_dir": self.cache_dir,
            "revision": self.revision,
            "local_files_only": self.offline,
            "trust_remote_code": self.trust_remote_code,
        }
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
            model = AutoModel.from_pretrained(self.model_id, **kwargs).to(device).eval()
        except Exception as exc:
            location = f" in cache directory {self.cache_dir!r}" if self.cache_dir else ""
            raise RuntimeError(
                f"Could not load PED-MolFormer model {self.model_id!r}{location}. "
                "Download or configure the model assets according to the README, or set offline=false for an allowed download."
            ) from exc

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._reference_embedding = self._embed_valid_smiles([self.reference_smiles])[0]
        LOGGER.info("Loaded PED-MolFormer model %s on %s", self.model_id, device)

    def _embed_valid_smiles(self, smiles: Sequence[str]) -> np.ndarray:
        self._load()
        assert self._tokenizer is not None and self._model is not None and self._device is not None
        import torch

        outputs: list[np.ndarray] = []
        for start in range(0, len(smiles), self.batch_size):
            batch = list(smiles[start : start + self.batch_size])
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                hidden = self._model(**inputs).last_hidden_state
            attention = inputs["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
            pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp_min(1)
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.empty((0, 0), dtype=np.float32)

    def _distance(self, embedding: np.ndarray) -> float:
        assert self._reference_embedding is not None
        if self.distance_metric == "euclidean":
            return float(np.linalg.norm(embedding - self._reference_embedding))
        denominator = float(np.linalg.norm(embedding) * np.linalg.norm(self._reference_embedding))
        return 1.0 if denominator == 0.0 else float(1.0 - np.dot(embedding, self._reference_embedding) / denominator)

    def score_smiles(self, smiles: Sequence[str]) -> list[float]:
        valid_positions: list[int] = []
        valid_smiles: list[str] = []
        scores = [self.invalid_score] * len(smiles)
        for index, item in enumerate(smiles):
            molecule = Chem.MolFromSmiles(str(item))
            if molecule is None:
                continue
            valid_positions.append(index)
            valid_smiles.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
        if not valid_smiles:
            return scores

        embeddings = self._embed_valid_smiles(valid_smiles)
        for index, embedding in zip(valid_positions, embeddings):
            value = self._distance(embedding)
            scores[index] = sigmoid_score(value, **self.transformation) if self.transformation else value
        return scores
