"""PED-derived GeoDiff embedding-distance oracle.

Adapted from the public PED project maintained by Molecular Informatics:
https://github.com/molecularinformatics/PED/tree/main

This module is an adapted SynGenMol integration layer.  It does not distribute
GeoDiff source code, checkpoints, configuration files, model caches, or any
other third-party model asset.  Users must obtain and configure a compatible
PED/GeoDiff installation independently.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .base import MolecularOracle, sigmoid_score

LOGGER = logging.getLogger(__name__)


class PedGeoDiffOracle(MolecularOracle):
    """Embedding-distance similarity oracle following the PED GeoDiff workflow.

    A compatible GeoDiff package is intentionally a user-managed optional
    dependency.  The adapter loads it lazily, generates one deterministic RDKit
    conformer per valid SMILES, and then extracts the 2D, 3D, or concatenated
    representation from the pretrained GeoDiff encoder.
    """

    def __init__(
        self,
        reference_smiles: str,
        *,
        checkpoint_path: str | Path,
        embedding_mode: str = "3D",
        distance_metric: str = "euclidean",
        batch_size: int = 64,
        num_workers: int = 1,
        device: str = "auto",
        seed: int = 2021,
        transformation: dict[str, float] | None = None,
        invalid_score: float = 0.0,
    ) -> None:
        reference = Chem.MolFromSmiles(reference_smiles)
        if reference is None:
            raise ValueError(f"Invalid reference SMILES: {reference_smiles!r}")
        if embedding_mode not in {"2D", "3D", "concat", "concat_norm"}:
            raise ValueError("embedding_mode must be one of: 2D, 3D, concat, concat_norm")
        if distance_metric not in {"euclidean", "cosine"}:
            raise ValueError("distance_metric must be 'euclidean' or 'cosine'")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_workers < 1:
            raise ValueError("num_workers must be positive")

        self.reference_smiles = Chem.MolToSmiles(reference, canonical=True, isomericSmiles=True)
        self.checkpoint_path = Path(checkpoint_path)
        self.embedding_mode = embedding_mode
        self.distance_metric = distance_metric
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.device_spec = str(device)
        self.seed = int(seed)
        self.transformation = transformation
        self.invalid_score = float(invalid_score)
        self._backend: dict[str, Any] | None = None
        self._reference_embedding: np.ndarray | None = None

    @staticmethod
    def _canonicalize(smiles: str) -> str | None:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

    def _load(self) -> None:
        if self._backend is not None:
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"GeoDiff checkpoint not found: {self.checkpoint_path}. Obtain the checkpoint through the PED/GeoDiff workflow and provide checkpoint_path explicitly."
            )
        try:
            import torch
            import yaml
            from easydict import EasyDict
            from torch_geometric.data import Batch
            from torch_scatter import scatter_mean
            from geodiff.models.epsnet import get_model
            from geodiff.utils.datasets import rdmol_to_data
            from geodiff.utils.transforms import AddHigherOrderEdges, Compose, CountNodesPerGraph
        except ImportError as exc:
            raise ImportError(
                "PED-GeoDiff scoring requires a compatible user-installed GeoDiff environment. "
                "Follow the PED setup instructions in the README; the public SynGenMol repository does not bundle GeoDiff."
            ) from exc

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:  # Older PyTorch releases do not expose weights_only.
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        config_path = self._find_config_path()
        with config_path.open("r", encoding="utf-8") as handle:
            config = EasyDict(yaml.safe_load(handle))
        device = torch.device("cuda" if self.device_spec == "auto" and torch.cuda.is_available() else self.device_spec)
        if self.device_spec == "auto" and not torch.cuda.is_available():
            device = torch.device("cpu")
        model_config = checkpoint["config"].model if "config" in checkpoint else config.model
        model = get_model(model_config)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device).eval()
        transforms = Compose([CountNodesPerGraph(), AddHigherOrderEdges(order=config.model.edge_order)])
        self._backend = {
            "torch": torch,
            "Batch": Batch,
            "scatter_mean": scatter_mean,
            "rdmol_to_data": rdmol_to_data,
            "model": model,
            "transforms": transforms,
            "device": device,
        }
        reference_positions, reference_embeddings = self._embed_valid_smiles([self.reference_smiles])
        if reference_positions != [0] or len(reference_embeddings) != 1:
            raise ValueError(
                "Could not generate a GeoDiff conformer embedding for the reference SMILES."
            )
        self._reference_embedding = reference_embeddings[0]
        LOGGER.info("Loaded PED-GeoDiff checkpoint %s on %s", self.checkpoint_path, device)

    def _find_config_path(self) -> Path:
        # PED/GeoDiff checkpoints conventionally live under log/model/checkpoints;
        # the associated YAML is stored two levels above.  Search nearby rather
        # than assuming a local repository layout.
        candidates: list[Path] = []
        for parent in (self.checkpoint_path.parent, *self.checkpoint_path.parents[:3]):
            candidates.extend(sorted(parent.glob("*.yml")))
            candidates.extend(sorted(parent.glob("*.yaml")))
        if not candidates:
            raise FileNotFoundError(
                f"No GeoDiff YAML configuration was found near {self.checkpoint_path}. "
                "Keep the checkpoint with the configuration supplied by the compatible PED/GeoDiff release."
            )
        return candidates[0]

    def _embed_molecule(self, smiles: str) -> Chem.Mol | None:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        molecule = Chem.AddHs(molecule)
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = self.seed & 0xFFFFFFFF
        parameters.numThreads = self.num_workers
        parameters.enforceChirality = True
        if AllChem.EmbedMolecule(molecule, parameters) != 0:
            return None
        try:
            AllChem.UFFOptimizeMolecule(molecule, maxIters=200)
        except Exception:
            pass
        return molecule

    def _embed_valid_smiles(self, smiles: Sequence[str]) -> tuple[list[int], np.ndarray]:
        """Embed scoreable molecules and retain their positions in ``smiles``.

        RDKit conformer generation can fail for an otherwise valid product.  A
        failure should not abort a batch-level optimization step because the
        caller can assign that individual molecule ``invalid_score`` while
        scoring all other candidates.  The returned positions therefore map
        each embedding back to the corresponding input entry.
        """
        self._load()
        assert self._backend is not None
        backend = self._backend
        torch = backend["torch"]
        outputs: list[np.ndarray] = []
        embedded_positions: list[int] = []

        conformers: list[tuple[int, str, Chem.Mol]] = []
        for position, item in enumerate(smiles):
            molecule = self._embed_molecule(item)
            if molecule is None:
                LOGGER.warning("GeoDiff conformer generation failed for one valid SMILES; assigning invalid_score.")
                continue
            conformers.append((position, item, molecule))

        for start in range(0, len(conformers), self.batch_size):
            chunk = conformers[start : start + self.batch_size]
            data_list = []
            for _, smiles_item, molecule in chunk:
                data = backend["rdmol_to_data"](mol=molecule, smiles=smiles_item)
                data = backend["transforms"](data)
                if hasattr(data, "pos_ref") and data.pos_ref is not None:
                    data_input = data.clone()
                    data_input["pos_ref"] = None
                    data_input.pos = data.pos_ref[: int(data.num_nodes)]
                    data = data_input
                data_list.append(data)
            batch = backend["Batch"].from_data_list(data_list).to(backend["device"])
            model = backend["model"]
            time_step = torch.zeros(batch.num_graphs, dtype=torch.long, device=backend["device"])
            with torch.no_grad():
                _, _, edge_index, edge_type, edge_length, local_edge_mask = model(
                    atom_type=batch.atom_type,
                    pos=batch.pos,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    time_step=time_step,
                    return_edges=True,
                    extend_order=False,
                    extend_radius=True,
                )
                edge_attributes = model.edge_encoder_global(edge_length=edge_length, edge_type=edge_type)
                global_nodes = model.encoder_global(
                    z=batch.atom_type,
                    edge_index=edge_index,
                    edge_length=edge_length,
                    edge_attr=edge_attributes,
                )
                embedding_3d = backend["scatter_mean"](global_nodes, batch.batch, dim=0)
                local_nodes = model.encoder_local(
                    z=batch.atom_type,
                    edge_index=edge_index[:, local_edge_mask],
                    edge_attr=edge_attributes[local_edge_mask],
                )
                embedding_2d = backend["scatter_mean"](local_nodes, batch.batch, dim=0)

            if self.embedding_mode == "2D":
                embedding = embedding_2d
            elif self.embedding_mode == "3D":
                embedding = embedding_3d
            elif self.embedding_mode == "concat":
                embedding = torch.cat([embedding_2d, embedding_3d], dim=1)
            else:
                embedding = torch.cat(
                    [
                        torch.nn.functional.normalize(embedding_2d, dim=1),
                        torch.nn.functional.normalize(embedding_3d, dim=1),
                    ],
                    dim=1,
                )
            outputs.append(embedding.detach().cpu().numpy().astype(np.float32))
            embedded_positions.extend(position for position, _, _ in chunk)
        return embedded_positions, (np.vstack(outputs) if outputs else np.empty((0, 0), dtype=np.float32))

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
            canonical = self._canonicalize(str(item))
            if canonical is None:
                continue
            valid_positions.append(index)
            valid_smiles.append(canonical)
        if not valid_smiles:
            return scores

        embedded_positions, embeddings = self._embed_valid_smiles(valid_smiles)
        for local_position, embedding in zip(embedded_positions, embeddings):
            index = valid_positions[local_position]
            value = self._distance(embedding)
            scores[index] = sigmoid_score(value, **self.transformation) if self.transformation else value
        return scores
