"""Configuration-driven construction of supported public scoring oracles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import MolecularOracle


def build_oracle(config: Mapping[str, Any]) -> MolecularOracle:
    """Construct a supported oracle without importing optional dependencies early."""
    config = dict(config)
    name = str(config.pop("name", config.pop("type", ""))).strip().lower()
    if name in {"tanimoto", "ecfp_tanimoto", "tanimoto_similarity"}:
        from .tanimoto import TanimotoSimilarityOracle

        return TanimotoSimilarityOracle(**config)
    if name in {"ped_molformer", "molformer", "ped-molformer"}:
        from .ped_molformer import PedMolFormerOracle

        return PedMolFormerOracle(**config)
    if name in {"ped_geodiff", "geodiff", "ped-geodiff"}:
        from .ped_geodiff import PedGeoDiffOracle

        return PedGeoDiffOracle(**config)
    raise ValueError(
        f"Unsupported public oracle {name!r}. Use tanimoto, ped_molformer, ped_geodiff, "
        "or pass a custom MolecularOracle implementation directly to the Python API."
    )
