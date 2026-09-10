"""Shared YAML configuration loading for SynGenMol command-line workflows."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_UNRESOLVED_ENVIRONMENT_REFERENCE = re.compile(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


def _expand_environment(value: Any) -> Any:
    """Recursively expand shell-style environment variables in YAML strings."""
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if _UNRESOLVED_ENVIRONMENT_REFERENCE.search(expanded):
            raise ValueError(
                "Configuration contains an unresolved environment-variable reference: "
                f"{value!r}. Export the variable before running SynGenMol."
            )
        return expanded
    return value


def load_yaml_mapping(path: str | Path, *, description: str) -> dict[str, Any]:
    """Load a YAML mapping and expand environment variables in string values."""
    with Path(path).open("r", encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle) or {}
    if not isinstance(configuration, dict):
        raise ValueError(f"{description} must be a YAML mapping")
    return _expand_environment(configuration)
