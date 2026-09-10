"""Project-specific building-block constraints for SynGenMol runs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rdkit import Chem

from syngenmol.chemistry.search_space import PreparedSearchSpace


def select_building_blocks_by_smarts(
    search_space: PreparedSearchSpace,
    smarts: str,
) -> set[int]:
    """Return prepared building-block token IDs matching a project SMARTS query.

    The result can be passed as ``initial_building_block_ids`` when a design
    requirement applies to the first selected building block.  The constraint
    changes the task-specific search space and therefore should be recorded in
    the run configuration and manifest.
    """
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise ValueError(f"Invalid SMARTS constraint: {smarts!r}")
    selected: set[int] = set()
    for row in search_space.building_blocks.itertuples(index=False):
        molecule = Chem.MolFromSmiles(row.smiles)
        if molecule is not None and molecule.HasSubstructMatch(query, useChirality=True):
            selected.add(search_space.tokenizer.token_to_id[row.name])
    return selected


def load_building_block_names(path: str | Path) -> tuple[str, ...]:
    """Load one exact building-block identifier per non-comment text line.

    This supports reproducible first-token constraint sets without embedding a
    long identifier list in a YAML configuration. Blank lines and text after a
    ``#`` comment marker are ignored; duplicate identifiers are retained only
    once in their first-seen order.
    """
    names: list[str] = []
    seen: set[str] = set()
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        name = raw_line.split("#", 1)[0].strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    if not names:
        raise ValueError(f"No building-block identifiers were found in {path}")
    return tuple(names)


def select_building_blocks_by_names(
    search_space: PreparedSearchSpace,
    names: Iterable[str],
) -> set[int]:
    """Resolve an explicit user-maintained list of building-block identifiers."""
    available = search_space.tokenizer.token_to_id
    requested = {str(name) for name in names}
    missing = sorted(requested - set(available))
    if missing:
        preview = ", ".join(repr(name) for name in missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"Constraint names are absent from the prepared search space: {preview}{suffix}")
    return {available[name] for name in requested}
