"""Deterministic building-block preprocessing and compatibility-cache generation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from rdkit import Chem, rdBase

from syngenmol.chemistry.reactions import ReactionTemplate, load_reaction_templates, write_reaction_templates
from syngenmol.tokenizer import BuildingBlockTokenizer


@dataclass(frozen=True)
class PreparedSearchSpace:
    """In-memory representation of a prepared SynGenMol design space."""

    building_blocks: pd.DataFrame
    tokenizer: BuildingBlockTokenizer
    templates: tuple[ReactionTemplate, ...]
    compatibility: dict[int, dict[int, tuple[int, ...]]]

    # These views are cached because they are derived from immutable fields and
    # are read inside sampling loops.  Rebuilding `token_to_mol` on every access
    # would re-parse the whole catalog per lookup, which makes route sampling
    # quadratic in catalog size.
    @cached_property
    def building_block_ids(self) -> tuple[int, ...]:
        return tuple(self.tokenizer.token_to_id[name] for name in self.building_blocks["name"])

    @cached_property
    def token_to_mol(self) -> dict[int, Chem.Mol]:
        return {
            self.tokenizer.token_to_id[row.name]: Chem.MolFromSmiles(row.smiles)
            for row in self.building_blocks.itertuples(index=False)
        }

    @cached_property
    def token_to_smiles(self) -> dict[int, str]:
        return {
            self.tokenizer.token_to_id[row.name]: row.smiles
            for row in self.building_blocks.itertuples(index=False)
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_catalog(
    input_path: str | Path,
    *,
    smiles_column: str,
    id_column: str,
) -> pd.DataFrame:
    input_path = Path(input_path)
    suffixes = "".join(input_path.suffixes).lower()
    if suffixes.endswith(".sdf") or suffixes.endswith(".sdf.gz"):
        supplier = Chem.ForwardSDMolSupplier(str(input_path), removeHs=False)
        rows: list[dict[str, str | None]] = []
        for index, molecule in enumerate(supplier):
            if molecule is None:
                rows.append({"name": None, "smiles": None, "source_row": index})
                continue
            identifier = molecule.GetProp(id_column) if molecule.HasProp(id_column) else None
            rows.append(
                {
                    "name": identifier,
                    "smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
                    "source_row": index,
                }
            )
        return pd.DataFrame(rows)

    if suffixes.endswith(".tsv") or suffixes.endswith(".tab"):
        table = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
    else:
        # Python's CSV sniffer provides a friendly path for user-provided CSV
        # catalogs that have an uncommon delimiter but a conventional extension.
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            sample = handle.read(8192)
        try:
            delimiter = csv.Sniffer().sniff(sample).delimiter
        except csv.Error:
            delimiter = ","
        table = pd.read_csv(input_path, sep=delimiter, dtype=str, keep_default_na=False)

    missing = [column for column in (smiles_column, id_column) if column not in table.columns]
    if missing:
        raise ValueError(
            f"Catalog {input_path} is missing required columns {missing}. "
            f"Available columns: {table.columns.tolist()}"
        )
    return pd.DataFrame(
        {
            "name": table[id_column],
            "smiles": table[smiles_column],
            "source_row": table.index,
        }
    )


def _canonicalize_building_blocks(
    raw: pd.DataFrame,
    *,
    reject_multicomponent: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    observed_names: set[str] = set()

    for row in raw.itertuples(index=False):
        name = "" if row.name is None else str(row.name).strip()
        smiles = "" if row.smiles is None else str(row.smiles).strip()
        reason: str | None = None
        canonical_smiles: str | None = None

        if not name:
            reason = "missing_identifier"
        elif any(character.isspace() for character in name):
            reason = "identifier_contains_whitespace"
        elif name in observed_names:
            reason = "duplicate_identifier"
        elif name in {"<pad>", "<unk>", "<s>", "</s>"}:
            reason = "identifier_collides_with_special_token"
        elif not smiles:
            reason = "missing_smiles"
        elif reject_multicomponent and "." in smiles:
            reason = "multicomponent_smiles"
        else:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                reason = "invalid_smiles"
            else:
                try:
                    Chem.SanitizeMol(molecule)
                    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
                except Exception:
                    reason = "unsanitizable_smiles"

        if reason is not None:
            rejected.append(
                {
                    "source_row": int(row.source_row),
                    "name": name,
                    "smiles": smiles,
                    "reason": reason,
                }
            )
            continue

        observed_names.add(name)
        accepted.append(
            {
                "name": name,
                "smiles": canonical_smiles,
                "source_row": int(row.source_row),
            }
        )

    return pd.DataFrame(accepted, columns=["name", "smiles", "source_row"]), pd.DataFrame(
        rejected, columns=["source_row", "name", "smiles", "reason"]
    )


def build_compatibility(
    building_blocks: pd.DataFrame,
    tokenizer: BuildingBlockTokenizer,
    templates: Iterable[ReactionTemplate],
) -> dict[int, dict[int, tuple[int, ...]]]:
    """Compute compatibility by SMARTS substructure matching only.

    The procedure is deterministic and uses no learned chemical-validity model
    or route-pretraining data.  It records only building-block token IDs.
    """
    token_molecules: list[tuple[int, Chem.Mol]] = []
    for row in building_blocks.itertuples(index=False):
        molecule = Chem.MolFromSmiles(row.smiles)
        if molecule is None:  # Defensive: preparation already validates this.
            raise ValueError(f"Prepared building block is invalid: {row.name}")
        token_molecules.append((tokenizer.token_to_id[row.name], molecule))

    compatibility: dict[int, dict[int, tuple[int, ...]]] = {}
    for template in templates:
        by_position: dict[int, tuple[int, ...]] = {}
        for position in (0, 1):
            ids = [
                token_id
                for token_id, molecule in token_molecules
                if template.matches_reactant(molecule, position)
            ]
            by_position[position] = tuple(sorted(ids))
        compatibility[template.identifier] = by_position
    return compatibility


def _active_templates(
    templates: Iterable[ReactionTemplate],
    compatibility: Mapping[int, Mapping[int, Iterable[int]]],
) -> list[ReactionTemplate]:
    return [
        template
        for template in templates
        if tuple(compatibility[template.identifier].get(0, ()))
        and tuple(compatibility[template.identifier].get(1, ()))
    ]


def prepare_search_space(
    *,
    input_path: str | Path,
    reaction_template_path: str | Path,
    output_dir: str | Path,
    smiles_column: str = "smiles",
    id_column: str = "name",
    reject_multicomponent: bool = True,
) -> PreparedSearchSpace:
    """Validate a catalog and create a portable prepared search-space directory."""
    input_path = Path(input_path)
    reaction_template_path = Path(reaction_template_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _read_catalog(input_path, smiles_column=smiles_column, id_column=id_column)
    accepted, rejected = _canonicalize_building_blocks(
        raw, reject_multicomponent=reject_multicomponent
    )
    if accepted.empty:
        raise ValueError("No valid building blocks remain after preprocessing")

    tokenizer = BuildingBlockTokenizer(accepted["name"].tolist())
    all_templates = load_reaction_templates(reaction_template_path)
    all_compatibility = build_compatibility(accepted, tokenizer, all_templates)
    active_original_templates = _active_templates(all_templates, all_compatibility)
    if not active_original_templates:
        raise ValueError("No reaction templates have compatible building blocks in both positions")

    # Reassign IDs after filtering so the saved template file and compatibility
    # map have a direct one-to-one, contiguous correspondence.
    active_templates = [
        ReactionTemplate.from_smarts(identifier=index, smarts=template.smarts)
        for index, template in enumerate(active_original_templates)
    ]
    compatibility = build_compatibility(accepted, tokenizer, active_templates)

    accepted[["name", "smiles"]].to_csv(output_dir / "building_blocks.csv", index=False)
    rejected.to_csv(output_dir / "rejected_records.csv", index=False)
    tokenizer.save(output_dir / "tokenizer")
    write_reaction_templates(active_templates, output_dir / "reaction_templates.smi")

    compatibility_payload = {
        str(reaction_id): {str(position): list(token_ids) for position, token_ids in by_position.items()}
        for reaction_id, by_position in compatibility.items()
    }
    (output_dir / "reaction_compatibility.json").write_text(
        json.dumps(compatibility_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    duplicate_structures = int(accepted["smiles"].duplicated(keep=False).sum())
    manifest = {
        "format_version": 1,
        "input_catalog": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "smiles_column": smiles_column,
            "id_column": id_column,
            "rows_read": int(len(raw)),
        },
        "reaction_templates": {
            "source_path": str(reaction_template_path),
            "source_sha256": sha256_file(reaction_template_path),
            "templates_read": int(len(all_templates)),
            "templates_retained": int(len(active_templates)),
            "prepared_sha256": sha256_file(output_dir / "reaction_templates.smi"),
        },
        "preprocessing": {"reject_multicomponent": bool(reject_multicomponent)},
        "building_blocks": {
            "accepted": int(len(accepted)),
            "rejected": int(len(rejected)),
            "duplicate_canonical_smiles_rows": duplicate_structures,
            "duplicate_structures_retained": True,
        },
        "software": {"rdkit_version": rdBase.rdkitVersion},
        "outputs": {
            "building_blocks": "building_blocks.csv",
            "tokenizer": "tokenizer/",
            "reaction_templates": "reaction_templates.smi",
            "compatibility": "reaction_compatibility.json",
            "rejected_records": "rejected_records.csv",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return PreparedSearchSpace(
        building_blocks=accepted[["name", "smiles"]].copy(),
        tokenizer=tokenizer,
        templates=tuple(active_templates),
        compatibility=compatibility,
    )


def load_prepared_search_space(directory: str | Path) -> PreparedSearchSpace:
    """Load assets created by :func:`prepare_search_space`."""
    directory = Path(directory)
    building_blocks = pd.read_csv(directory / "building_blocks.csv", dtype=str)
    if list(building_blocks.columns) != ["name", "smiles"]:
        raise ValueError("Prepared building_blocks.csv must contain exactly name,smiles columns")
    tokenizer = BuildingBlockTokenizer.load(directory / "tokenizer")
    templates = tuple(load_reaction_templates(directory / "reaction_templates.smi"))
    payload = json.loads((directory / "reaction_compatibility.json").read_text(encoding="utf-8"))
    compatibility = {
        int(reaction_id): {
            int(position): tuple(int(token_id) for token_id in token_ids)
            for position, token_ids in by_position.items()
        }
        for reaction_id, by_position in payload.items()
    }
    expected_ids = {template.identifier for template in templates}
    if set(compatibility) != expected_ids:
        raise ValueError("Reaction compatibility cache does not match prepared template identifiers")
    return PreparedSearchSpace(building_blocks, tokenizer, templates, compatibility)
