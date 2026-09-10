"""Command-line scoring for an existing table of molecular SMILES."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from syngenmol.config import load_yaml_mapping
from syngenmol.oracles import build_oracle, require_aligned_scores


def _load_oracle_config(path: str | Path) -> dict[str, Any]:
    configuration = load_yaml_mapping(path, description="Oracle configuration")
    if "oracle" in configuration:
        configuration = configuration["oracle"]
    if not isinstance(configuration, dict):
        raise ValueError("Oracle configuration must be a mapping or contain an 'oracle' mapping")
    return configuration


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("score", help="Score a CSV table of molecules using a configured oracle.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    parser.add_argument("--oracle-config", required=True, help="YAML oracle configuration.")
    parser.add_argument("--smiles-column", default="smiles", help="Column containing molecular SMILES.")
    parser.add_argument("--score-column", default="score", help="Name of the output score column.")
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    table = pd.read_csv(args.input)
    if args.smiles_column not in table.columns:
        raise ValueError(
            f"Input file has no SMILES column {args.smiles_column!r}. "
            f"Available columns: {table.columns.tolist()}"
        )
    oracle = build_oracle(_load_oracle_config(args.oracle_config))
    smiles = table[args.smiles_column].fillna("").astype(str).tolist()
    table[args.score_column] = require_aligned_scores(smiles, oracle.score_smiles(smiles))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Scored {len(table):,} molecules and wrote {output}")
    return 0
