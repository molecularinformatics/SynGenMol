"""Command-line interface for deterministic SynGenMol search-space preparation."""

from __future__ import annotations

import argparse

from syngenmol.chemistry.search_space import prepare_search_space


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "prepare-search-space",
        help="Validate a building-block catalog and create tokenizer and compatibility assets.",
    )
    parser.add_argument("--input", required=True, help="Input CSV, TSV, or SDF catalog.")
    parser.add_argument(
        "--smiles-column",
        default="smiles",
        help="SMILES column for CSV/TSV input. Ignored for SDF input.",
    )
    parser.add_argument(
        "--id-column",
        default="name",
        help="Identifier column for CSV/TSV input or property name for SDF input.",
    )
    parser.add_argument(
        "--reaction-templates",
        required=True,
        help="Path to a file containing one two-reactant SMARTS template per line.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for prepared search-space assets.")
    parser.add_argument(
        "--allow-multicomponent",
        action="store_true",
        help="Retain dot-disconnected catalog entries instead of rejecting them.",
    )
    parser.set_defaults(handler=run)
    return parser


def run(args: argparse.Namespace) -> int:
    search_space = prepare_search_space(
        input_path=args.input,
        reaction_template_path=args.reaction_templates,
        output_dir=args.output_dir,
        smiles_column=args.smiles_column,
        id_column=args.id_column,
        reject_multicomponent=not args.allow_multicomponent,
    )
    print(f"Prepared {len(search_space.building_blocks):,} building blocks.")
    print(f"Retained {len(search_space.templates):,} two-reactant templates.")
    print(f"Wrote prepared search space to: {args.output_dir}")
    return 0
