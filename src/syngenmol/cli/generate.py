"""Generate and score valid SynGenMol molecular-assembly trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rdkit import Chem

from syngenmol.chemistry.constraints import (
    load_building_block_names,
    select_building_blocks_by_names,
    select_building_blocks_by_smarts,
)
from syngenmol.config import load_yaml_mapping
from syngenmol.chemistry.search_space import load_prepared_search_space
from syngenmol.models import build_policy_model, load_policy_checkpoint
from syngenmol.oracles import build_oracle, require_aligned_scores


def _load_yaml(path: str | Path) -> dict[str, Any]:
    return load_yaml_mapping(path, description="Generation configuration")


def _resolve_initial_constraint(search_space, constraint: dict[str, Any] | None) -> set[int] | None:
    if not constraint:
        return None
    selected: set[int] = set()
    if "smarts" in constraint:
        selected.update(select_building_blocks_by_smarts(search_space, str(constraint["smarts"])))
    if "names" in constraint:
        selected.update(select_building_blocks_by_names(search_space, constraint["names"]))
    if "names_file" in constraint:
        selected.update(
            select_building_blocks_by_names(
                search_space,
                load_building_block_names(str(constraint["names_file"])),
            )
        )
    if not selected:
        raise ValueError("The initial building-block constraint selected no prepared building blocks")
    return selected


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("generate", help="Generate and score valid molecular-assembly trajectories.")
    parser.add_argument("--config", required=True, help="YAML generation configuration.")
    parser.add_argument("--output", default=None, help="Override the output CSV path in the configuration.")
    parser.set_defaults(handler=run)
    return parser


def _rows_from_trajectories(trajectories, tokenizer) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trajectory in trajectories:
        token_ids = trajectory.token_ids.detach().cpu().tolist()
        rows.append(
            {
                "name": " ".join(tokenizer.convert_ids_to_tokens(token_ids)),
                "token_ids": json.dumps(token_ids),
                "reaction_ids": json.dumps(list(trajectory.reaction_ids)),
                "product_smiles": Chem.MolToSmiles(
                    trajectory.product,
                    canonical=True,
                    isomericSmiles=True,
                ),
                "entropy": float(trajectory.mean_entropy.detach().cpu()),
            }
        )
    return rows


def run(args: argparse.Namespace) -> int:
    config = _load_yaml(args.config)
    generation = dict(config.get("generation", {}))
    num_molecules = int(generation.get("num_molecules", 100))
    max_reactions = int(generation.get("max_reactions", 1))
    if num_molecules < 1 or max_reactions < 1:
        raise ValueError("generation.num_molecules and generation.max_reactions must be positive")

    seed = generation.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
    search_space = load_prepared_search_space(config["search_space"])
    initial_ids = _resolve_initial_constraint(search_space, config.get("initial_building_block_constraint"))
    model_config = dict(config.get("model", {}))
    policy = build_policy_model(search_space, initial_building_block_ids=initial_ids, **model_config)

    checkpoint = config.get("checkpoint")
    if checkpoint:
        load_policy_checkpoint(policy, checkpoint, map_location="cpu")
    device_name = str(config.get("device", "auto"))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    if device_name == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    policy = policy.to(device).eval()

    rows: list[dict[str, object]] = []
    seen_products: set[str] = set()
    attempts = 0
    max_attempts = max(100, num_molecules * 100)
    while len(rows) < num_molecules and attempts < max_attempts:
        trajectories = policy.generate_trajectory_branches(
            max_reactions=max_reactions,
            temperature=float(generation.get("temperature", 1.0)),
            top_p=float(generation.get("top_p", 1.0)),
            epsilon=float(generation.get("epsilon", 0.0)),
            device=device,
        )
        for row in _rows_from_trajectories(trajectories, search_space.tokenizer.tokenizer):
            product_smiles = str(row["product_smiles"])
            if product_smiles not in seen_products:
                seen_products.add(product_smiles)
                rows.append(row)
                if len(rows) >= num_molecules:
                    break
        attempts += 1

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No valid molecular products were generated")
    if len(table) < num_molecules:
        print(
            f"Generated {len(table):,} unique products after {attempts:,} sampling attempts, "
            f"fewer than the requested {num_molecules:,}."
        )
    oracle_config = config.get("oracle")
    if oracle_config:
        if not isinstance(oracle_config, dict):
            raise ValueError("oracle must be a mapping when provided")
        oracle = build_oracle(oracle_config)
        table["score"] = require_aligned_scores(
            table["product_smiles"].tolist(),
            oracle.score_smiles(table["product_smiles"].tolist()),
        )
        table = table.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)

    output = Path(args.output or config.get("output", "outputs/generated_molecules.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Wrote {len(table):,} unique generated products to {output}")
    if "score" in table:
        print(f"Score range: {table['score'].min():.4f} to {table['score'].max():.4f}")
    return 0
