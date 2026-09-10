"""Prepare valid, optionally oracle-scored trajectories for task-specific warm-up."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase

from syngenmol.chemistry.constraints import load_building_block_names, select_building_blocks_by_names
from syngenmol.chemistry.search_space import PreparedSearchSpace, load_prepared_search_space
from syngenmol.config import load_yaml_mapping
from syngenmol.models import build_policy_model
from syngenmol.oracles import build_oracle, require_aligned_scores


def _load_oracle_config(path: str | Path) -> dict[str, Any]:
    configuration = load_yaml_mapping(path, description="Oracle configuration")
    configuration = configuration.get("oracle", configuration)
    if not isinstance(configuration, dict):
        raise ValueError("Oracle configuration must be a mapping or contain an 'oracle' mapping")
    return configuration


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "prepare-warmup",
        help="Prepare template-feasible warm-up trajectories and optionally score their products.",
    )
    parser.add_argument("--search-space", required=True, help="Prepared search-space directory.")
    parser.add_argument(
        "--oracle-config",
        default=None,
        help="Optional YAML oracle configuration. Required unless --skip-scoring is used.",
    )
    parser.add_argument("--output", required=True, help="Output CSV for prepared trajectories.")
    parser.add_argument("--num-trajectories", type=int, default=10_000, help="Target number of unique products.")
    parser.add_argument("--max-reactions", type=int, default=1, help="Maximum virtual reaction depth.")
    parser.add_argument(
        "--initial-building-block-names-file",
        default=None,
        help=(
            "Optional text file containing one prepared building-block identifier per line. "
            "When supplied, restrict the first sampled building-block token to this exact set."
        ),
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("policy", "random-balanced"),
        default="policy",
        help=(
            "Trajectory sampler. 'policy' uses a randomly initialized masked policy; "
            "'random-balanced' directly samples valid one-step template routes while reducing "
            "the probability of building blocks that have already been used."
        ),
    )
    parser.add_argument(
        "--usage-penalty",
        type=float,
        default=1.0,
        help=(
            "For random-balanced sampling, use weight (1 + prior uses)^(-usage-penalty) "
            "when selecting each building block (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--max-building-block-uses",
        type=int,
        default=20,
        help=(
            "For random-balanced sampling, do not include a building block in more than this "
            "many accepted trajectories across either route position (default: 20)."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum route proposals before returning fewer than the requested trajectories.",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Write an unscored trajectory table. It can be evaluated later with `syngenmol score`.",
    )
    # Policy-sampler settings are retained for compatibility with the existing
    # recorded demonstration. The direct random-balanced sampler does not build
    # or evaluate a language model.
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed for reproducible sampling.")
    parser.add_argument("--device", default="auto", help="Torch device, or 'auto', for policy sampling.")
    parser.set_defaults(handler=run)
    return parser


def _rows_from_trajectories(trajectories, tokenizer) -> list[dict[str, object]]:
    """Convert masked-policy rollouts into the common warm-up table schema."""
    rows: list[dict[str, object]] = []
    for trajectory in trajectories:
        token_ids = trajectory.token_ids.detach().cpu().tolist()
        names = [str(name) for name in tokenizer.convert_ids_to_tokens(token_ids)]
        rows.append(
            {
                "name": " ".join(names),
                "token_ids": json.dumps(token_ids),
                "reaction_ids": json.dumps(list(trajectory.reaction_ids)),
                "product_smiles": Chem.MolToSmiles(trajectory.product, canonical=True, isomericSmiles=True),
                "entropy": float(trajectory.mean_entropy.detach().cpu()),
                "sampling_mode": "policy",
                "first_building_block": names[0] if names else "",
                "second_building_block": names[1] if len(names) > 1 else "",
            }
        )
    return rows


def _sample_unique_policy_rows(
    model,
    search_space: PreparedSearchSpace,
    *,
    count: int,
    max_reactions: int,
    temperature: float,
    top_p: float,
    epsilon: float,
    device: torch.device,
    max_attempts: int,
) -> list[dict[str, object]]:
    """Sample unique masked-policy trajectories, discarding empty proposals."""
    rows: list[dict[str, object]] = []
    seen_products: set[str] = set()
    empty_proposals = 0
    attempts = 0
    while len(rows) < count and attempts < max_attempts:
        trajectories = model.generate_trajectory_branches(
            max_reactions=max_reactions,
            temperature=temperature,
            top_p=top_p,
            epsilon=epsilon,
            device=device,
        )
        attempts += 1
        if not trajectories:
            empty_proposals += 1
            continue
        for row in _rows_from_trajectories(trajectories, search_space.tokenizer.tokenizer):
            product_smiles = str(row["product_smiles"])
            if product_smiles in seen_products:
                continue
            seen_products.add(product_smiles)
            rows.append(row)
            if len(rows) >= count:
                break
    if empty_proposals:
        print(f"Policy sampler discarded {empty_proposals:,} empty route proposals.", flush=True)
    print(
        f"Policy sampler retained {len(rows):,} unique products after {attempts:,} proposals.",
        flush=True,
    )
    return rows


def _inverse_frequency_weights(
    token_ids: list[int],
    uses: dict[int, int],
    *,
    penalty: float,
) -> np.ndarray:
    """Return normalized inverse-use weights for a nonempty token collection."""
    values = np.asarray([1.0 / (1.0 + uses[token_id]) ** penalty for token_id in token_ids])
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("Could not construct finite random-sampling weights")
    return values / total


def _weighted_token_choice(
    token_ids: list[int],
    uses: dict[int, int],
    *,
    penalty: float,
    rng: np.random.Generator,
) -> int:
    return int(rng.choice(token_ids, p=_inverse_frequency_weights(token_ids, uses, penalty=penalty)))


def _first_reactions(search_space: PreparedSearchSpace) -> dict[int, tuple[int, ...]]:
    """Map each possible first building-block token to feasible reaction IDs."""
    index: defaultdict[int, list[int]] = defaultdict(list)
    for reaction_id, by_position in search_space.compatibility.items():
        if not by_position[1]:
            continue
        for token_id in by_position[0]:
            index[int(token_id)].append(int(reaction_id))
    return {token_id: tuple(sorted(reaction_ids)) for token_id, reaction_ids in index.items()}


def _eligible_partners(
    *,
    first_token_id: int,
    partner_ids: tuple[int, ...],
    uses: dict[int, int],
    max_building_block_uses: int,
) -> list[int]:
    """Exclude partners that would exceed the accepted-trajectory use cap."""
    return [
        int(partner_id)
        for partner_id in partner_ids
        if uses[int(partner_id)] + int(partner_id == first_token_id) < max_building_block_uses
    ]


def _sample_random_balanced_rows(
    search_space: PreparedSearchSpace,
    *,
    count: int,
    seed: int | None,
    usage_penalty: float,
    max_building_block_uses: int,
    max_attempts: int,
    initial_building_block_ids: set[int] | None = None,
) -> tuple[list[dict[str, object]], dict[str, int | float]]:
    """Directly sample valid one-step routes with inverse-use BB weighting.

    The first building block is sampled from template-compatible first-reactant
    tokens using a probability inversely proportional to its accepted-use count.
    A template-compatible partner is then sampled with the same rule. This is a
    deliberately simple way to form a diverse warm-up pool without solving an
    explicit coverage problem. Duplicate products and failed template outcomes
    are rejected without changing any building-block use count.
    """
    if usage_penalty < 0.0:
        raise ValueError("usage_penalty must be nonnegative")
    if max_building_block_uses < 1:
        raise ValueError("max_building_block_uses must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    tokenizer = search_space.tokenizer.tokenizer
    token_name = {int(token_id): str(tokenizer.convert_ids_to_tokens(int(token_id))) for token_id in search_space.building_block_ids}
    templates = {template.identifier: template for template in search_space.templates}
    first_reactions = _first_reactions(search_space)
    first_token_ids = sorted(first_reactions)
    if initial_building_block_ids is not None:
        first_token_ids = [
            token_id for token_id in first_token_ids if token_id in initial_building_block_ids
        ]
    if not first_token_ids:
        if initial_building_block_ids is None:
            raise RuntimeError("No building block can serve as the first reactant in a prepared template")
        raise RuntimeError(
            "No constrained initial building block can serve as the first reactant in a prepared template"
        )

    rng = np.random.default_rng(seed)
    uses = {int(token_id): 0 for token_id in search_space.building_block_ids}
    seen_products: set[str] = set()
    rows: list[dict[str, object]] = []
    attempts = 0
    rejected_without_partner = 0
    rejected_without_product = 0
    rejected_duplicate_product = 0

    with rdBase.BlockLogs():
        while len(rows) < count and attempts < max_attempts:
            eligible_first = [
                token_id for token_id in first_token_ids if uses[token_id] < max_building_block_uses
            ]
            if not eligible_first:
                break
            attempts += 1
            first_token_id = _weighted_token_choice(
                eligible_first,
                uses,
                penalty=usage_penalty,
                rng=rng,
            )

            reaction_options: list[tuple[int, list[int]]] = []
            for reaction_id in first_reactions[first_token_id]:
                partners = _eligible_partners(
                    first_token_id=first_token_id,
                    partner_ids=search_space.compatibility[reaction_id][1],
                    uses=uses,
                    max_building_block_uses=max_building_block_uses,
                )
                if partners:
                    reaction_options.append((reaction_id, partners))
            if not reaction_options:
                rejected_without_partner += 1
                continue

            # Drawing a reaction in proportion to its eligible partner count is
            # equivalent to a uniform draw over feasible reaction-partner pairs
            # before the inverse-use partner correction is applied.
            reaction_weights = np.asarray([len(partners) for _, partners in reaction_options], dtype=float)
            reaction_id, partners = reaction_options[
                int(rng.choice(len(reaction_options), p=reaction_weights / reaction_weights.sum()))
            ]
            second_token_id = _weighted_token_choice(
                partners,
                uses,
                penalty=usage_penalty,
                rng=rng,
            )
            products = templates[reaction_id].run(
                search_space.token_to_mol[first_token_id],
                search_space.token_to_mol[second_token_id],
            )
            if not products:
                rejected_without_product += 1
                continue
            # A template application yields exactly one product; see
            # ``chemistry.reactions.ReactionTemplate.run``.
            product = products[0]
            product_smiles = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
            if product_smiles in seen_products:
                rejected_duplicate_product += 1
                continue

            first_prior_uses = uses[first_token_id]
            second_prior_uses = uses[second_token_id]
            uses[first_token_id] += 1
            uses[second_token_id] += 1
            seen_products.add(product_smiles)
            rows.append(
                {
                    "name": f"{token_name[first_token_id]} {token_name[second_token_id]}",
                    "token_ids": json.dumps([first_token_id, second_token_id]),
                    "reaction_ids": json.dumps([reaction_id]),
                    "product_smiles": product_smiles,
                    "entropy": float("nan"),
                    "sampling_mode": "random_balanced",
                    "first_building_block": token_name[first_token_id],
                    "second_building_block": token_name[second_token_id],
                    "first_building_block_prior_uses": first_prior_uses,
                    "second_building_block_prior_uses": second_prior_uses,
                }
            )
            if len(rows) % 500 == 0 or len(rows) == count:
                print(
                    f"Random-balanced sampler retained {len(rows):,}/{count:,} unique products after "
                    f"{attempts:,} proposals.",
                    flush=True,
                )

    values = list(uses.values())
    statistics: dict[str, int | float] = {
        "attempts": attempts,
        "rejected_without_partner": rejected_without_partner,
        "rejected_without_product": rejected_without_product,
        "rejected_duplicate_product": rejected_duplicate_product,
        "building_blocks_observed": sum(value > 0 for value in values),
        "building_blocks_never_observed": sum(value == 0 for value in values),
        "building_block_use_min": min(values),
        "building_block_use_mean": float(np.mean(values)),
        "building_block_use_max": max(values),
    }
    print(
        "Random-balanced sampler summary: "
        f"{statistics['building_blocks_observed']:,}/{len(values):,} building blocks observed; "
        f"use range {statistics['building_block_use_min']} to {statistics['building_block_use_max']}; "
        f"{rejected_without_product:,} failed template outcomes and "
        f"{rejected_duplicate_product:,} duplicate products rejected.",
        flush=True,
    )
    return rows, statistics


def run(args: argparse.Namespace) -> int:
    if args.num_trajectories < 1:
        raise ValueError("num-trajectories must be positive")
    if args.max_reactions < 1:
        raise ValueError("max-reactions must be positive")
    if args.usage_penalty < 0.0:
        raise ValueError("usage-penalty must be nonnegative")
    if args.max_building_block_uses < 1:
        raise ValueError("max-building-block-uses must be positive")
    if args.skip_scoring and args.oracle_config:
        print("--skip-scoring was set; ignoring --oracle-config.", flush=True)
    if not args.skip_scoring and not args.oracle_config:
        raise ValueError("--oracle-config is required unless --skip-scoring is used")
    if args.sampling_strategy == "random-balanced" and args.max_reactions != 1:
        raise ValueError("random-balanced sampling currently supports exactly one reaction step")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    search_space = load_prepared_search_space(args.search_space)
    initial_building_block_ids = None
    if args.initial_building_block_names_file:
        initial_building_block_ids = select_building_blocks_by_names(
            search_space,
            load_building_block_names(args.initial_building_block_names_file),
        )
    max_attempts = args.max_attempts or max(100, args.num_trajectories * 100)
    if max_attempts < 1:
        raise ValueError("max-attempts must be positive")

    if args.sampling_strategy == "random-balanced":
        rows, _ = _sample_random_balanced_rows(
            search_space,
            count=args.num_trajectories,
            seed=args.seed,
            usage_penalty=args.usage_penalty,
            max_building_block_uses=args.max_building_block_uses,
            max_attempts=max_attempts,
            initial_building_block_ids=initial_building_block_ids,
        )
    else:
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
        if args.device == "auto" and not torch.cuda.is_available():
            device = torch.device("cpu")
        model = build_policy_model(
            search_space,
            n_embd=args.n_embd,
            n_layer=args.n_layer,
            n_head=args.n_head,
            initial_building_block_ids=initial_building_block_ids,
        ).to(device).eval()
        with rdBase.BlockLogs():
            rows = _sample_unique_policy_rows(
                model,
                search_space,
                count=args.num_trajectories,
                max_reactions=args.max_reactions,
                temperature=args.temperature,
                top_p=args.top_p,
                epsilon=args.epsilon,
                device=device,
                max_attempts=max_attempts,
            )

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No valid molecular products were generated")
    if len(table) < args.num_trajectories:
        print(
            f"Prepared {len(table):,} unique trajectories, fewer than the requested "
            f"{args.num_trajectories:,} after {max_attempts:,} allowed proposals.",
            flush=True,
        )

    if not args.skip_scoring:
        oracle = build_oracle(_load_oracle_config(args.oracle_config))
        smiles = table["product_smiles"].tolist()
        table["score"] = require_aligned_scores(smiles, oracle.score_smiles(smiles))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    descriptor = "scored" if "score" in table.columns else "unscored"
    print(f"Wrote {len(table):,} unique {descriptor} warm-up trajectories to {output}")
    return 0
