"""Configuration-driven task-specific warm-up and PPO optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from syngenmol.chemistry.constraints import (
    load_building_block_names,
    select_building_blocks_by_names,
    select_building_blocks_by_smarts,
)
from syngenmol.config import load_yaml_mapping
from syngenmol.chemistry.search_space import load_prepared_search_space
from syngenmol.models import build_policy_model
from syngenmol.oracles import build_oracle
from syngenmol.rl import PPOSettings, SynGenMolPPOTrainer, warmup_from_csv


def _load_config(path: str | Path) -> dict[str, Any]:
    return load_yaml_mapping(path, description="Training configuration")


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("train", help="Run task-specific warm-up and PPO optimization.")
    parser.add_argument("--config", required=True, help="YAML training configuration.")
    parser.add_argument("--output-dir", default=None, help="Override the output directory in the configuration.")
    parser.set_defaults(handler=run)
    return parser


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


def run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    seed = config.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
    search_space = load_prepared_search_space(config["search_space"])
    model_config = dict(config.get("model", {}))
    initial_ids = _resolve_initial_constraint(search_space, config.get("initial_building_block_constraint"))
    policy = build_policy_model(search_space, initial_building_block_ids=initial_ids, **model_config)

    try:
        from trl import AutoModelForCausalLMWithValueHead
    except ImportError as exc:
        raise ImportError("SynGenMol training requires TRL. Install the release training dependencies.") from exc
    model = AutoModelForCausalLMWithValueHead.from_pretrained(policy)
    device_name = str(config.get("device", "auto"))
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)
    if device_name == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model.to(device)

    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/syngenmol"))
    output_dir.mkdir(parents=True, exist_ok=True)

    warmup_config = config.get("warmup")
    warmup_result = None
    if warmup_config and warmup_config.get("trajectories_csv"):
        settings = {key: value for key, value in warmup_config.items() if key != "trajectories_csv"}
        warmup_result = warmup_from_csv(
            model,
            search_space.tokenizer.tokenizer,
            warmup_config["trajectories_csv"],
            **settings,
        )
        (output_dir / "warmup_metrics.json").write_text(
            json.dumps(
                {
                    "reward_mean": warmup_result.reward_mean,
                    "reward_std": warmup_result.reward_std,
                    "train_losses": warmup_result.train_losses,
                    "validation_losses": warmup_result.validation_losses,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    oracle = build_oracle(config["oracle"])
    ppo_config = dict(config.get("ppo", {}))
    steps = int(ppo_config.pop("steps", 1_000))
    max_reactions = int(ppo_config.pop("max_reactions", 1))
    temperature = float(ppo_config.pop("temperature", 1.0))
    top_p = float(ppo_config.pop("top_p", 1.0))
    epsilon = float(ppo_config.pop("epsilon", 0.0))
    trainer = SynGenMolPPOTrainer(
        model=model,
        tokenizer=search_space.tokenizer.tokenizer,
        oracle=oracle,
        settings=PPOSettings(**ppo_config),
        output_dir=output_dir,
    )
    history: list[dict[str, float]] = []
    for step in range(steps):
        _, metrics = trainer.step(
            max_reactions=max_reactions,
            temperature=temperature,
            top_p=top_p,
            epsilon=epsilon,
        )
        metrics["step"] = step
        history.append(metrics)
        print(f"PPO step {step + 1}/{steps}: mean score={metrics['score_mean']:.4f}, max score={metrics['score_max']:.4f}")
    (output_dir / "training_metrics.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_state_dict": model.state_dict(), "config": config}, output_dir / "model_final.pt")
    print(f"Saved model and metrics to {output_dir}")
    return 0
