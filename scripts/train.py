"""Training entry point.

Loads a YAML config, starts a Stable-Baselines3 training run, and logs everything
to Weights & Biases. The actual environment binding (qwop-gym vs. own wrapper)
will be wired in once ADR-0001 has been accepted; until then the script imports
`make_env` from `qwop_rl.envs` lazily and fails with a clear message if it's
missing.

Usage:
    python scripts/train.py --config configs/ppo_default.yaml
    python scripts/train.py --config configs/ppo_default.yaml --run-name pl-baseline
    python scripts/train.py --config configs/ppo_default.yaml --tags experiment baseline

Environment variables (loaded from .env if present):
    WANDB_API_KEY        Required for online logging
    WANDB_PROJECT        Default: qwop-rl-dhbw
    WANDB_ENTITY         Optional team/user namespace
    WANDB_MODE           online | offline | disabled (default: online)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL agent on QWOP.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML training configuration.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override the W&B run name. Default: auto-generated from user/date.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="W&B tags to attach to this run.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging entirely (useful for quick local debugging).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def default_run_name() -> str:
    """Build a run name from $USER and today's date, e.g. 'pl-2026-06-04'."""
    user = os.environ.get("USER", "anon").lower().split(".")[0]
    initials = "".join(part[0] for part in user.split("-") if part) or "anon"
    return f"{initials}-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}"


def main() -> int:
    # Load .env if present — populates WANDB_API_KEY, WANDB_PROJECT, etc.
    load_dotenv()

    args = parse_args()
    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    run_name = args.run_name or default_run_name()

    # ── W&B init ──────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb and os.environ.get("WANDB_MODE") != "disabled"
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print(
                "[train] wandb not installed. Run `pip install wandb` or use --no-wandb.",
                file=sys.stderr,
            )
            return 1
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "qwop-rl-dhbw"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=run_name,
            tags=args.tags,
            config=config,
            sync_tensorboard=True,
            save_code=True,
        )
        print(f"[train] W&B run started: {run_name}")
    else:
        print("[train] W&B logging disabled.")

    # ── Env + Agent (placeholder until ADR-0001) ──────────────────────────
    try:
        from qwop_rl.envs import make_env  # noqa: F401  not yet implemented
    except ImportError:
        print(
            "[train] qwop_rl.envs.make_env is not implemented yet — "
            "waiting for ADR-0001 (QWOP binding decision).",
            file=sys.stderr,
        )
        if use_wandb:
            wandb.finish(exit_code=0, quiet=True)
        return 0

    # TODO: actual training loop, e.g.:
    #
    #     env = make_env(config["env"])
    #     model = PPO(config["ppo"]["policy"], env, **config["ppo"], tensorboard_log="logs/")
    #     model.learn(
    #         total_timesteps=config["training"]["total_timesteps"],
    #         callback=WandbCallback(model_save_path=f"models/{run_name}"),
    #     )
    #     if use_wandb:
    #         wandb.save(f"models/{run_name}/*.zip")

    print("[train] Training loop not yet implemented.")
    if use_wandb:
        wandb.finish(quiet=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
