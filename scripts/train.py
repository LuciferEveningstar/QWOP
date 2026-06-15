"""Training entry point.

Loads a YAML config, builds a vectorised QWOP environment, runs Stable-Baselines3
PPO, and logs everything to Weights & Biases (unless disabled via --no-wandb or
WANDB_MODE=disabled).

Usage:
    python scripts/train.py --config configs/ppo_default.yaml
    python scripts/train.py --config configs/ppo_default.yaml --run-name pl-baseline
    python scripts/train.py --config configs/ppo_default.yaml --tags experiment baseline
    python scripts/train.py --config configs/ppo_default.yaml --no-wandb

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
        "--model",
        type=Path,
        default=None,
        help="Checkpoint zum Weitermachen (optional). Z.B. models/ppo_ent001/best.zip",
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
    wandb_run = None
    if use_wandb:
        try:
            import wandb
        except ImportError:
            print(
                "[train] wandb not installed. Run `pip install wandb` or use --no-wandb.",
                file=sys.stderr,
            )
            return 1
        wandb_run = wandb.init(
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

    # ── Env + Agent ───────────────────────────────────────────────────────
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from qwop_rl.envs import make_env

    training_cfg = config.get("training", {})
    n_envs = int(training_cfg.get("n_envs", 1))
    seed = training_cfg.get("seed")

    def _env_factory() -> Any:
        return make_env(config.get("env"))

    vec_env = DummyVecEnv([_env_factory for _ in range(n_envs)])
    if seed is not None:
        vec_env.seed(int(seed))

    paths_cfg = config.get("paths", {})
    log_dir = Path(paths_cfg.get("log_dir", f"logs/{run_name}"))
    model_dir = Path(paths_cfg.get("model_dir", f"models/{run_name}"))
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    ppo_cfg = dict(config.get("ppo", {}))
    policy = ppo_cfg.pop("policy", "MlpPolicy")
    if args.model is not None:
        print(f"[train] Lade Checkpoint: {args.model}")
        model = PPO.load(
            args.model,
            env=vec_env,
            tensorboard_log=str(log_dir),
            **ppo_cfg,
        )
    else:
        model = PPO(
            policy,
            vec_env,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=int(seed) if seed is not None else None,
            **ppo_cfg,
        )

    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    checkpoint_freq = int(training_cfg.get("checkpoint_freq", 50_000))
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="model",
        verbose=1,
    )
    callbacks = [checkpoint_callback]

    if use_wandb and wandb_run is not None:
        from wandb.integration.sb3 import WandbCallback

        callbacks.append(WandbCallback(
            model_save_path=str(model_dir),
            verbose=2,
        ))

    callback = CallbackList(callbacks)

    total_timesteps = int(training_cfg.get("total_timesteps", 100_000))
    print(f"[train] Starting PPO for {total_timesteps:,} timesteps (n_envs={n_envs}).")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    final_path = model_dir / "final.zip"
    model.save(final_path)
    print(f"[train] Model saved to {final_path}")

    if use_wandb and wandb_run is not None:
        wandb_run.save(str(final_path))
        wandb_run.finish(quiet=True)

    vec_env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
