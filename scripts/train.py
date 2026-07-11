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
import functools
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from dotenv import load_dotenv

if TYPE_CHECKING:
    from collections.abc import Callable

    import gymnasium as gym


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


def _make_qwop_env(env_config: dict[str, Any] | None) -> gym.Env:
    """Top-level env factory — spawn-sicher für SubprocVecEnv.

    Muss auf Modulebene liegen (nicht als Closure in main()), damit sie unter
    der macOS-``spawn``-Startmethode picklebar ist: SubprocVecEnv serialisiert
    jede ``env_fn`` und re-importiert das Modul im Worker-Prozess. Eine
    lokale Closure über ``config`` wäre dabei nicht zuverlässig picklebar.

    Ist ``QWOP_HEADLESS=1`` gesetzt, wird der Headless-Patch appliziert, BEVOR
    das Env gebaut wird. Die Env-Var wird bei spawn an jeden Worker vererbt, der
    Patch also pro Worker re-appliziert (siehe qwop_rl.envs._headless).
    """
    from qwop_rl.envs import make_env

    if os.environ.get("QWOP_HEADLESS") == "1":
        from qwop_rl.envs._headless import apply_headless_patch

        apply_headless_patch()

    return make_env(env_config)


# Whitelist der gepunkteten Keys, die ein W&B-Sweep injizieren darf. Bewusst
# fest verdrahtet, damit die vom Sweep zurückgespiegelte volle YAML-Config nicht
# versehentlich verschachtelte Strukturen überschreibt.
_SWEEP_OVERRIDE_KEYS = (
    "ppo.learning_rate",
    "ppo.ent_coef",
    "ppo.n_steps",
    "ppo.batch_size",
    "env.kwargs.failure_cost",
)


def apply_sweep_overrides(config: dict[str, Any], flat: dict[str, Any]) -> dict[str, Any]:
    """Merge flache W&B-Sweep-Parameter in die verschachtelte YAML-Config.

    Der Sweep-Agent injiziert die gesampelten Hyperparameter via ``wandb.config``
    als flaches Dict mit gepunkteten Keys (z.B. ``ppo.learning_rate``,
    ``env.kwargs.failure_cost``). Diese werden — nur für Keys aus
    :data:`_SWEEP_OVERRIDE_KEYS` — in die passende verschachtelte Stelle
    geschrieben; fehlende Zwischen-Dicts werden angelegt.

    Für normale (Nicht-Sweep-)Läufe ist das ein No-op, weil ``flat`` dann keine
    dieser gepunkteten Keys enthält. Mutiert und returned ``config``.
    """
    for dotted in _SWEEP_OVERRIDE_KEYS:
        if dotted not in flat:
            continue
        parts = dotted.split(".")
        node = config
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = flat[dotted]
    return config


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
        # Bei einem W&B-Sweep injiziert der Agent die gesampelten Hyperparameter
        # in wandb.config. Vor dem Bau von Env/Modell in die YAML-Config mergen.
        config = apply_sweep_overrides(config, dict(wandb.config))
    else:
        print("[train] W&B logging disabled.")

    # ── Env + Agent ───────────────────────────────────────────────────────
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from qwop_rl.envs._vecenv import NonDaemonicSubprocVecEnv

    training_cfg = config.get("training", {})
    n_envs = int(training_cfg.get("n_envs", 1))
    seed = training_cfg.get("seed")

    # Auto-Headless bei Parallelität: mehrere sichtbare Chrome-Fenster würden von
    # macOS gedrosselt (Stolperstein 12), sobald überdeckt. Ab n_envs>1 also
    # headless erzwingen. Die Env-Var wird an jeden gespawnten Worker vererbt und
    # dort in _make_qwop_env ausgewertet. Bei n_envs==1 bleibt der Browser
    # sichtbar (Debugging). Explizit vorab gesetztes QWOP_HEADLESS respektieren.
    if n_envs > 1 and "QWOP_HEADLESS" not in os.environ:
        os.environ["QWOP_HEADLESS"] = "1"
        print(f"[train] n_envs={n_envs} > 1 → Headless-Modus aktiviert.")

    # functools.partial über die top-level _make_qwop_env ist spawn-sicher
    # (picklebar), anders als eine lokale Closure. n_envs>1 → echte
    # Prozess-Parallelität via SubprocVecEnv; n_envs==1 → DummyVecEnv (einfacher
    # zu debuggen, kein IPC-Overhead, Einzelfenster-Fall).
    env_cfg = config.get("env")
    env_fns: list[Callable[[], gym.Env]] = [
        functools.partial(_make_qwop_env, env_cfg) for _ in range(n_envs)
    ]
    if n_envs > 1:
        # NonDaemonicSubprocVecEnv statt SB3s SubprocVecEnv: qwop-gym-Envs
        # spawnen selbst einen WSServer-Kindprozess, was daemonische Worker
        # nicht dürfen (siehe qwop_rl.envs._vecenv).
        vec_env: Any = NonDaemonicSubprocVecEnv(env_fns, start_method="spawn")
    else:
        vec_env = DummyVecEnv(env_fns)
    if seed is not None:
        vec_env.seed(int(seed))

    paths_cfg = config.get("paths", {})
    log_dir = Path(paths_cfg.get("log_dir", f"logs/{run_name}"))
    model_dir = Path(paths_cfg.get("model_dir", f"models/{run_name}"))
    # In einem W&B-Sweep teilen sich alle Trials dieselbe base-Config (und damit
    # denselben model_dir) — sonst würde jeder Trial das final.zip des vorherigen
    # überschreiben. Pro Trial einen eindeutigen Unterordner (run_name = Timestamp)
    # anhängen, damit alle Modelle erhalten bleiben und einzeln evaluierbar sind.
    if os.environ.get("WANDB_SWEEP_ID"):
        model_dir = model_dir / run_name
        log_dir = log_dir / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    ppo_cfg = dict(config.get("ppo", {}))
    policy = ppo_cfg.pop("policy", "MlpPolicy")
    model_load_file = training_cfg.get("model_load_file")
    reset_num_timesteps = True

    if model_load_file:
        print(f"[train] Loading model from {model_load_file}")
        model = PPO.load(model_load_file, env=vec_env, tensorboard_log=str(log_dir))
        reset_num_timesteps = False
    else:
        model = PPO(
            policy,
            vec_env,
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=int(seed) if seed is not None else None,
            **ppo_cfg,
        )

    checkpoint_freq = int(training_cfg.get("checkpoint_freq", 1_000_000))
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(model_dir),
        name_prefix="ckpt",
    )

    callbacks: list = [checkpoint_cb]
    if use_wandb and wandb_run is not None:
        from wandb.integration.sb3 import WandbCallback

        callbacks.append(
            WandbCallback(
                model_save_path=str(model_dir),
                verbose=2,
            )
        )

    total_timesteps = int(training_cfg.get("total_timesteps", 100_000))
    print(f"[train] Starting PPO for {total_timesteps:,} timesteps (n_envs={n_envs}).")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps,
    )

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
