"""Compare multiple trained QWOP models over several episodes each.

Runs each model headless-ish (browser must stay foreground!) for N episodes
and reports success_rate and average in-game time of successful runs.

Usage:
    python scripts/compare_models.py --episodes 20 ^
        models/ppo_default/best_peak_36.6m.zip ^
        models/ppo_default/best_peak_36.65m.zip ^
        models/ppo_default/best_peak_36.7m.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare QWOP models.")
    parser.add_argument(
        "models",
        nargs="+",
        type=Path,
        help="One or more paths to saved SB3 models (.zip).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Episodes to run per model. Default: 20.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions instead of sampling.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from stable_baselines3 import PPO

    from qwop_rl.envs import make_env

    # One env for all models — no browser window needed visibly, but keep
    # foreground per CLAUDE.md Stolperstein 12.
    env = make_env({"id": "QWOP-v1", "kwargs": {"game_in_browser": False}})

    results = []
    for model_path in args.models:
        if not model_path.exists():
            print(f"[compare] SKIP, not found: {model_path}", file=sys.stderr)
            continue

        model = PPO.load(model_path, env=env)
        successes = 0
        success_times = []

        for _ep in range(args.episodes):
            obs, _info = env.reset()
            terminated = truncated = False
            last_info: dict = {}
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, _reward, terminated, truncated, last_info = env.step(action)
            if last_info.get("is_success"):
                successes += 1
                success_times.append(float(last_info.get("time", 0.0)))

        rate = successes / args.episodes
        avg_time = sum(success_times) / len(success_times) if success_times else None
        results.append((model_path.name, rate, avg_time, successes))
        avg_str = f"{avg_time:.3f}s" if avg_time is not None else "—"
        print(
            f"[compare] {model_path.name:<32} "
            f"success_rate={rate:.2f} ({successes}/{args.episodes})  "
            f"avg_time={avg_str}"
        )

    env.close()

    # Summary table
    print("\n=== Zusammenfassung (sortiert nach success_rate) ===")
    print(f"{'Modell':<34}{'success_rate':<14}{'avg_time':<10}")
    for name, rate, avg_time, succ in sorted(results, key=lambda r: -r[1]):
        avg_str = f"{avg_time:.3f}s" if avg_time is not None else "—"
        print(f"{name:<34}{rate:<14.2f}{avg_str:<10}")

    return 0


if __name__ == "__main__":
    sys.exit(main())