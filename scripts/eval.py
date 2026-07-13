"""Evaluate a trained QWOP agent — load a saved model, play visibly in the browser.

Usage:
    python scripts/eval.py --model models/pl-2026-06-11-test500k/final.zip
    python scripts/eval.py --model models/pl-2026-06-11-test500k/final.zip --episodes 5

The browser window must stay in the foreground while playing — same constraint
as training (see CLAUDE.md, Stolperstein 12).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a saved QWOP model.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the saved SB3 model (final.zip).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play. Default: 5.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Pick the argmax action instead of sampling. Cleaner playback.",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=4,
        help=(
            "Render every Nth step (default: 4 → ~15 Hz). Lower = smoother but"
            " slower; higher = faster but choppier. Set to 0 to disable rendering."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1

    from stable_baselines3 import PPO

    from qwop_rl.envs import make_env

    # Show the game in the browser. auto_draw=False (default) means we have to
    # call env.render() ourselves — doing it every Nth step lets us trade
    # smoothness against speed.
    env = make_env(
    {
        "id": "QWOP-v1",
        "kwargs": {
            "browser": "C:/Program Files/Google/Chrome/Application/chrome.exe",
            "driver": "C:/tools/chromedriver/chromedriver-win64/chromedriver.exe",
        },
    }
)
    model = PPO.load(args.model, env=env)

    for ep in range(1, args.episodes + 1):
        obs, _info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            steps += 1
            if args.render_every and steps % args.render_every == 0:
                env.render()
        print(f"[eval] Episode {ep}: reward={total_reward:.2f}, steps={steps}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
