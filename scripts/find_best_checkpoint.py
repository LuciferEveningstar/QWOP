"""Evaluiert alle Checkpoints und findet das beste Modell.

Usage:
    python scripts/find_best_checkpoint.py --checkpoints models/ppo_ent001/checkpoints --episodes 10
    python scripts/find_best_checkpoint.py --checkpoints models/ppo_ent001/checkpoints --episodes 10 --step-filter 1000000
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from qwop_rl.envs import make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--step-filter", type=int, default=None,
                        help="Nur Checkpoints ab dieser Step-Zahl evaluieren, z.B. 1000000")
    return parser.parse_args()


def kill_chrome() -> None:
    subprocess.call(
        ["taskkill", "/F", "/IM", "chromedriver.exe", "/T"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    subprocess.call(
        ["taskkill", "/F", "/IM", "chrome.exe", "/T"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(3)


def evaluate(model_path: Path, n_episodes: int) -> tuple[float, float]:
    env = DummyVecEnv([lambda: make_env()])
    model = PPO.load(model_path, env=env)
    rewards = []
    obs = env.reset()
    ep_reward = 0.0
    ep_count = 0
    while ep_count < n_episodes:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, _ = env.step(action)
        ep_reward += reward[0]
        if done[0]:
            rewards.append(ep_reward)
            ep_reward = 0.0
            ep_count += 1
            obs = env.reset()
    env.close()
    kill_chrome()
    return float(np.mean(rewards)), float(np.std(rewards))


def main() -> None:
    args = parse_args()
    checkpoints = sorted(args.checkpoints.glob("model_*_steps.zip"),
                         key=lambda p: int(p.stem.split("_")[1]))

    if args.step_filter:
        checkpoints = [c for c in checkpoints
                       if int(c.stem.split("_")[1]) >= args.step_filter]

    print(f"Evaluiere {len(checkpoints)} Checkpoints mit je {args.episodes} Episoden...\n")
    print(f"{'Steps':>12}  {'Mean Reward':>12}  {'Std':>8}  Datei")
    print("-" * 65)

    results = []
    for ckpt in checkpoints:
        steps = int(ckpt.stem.split("_")[1])
        mean, std = evaluate(ckpt, args.episodes)
        results.append((steps, mean, std, ckpt))
        marker = " ◄ bisher bestes" if mean == max(r[1] for r in results) else ""
        print(f"{steps:>12,}  {mean:>12.2f}  {std:>8.2f}  {ckpt.name}{marker}")

    best = max(results, key=lambda r: r[1])
    print(f"\nBestes Modell: {best[3].name}")
    print(f"  Steps:       {best[0]:,}")
    print(f"  Mean Reward: {best[1]:.2f} ± {best[2]:.2f}")
    print(f"\nKopiere es mit:")
    print(f"  copy {best[3]} models\\ppo_ent001\\best.zip")


if __name__ == "__main__":
    main()
