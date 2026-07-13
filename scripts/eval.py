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
import time
from pathlib import Path

import numpy as np


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
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help=(
            "Künstliche Verlangsamung: max. Frames pro Sekunde (echter time.sleep"
            " pro Step). Default 0 = kein Limit (so schnell wie möglich). z.B. 30"
            " für flüssiges, echtzeitnahes Abspielen, 10 für Zeitlupe."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Fester Seed für Episode 1 (jede weitere Episode: seed+1, ...). Macht"
            " Läufe reproduzierbar. Ohne --seed wird ein Zufalls-Basisseed genutzt"
            " und am Ende ausgegeben. Zum EXAKTEN Nachspielen des schnellsten Laufs:"
            " --seed <ausgegebener Seed> --episodes 1 --deterministic."
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
                "game_in_browser": True,
                "stat_in_browser": True,
            },
        }
    )
    model = PPO.load(args.model, env=env)

    # Basis-Seed: fest (--seed) oder zufällig. Jede Episode nutzt base_seed+ep,
    # sodass jeder Lauf über seinen Seed exakt reproduzierbar ist. numpy statt
    # random.random(), damit auch ohne --seed ein konkreter, ausgebbarer Wert
    # entsteht.
    base_seed = args.seed if args.seed is not None else int(np.random.randint(0, 2**31 - 1))
    print(f"[eval] Basis-Seed: {base_seed}")

    # Schnellsten erfolgreichen Lauf (höchste avgspeed bei is_success) mitschreiben.
    best_speed = -1.0
    best_seed: int | None = None
    best_stats = ""

    for ep in range(1, args.episodes + 1):
        ep_seed = base_seed + ep
        obs, _info = env.reset(seed=ep_seed)
        # Auch den Policy-RNG seeden, sonst ist Sampling nicht reproduzierbar.
        model.set_random_seed(ep_seed)
        total_reward = 0.0
        steps = 0
        info: dict = {}
        # Torso-Höhe pro Step sammeln (obs[1] = normalisierte Rumpfhöhe, siehe
        # UprightRewardWrapper). Direkt aus der Obs gelesen, damit die Analyse
        # auch OHNE aktives reward_shaping funktioniert — z.B. um einen Robb-
        # Checkpoint zu vermessen und den height_threshold zu kalibrieren.
        torso_heights: list[float] = []
        # Ziel-Dauer pro Step aus --fps (0 = kein Limit). time.sleep bremst das
        # sonst so-schnell-wie-möglich laufende Abspielen auf echtzeitnahes Tempo.
        frame_budget = 1.0 / args.fps if args.fps > 0 else 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            step_start = time.perf_counter()
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            obs_arr = np.asarray(obs, dtype=np.float64)
            if obs_arr.ndim == 1 and obs_arr.shape[0] > 1:
                torso_heights.append(float(obs_arr[1]))
            if args.render_every and steps % args.render_every == 0:
                env.render()
            # Auf das Frame-Budget warten (nur wenn --fps gesetzt).
            if frame_budget:
                elapsed = time.perf_counter() - step_start
                if elapsed < frame_budget:
                    time.sleep(frame_budget - elapsed)

        # info aus dem letzten Step enthält die fairen, frames_per_step-
        # unabhängigen Kennzahlen (qwop_env._build_info): time = In-Game-
        # Simulationszeit bis Episodenende, distance = zurückgelegte Strecke,
        # avgspeed = distance/time. "time to finish" ist bei Erfolg genau die
        # Zeit bis 100m — im Gegensatz zu steps/reward über Modelle mit
        # unterschiedlichem frames_per_step vergleichbar.
        finish_time = info.get("time")
        distance = info.get("distance")
        avgspeed = info.get("avgspeed")
        success = bool(info.get("is_success"))
        outcome = "ZIEL" if success else "Sturz/Timeout"
        time_str = f"{finish_time:.2f}s" if finish_time is not None else "n/a"
        dist_str = f"{distance:.1f}" if distance is not None else "n/a"
        speed_str = f"{avgspeed:.2f}" if avgspeed is not None else "n/a"
        # torso_height: mean + max über die Episode (normalisiert, [-1,1]).
        # mean = wie aufrecht im Schnitt (Robben → niedrig, Laufen → höher).
        if torso_heights:
            h_mean = sum(torso_heights) / len(torso_heights)
            h_max = max(torso_heights)
            height_str = f"mean={h_mean:.3f}/max={h_max:.3f}"
        else:
            height_str = "n/a"
        print(
            f"[eval] Episode {ep} (seed={ep_seed}): {outcome} | time={time_str} "
            f"| distance={dist_str} | avgspeed={speed_str} | torso_height={height_str} "
            f"| reward={total_reward:.2f} | steps={steps}"
        )

        # Schnellsten ERFOLGREICHEN Lauf mitschreiben (höchste avgspeed bei Finish).
        if success and avgspeed is not None and float(avgspeed) > best_speed:
            best_speed = float(avgspeed)
            best_seed = ep_seed
            best_stats = f"avgspeed={speed_str}, time={time_str}, distance={dist_str}"

    env.close()

    # Zusammenfassung: der schnellste erfolgreiche Lauf + exakter Reproduktions-Befehl.
    print()
    if best_seed is not None:
        print(f"[eval] Schnellster ZIEL-Lauf: seed={best_seed} ({best_stats})")
        # Repro-Befehl MUSS dieselbe deterministic-Einstellung spiegeln wie dieser
        # Lauf — sonst weicht die Trajektorie ab. Bei Sampling (Default) macht der
        # Policy-RNG (set_random_seed) den Lauf reproduzierbar.
        det_flag = " --deterministic" if args.deterministic else ""
        print(
            f"[eval] Exakt nachspielen (langsam): python scripts/eval.py "
            f"--model {args.model} --episodes 1 --seed {best_seed - 1}"
            f"{det_flag} --fps 30 --render-every 1"
        )
    else:
        print("[eval] Kein erfolgreicher (ZIEL-)Lauf in dieser Session.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
