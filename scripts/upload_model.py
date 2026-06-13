"""Lädt das beste Modell als W&B Artifact hoch.

Usage:
    python scripts/upload_model.py --model models/ppo_ent001_finetune/best.zip --name best-model-henrik
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", type=str, default="best-model-henrik")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not args.model.exists():
        print(f"Modell nicht gefunden: {args.model}")
        return

    import wandb
    run = wandb.init(project="qwop-rl-dhbw", entity="qwop-rl", job_type="model-upload")
    artifact = wandb.Artifact(args.name, type="model")
    artifact.add_file(str(args.model))
    run.log_artifact(artifact)
    run.finish()
    print(f"Hochgeladen: {args.name}")


if __name__ == "__main__":
    main()
