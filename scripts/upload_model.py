"""Upload a trained model to W&B as a versioned Artifact.

Usage:
    python scripts/upload_model.py --model models/ppo_default/BEST_36.7m_det50.zip --alias best

The other team members can then load it with:
    artifact = run.use_artifact("qwop-rl/qwop-rl-dhbw/qwop-ppo-model:best")
    path = artifact.download()
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a model to W&B as an artifact.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the model .zip to upload.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="qwop-ppo-model",
        help="Artifact name. Default: qwop-ppo-model.",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="best",
        help="Alias to tag this version with. Default: best.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional description of this model.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1

    import wandb

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "qwop-rl-dhbw"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        job_type="upload-model",
        name=f"upload-{args.model.stem}",
    )

    artifact = wandb.Artifact(
        name=args.name,
        type="model",
        description=args.notes or f"PPO model {args.model.name}",
    )
    artifact.add_file(str(args.model))

    run.log_artifact(artifact, aliases=[args.alias])
    print(f"[upload] Uploading {args.model.name} as '{args.name}:{args.alias}' ...")

    run.finish()
    print("[upload] Done. Available in W&B under Artifacts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
