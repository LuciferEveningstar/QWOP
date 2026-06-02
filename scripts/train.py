"""Training entry point — placeholder until the QWOP environment is implemented.

Usage:
    python scripts/train.py --config configs/ppo_default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL agent on QWOP.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML training configuration.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1

    print(f"[train] Would start training with config: {args.config}")
    print("[train] Not yet implemented — waiting for ADR-0001 (QWOP binding).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
