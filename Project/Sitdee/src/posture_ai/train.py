"""CLI entry point for the leakage-safe MultiPosture track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.posture_ai.models.training import train_multiposture


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate MultiPosture without participant leakage")
    parser.add_argument("--dataset", default="multiposture", choices=["multiposture"])
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=5)
    args = parser.parse_args()
    root = Path(args.project_root)
    csv_path = root / "datasets" / "raw" / "multiposture" / "data.csv"
    result = train_multiposture(csv_path, root, seed=args.seed, cv_splits=args.cv_splits)
    print(json.dumps({k: v for k, v in result.items() if k != "ranking"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

