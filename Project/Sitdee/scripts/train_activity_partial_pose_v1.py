"""Explicitly gated trainer for human-labelled partial-pose Activity AI V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.data.activity_partial_training import (  # noqa: E402
    activity_training_readiness,
    evaluate_grouped_activity_models,
    export_activity_candidate,
    load_behavior_rows,
    v4_cctv_baseline_readiness,
)
from src.posture_ai.features.activity_partial_pose_v1 import FEATURE_VARIANTS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Human-label-gated partial-pose Activity AI V1 trainer")
    parser.add_argument("--data-root", default="data/behavior_partial_pose_v1")
    parser.add_argument("--train", action="store_true", help="Required before fitting any candidate model")
    parser.add_argument("--output-dir", default="outputs/models/activity_partial_pose_v1_candidate")
    args = parser.parse_args()
    try:
        rows = load_behavior_rows(args.data_root)
        readiness = activity_training_readiness(rows)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(json.dumps({"status": "ACTIVITY_TRAINING_BLOCKED", "detail": str(exc)}, ensure_ascii=False))
        return 2
    output = {"readiness": readiness, "frozen_v4_cctv_baseline": v4_cctv_baseline_readiness(rows), "training_performed": False}
    if not args.train:
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    if readiness["status"] != "READY_FOR_GROUPED_ACTIVITY_TRAINING":
        print(json.dumps({**output, "status": "ACTIVITY_TRAINING_BLOCKED_NO_SUFFICIENT_VERIFIED_DATA"}, ensure_ascii=False, indent=2))
        return 2
    evaluation = evaluate_grouped_activity_models(rows, variants=FEATURE_VARIANTS)
    destination = export_activity_candidate(evaluation, rows, args.output_dir)
    print(json.dumps({"status": "ACTIVITY_TRAINING_COMPLETED", "training_performed": True, "evaluation": evaluation, "output_dir": str(destination)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
