"""Beginner-friendly launcher for Live Webcam Prototype V1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.inference.webcam_v1 import PERFORMANCE_PRESETS, run_webcam_session  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen two-class RGB-native V4 live webcam prototype.")
    parser.add_argument("--source", default="0", help="Camera index: 0, 1, or 2")
    parser.add_argument("--model", default="outputs/models/rgb_native_v4")
    parser.add_argument("--pose-model", default="metadata/pose_landmarker_heavy.task")
    parser.add_argument("--config", default="configs/posture_engine.yaml")
    parser.add_argument("--validation", action="store_true", help="Run the timed beginner validation sequence")
    parser.add_argument("--calibration", action="store_true", help="Show camera-relative geometry diagnostics")
    parser.add_argument("--record", action="store_true", help="Save an annotated MP4; recording is off by default")
    parser.add_argument("--sound", action="store_true", help="Enable optional Windows warning sound")
    parser.add_argument("--no-mirror", action="store_true", help="Show the processing orientation instead of a user-facing mirror")
    parser.add_argument("--validation-step-seconds", type=float, default=5.0)
    parser.add_argument("--validation-poor-hold-seconds", type=float, default=20.0)
    parser.add_argument("--neutral-calibration-seconds", type=float, default=4.0, help="Stable neutral-pose baseline duration")
    parser.add_argument("--stable-core-seconds", type=float, default=1.5, help="Continuous core-visibility time required before start")
    parser.add_argument("--max-frames", type=int, help="Optional bounded smoke-test frame count")
    parser.add_argument("--preset", choices=sorted(PERFORMANCE_PRESETS), default="DESKTOP", help="Execution preset; LOW_SPEC processes every third frame")
    parser.add_argument("--process-every", type=int, help="Override preset frame interval; first frame is always processed")
    args = parser.parse_args()
    try:
        result = run_webcam_session(
            args.source,
            model_dir=args.model,
            pose_model=args.pose_model,
            config_path=args.config,
            validation=args.validation,
            calibration=args.calibration,
            sound=args.sound,
            record=args.record,
            mirror=not args.no_mirror,
            validation_step_seconds=args.validation_step_seconds,
            validation_poor_hold_seconds=args.validation_poor_hold_seconds,
            neutral_calibration_seconds=args.neutral_calibration_seconds,
            stable_core_seconds=args.stable_core_seconds,
            max_frames=args.max_frames,
            performance_preset=args.preset,
            process_every=args.process_every,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Could not start the webcam prototype: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"summary": result["summary"], "paths": result["paths"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
