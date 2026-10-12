"""Explicit launcher for Multi-Person Posture Monitoring Prototype V1.

This script never opens a camera unless the operator supplies --source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.multiperson.demo import run_multiperson_demo  # noqa: E402
from src.posture_ai.multiperson.runtime import PERFORMANCE_PRESETS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the explicit multi-person posture prototype over USB, file, or RTSP.")
    parser.add_argument("--source", required=True, help="Camera index, prerecorded video path, rtsp alias, RTSP URL, or Hikvision profile")
    parser.add_argument("--max-people", type=int, default=3, choices=range(1, 5), help="Maximum number of simultaneous anonymous subjects")
    parser.add_argument("--preset", choices=sorted(PERFORMANCE_PRESETS), default="RTSP_CCTV")
    parser.add_argument("--process-every", type=int, help="Override the preset frame interval")
    parser.add_argument("--model-dir", default=str(ROOT / "outputs" / "models" / "rgb_native_v4"))
    parser.add_argument("--pose-model", default=str(ROOT / "metadata" / "pose_landmarker_heavy.task"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "posture_engine.yaml"))
    parser.add_argument("--rtsp-config", default=str(ROOT / "configs" / "rtsp_profiles.yaml"))
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"))
    parser.add_argument("--camera-id", help="Non-secret camera identifier such as CAM01")
    parser.add_argument("--telemetry", help="Optional JSONL output path; capped at 2,000 events")
    parser.add_argument("--max-frames", type=int, help="Optional bounded run for a manual smoke test")
    parser.add_argument("--max-seconds", type=float, help="Optional bounded duration for a manual/RTSP smoke test")
    parser.add_argument("--max-reconnects", type=int, default=5)
    parser.add_argument("--tracking-mode", choices=("generic", "seat"), default="generic", help="Optional seat-aware assignment mode")
    parser.add_argument("--seat-layout", help="Explicit real seat-layout YAML required by --tracking-mode seat")
    parser.add_argument("--debug", action="store_true", help="Show geometry/tracking diagnostics on the overlay")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV preview window")
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--synchronous-source", action="store_true", help="Disable RTSP latest-frame reader for compatibility diagnostics")
    args = parser.parse_args()
    try:
        result = run_multiperson_demo(
            args.source,
            max_people=args.max_people,
            performance_preset=args.preset,
            process_every=args.process_every,
            model_dir=args.model_dir,
            pose_model=args.pose_model,
            config_path=args.config,
            rtsp_config_path=args.rtsp_config,
            profile=args.profile,
            camera_id=args.camera_id,
            telemetry_path=args.telemetry,
            max_frames=args.max_frames,
            max_seconds=args.max_seconds,
            max_reconnects=args.max_reconnects,
            tracking_mode=args.tracking_mode,
            seat_layout_path=args.seat_layout,
            display=not args.no_display,
            display_width=args.display_width,
            display_height=args.display_height,
            debug=args.debug,
            latest_frame_only=not args.synchronous_source,
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Could not start multi-person prototype: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
