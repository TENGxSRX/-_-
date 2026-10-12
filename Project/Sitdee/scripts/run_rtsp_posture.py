"""Explicit source-runtime entry point for future RTSP posture sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.inference.stream_runtime import run_source_posture_session  # noqa: E402
from src.posture_ai.inference.video_source import SourceConfigurationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen V4 on an explicit USB/file/RTSP source")
    parser.add_argument("--source", default="rtsp", help="0/1/2, video file, rtsp:// URL, rtsp alias, or Hikvision profile")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"))
    parser.add_argument("--camera-id", default=None, help="Non-secret camera identifier such as CAM01")
    parser.add_argument("--model", default="outputs/models/rgb_native_v4")
    parser.add_argument("--pose-model", default="metadata/pose_landmarker_heavy.task")
    parser.add_argument("--config", default="configs/posture_engine.yaml")
    parser.add_argument("--rtsp-config", default="configs/rtsp_profiles.yaml")
    parser.add_argument("--preset", choices=("RTSP_DESKTOP", "RTSP_BALANCED"), default="RTSP_DESKTOP")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-reconnects", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/live")
    parser.add_argument("--display", action="store_true", help="Show a local preview; off by default")
    args = parser.parse_args()
    if args.max_reconnects < 0:
        parser.error("--max-reconnects must not be negative")
    try:
        result = run_source_posture_session(
            args.source,
            model_dir=args.model,
            pose_model=args.pose_model,
            config_path=args.config,
            rtsp_config_path=args.rtsp_config,
            profile=args.profile,
            camera_id=args.camera_id,
            preset=args.preset,
            max_frames=args.max_frames,
            max_reconnects=args.max_reconnects,
            output_root=args.output_root,
            display=args.display,
        )
    except (SourceConfigurationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Source runtime could not start: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"summary": result["summary"], "paths": result["paths"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
