"""Interactive single-person RTSP AI Lab for the frozen V4 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.inference.rtsp_ai_lab import run_rtsp_ai_lab_session  # noqa: E402
from src.posture_ai.inference.video_source import SourceConfigurationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen single-person V4 AI Lab on USB, file, or Hikvision RTSP")
    parser.add_argument("--source", default="rtsp", help="rtsp, HIKVISION_MAIN/SUB, rtsp:// URL, video file, or explicit USB index")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default="HIKVISION_MAIN")
    parser.add_argument("--camera-id", default=None, help="Non-secret camera identifier such as CAM01")
    parser.add_argument("--model", default="outputs/models/rgb_native_v4")
    parser.add_argument("--pose-model", default="metadata/pose_landmarker_heavy.task")
    parser.add_argument("--config", default="configs/posture_engine.yaml")
    parser.add_argument("--rtsp-config", default="configs/rtsp_profiles.yaml")
    parser.add_argument("--preset", choices=("RTSP_DESKTOP", "RTSP_BALANCED", "RTSP_LOW_LATENCY"), default="RTSP_LOW_LATENCY")
    parser.add_argument("--max-people", type=int, choices=(1,), default=1, help="Lab is intentionally single-person")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-reconnects", type=int, default=5)
    parser.add_argument("--output-root", default="outputs/rtsp_ai_lab")
    parser.add_argument("--no-display", action="store_true", help="Run without the preview window")
    parser.add_argument("--mirror", action="store_true", help="Mirror the preview; text remains readable")
    parser.add_argument("--debug", action="store_true", help="Start with detailed timing/probability diagnostics")
    parser.add_argument("--detailed-landmarks", action="store_true", help="Start with all MediaPipe landmarks")
    parser.add_argument("--inference-width", type=int, help="Maximum width for the MediaPipe inference copy")
    parser.add_argument("--inference-height", type=int, help="Maximum height for the MediaPipe inference copy")
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    args = parser.parse_args()
    if args.max_reconnects < 0:
        parser.error("--max-reconnects must not be negative")
    try:
        result = run_rtsp_ai_lab_session(
            args.source,
            model_dir=args.model,
            pose_model=args.pose_model,
            config_path=args.config,
            rtsp_config_path=args.rtsp_config,
            profile=args.profile,
            camera_id=args.camera_id,
            preset=args.preset,
            max_people=args.max_people,
            max_frames=args.max_frames,
            max_seconds=args.max_seconds,
            max_reconnects=args.max_reconnects,
            output_root=args.output_root,
            display=not args.no_display,
            mirror=args.mirror,
            debug=args.debug,
            detailed_landmarks=args.detailed_landmarks,
            inference_width=args.inference_width,
            inference_height=args.inference_height,
            display_width=args.display_width,
            display_height=args.display_height,
        )
    except (SourceConfigurationError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"RTSP AI Lab could not start: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"summary": result["summary"], "paths": result["paths"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
