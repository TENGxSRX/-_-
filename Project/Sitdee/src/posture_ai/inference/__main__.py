"""Command-line entry point for offline image/video inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.posture_ai.inference.pipeline import infer_source
from src.posture_ai.inference.ergonomic_pipeline import run_posture_engine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run posture inference on an image, directory, video, or future webcam source 0")
    parser.add_argument("--source", required=True, help="Image path, image directory, video path, or explicit 0 for a future webcam run")
    parser.add_argument("--model", default="outputs/models/multiposture_v1")
    parser.add_argument("--posture-model", choices=("rgb_native_v4", "rgb_native_v5"), help="Named RGB-native backend; the artifact must already exist")
    parser.add_argument("--pose-model", default="metadata/pose_landmarker_heavy.task")
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--persistence-frames", type=int, default=3)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output")
    parser.add_argument("--engine-v1", action="store_true", help="Run the ergonomic posture engine with temporal risk logic and session telemetry")
    parser.add_argument("--config", default="configs/posture_engine.yaml", help="Posture-engine YAML project settings")
    parser.add_argument("--annotated-video", help="Optional output path for annotated prerecorded video")
    parser.add_argument("--event-log", help="Optional JSONL event-log output path")
    parser.add_argument("--session-report", help="Optional session-summary JSON output path")
    parser.add_argument("--performance-report", help="Optional offline performance JSON output path")
    parser.add_argument("--source-label", help="Verified external source label for annotation; defaults to UNLABELED")
    args = parser.parse_args()
    if args.posture_model:
        if args.model != "outputs/models/multiposture_v1":
            raise SystemExit("Use either --model or --posture-model, not both")
        args.model = f"outputs/models/{args.posture_model}"
    if args.engine_v1:
        if str(args.source) == "0":
            raise SystemExit("source 0 is intentionally disabled on this webcam-free machine")
        result = run_posture_engine(
            args.source,
            args.model,
            args.pose_model,
            config_path=args.config,
            max_frames=args.max_frames,
            annotated_video=args.annotated_video,
            event_log=args.event_log,
            session_report=args.session_report,
            performance_report=args.performance_report,
            source_label=args.source_label,
        )
        records = result["records"]
        if args.output:
            Path(args.output).write_text(
                "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
                encoding="utf-8",
            )
        else:
            print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False, indent=2))
        return
    records = infer_source(
        args.source,
        args.model,
        args.pose_model,
        min_visibility=args.min_visibility,
        min_confidence=args.min_confidence,
        window_size=args.window_size,
        persistence_frames=args.persistence_frames,
        max_frames=args.max_frames,
        backend=args.posture_model,
    )
    text = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else "")
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
