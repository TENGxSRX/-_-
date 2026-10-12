"""Small prerecorded-video benchmark for the live execution presets.

This tool accepts a video file only.  It never opens a camera index and does
not train or change any model.  The default blank smoke clip is intentionally
tiny, so results describe pipeline overhead and scheduling rather than Acer
performance or production throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import psutil
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.inference.webcam_v1 import PERFORMANCE_PRESETS, should_process_frame  # noqa: E402


def _make_landmarker(pose_model: Path) -> PoseLandmarker:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def benchmark_preset(source: Path, pose_model: Path, preset_name: str, max_frames: int | None) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open prerecorded video: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    process_every = int(PERFORMANCE_PRESETS[preset_name]["process_every"])
    decoded = 0
    processed = 0
    detections = 0
    frame_index = 0
    started = time.perf_counter()
    process = psutil.Process(os.getpid())
    rss_start = process.memory_info().rss
    landmarker = _make_landmarker(pose_model)
    try:
        while max_frames is None or decoded < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if should_process_frame(frame_index, process_every, processed > 0):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int(round(frame_index * 1000.0 / source_fps)) if source_fps > 0 else frame_index
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                processed += 1
                detections += int(bool(result.pose_landmarks))
            frame_index += 1
    finally:
        landmarker.close()
        capture.release()
    elapsed = max(time.perf_counter() - started, 1e-9)
    rss_end = process.memory_info().rss
    return {
        "preset": preset_name,
        "process_every": process_every,
        "decoded_frames": decoded,
        "mediapipe_calls": processed,
        "pose_detections": detections,
        "source_fps": source_fps,
        "wall_seconds": elapsed,
        "decoded_wall_fps": decoded / elapsed,
        "mediapipe_calls_per_second": processed / elapsed,
        "calls_per_decoded_frame": processed / decoded if decoded else 0.0,
        "rss_start_mb": rss_start / (1024**2),
        "rss_end_mb": rss_end / (1024**2),
        "rss_delta_mb": (rss_end - rss_start) / (1024**2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark live presets on a prerecorded video file only.")
    parser.add_argument("--source", type=Path, default=ROOT / "outputs" / "videos" / "engine_blank_analyzed.mp4")
    parser.add_argument("--pose-model", type=Path, default=ROOT / "metadata" / "pose_landmarker_heavy.task")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "metrics" / "performance_presets_offline.json")
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    source = args.source.resolve()
    if source.stem.isdigit() or source.name.isdigit():
        parser.error("camera indexes are not accepted; provide a prerecorded video path")
    if not source.exists():
        parser.error(f"video path does not exist: {source}")
    pose_model = args.pose_model.resolve()
    if not pose_model.exists():
        parser.error(f"pose model does not exist: {pose_model}")
    rows = [benchmark_preset(source, pose_model, name, args.max_frames) for name in ("DESKTOP", "BALANCED", "LOW_SPEC")]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "source": str(source),
        "pose_model": str(pose_model),
        "benchmark_type": "small_prerecorded_video_smoke_only",
        "camera_opened": False,
        "model_training": False,
        "acer_specific_result": False,
        "interpretation": "Measures scheduling and MediaPipe call overhead on the supplied small video; it is not an Acer FPS claim and not a posture-accuracy result.",
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
