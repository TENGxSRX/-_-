"""Experimental offline V4 + geometry demo.

This is intentionally separate from the frozen V4 artifact.  It combines the
V4 direction prediction with non-clinical, camera-relative geometry indicators
for inspection on an image or prerecorded video.  It never opens webcam 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.posture_ai.ergonomics.geometry import calculate_geometry  # noqa: E402
from src.posture_ai.ergonomics.indicators import evaluate_indicators  # noqa: E402
from src.posture_ai.inference.pipeline import TemporalSmoother  # noqa: E402
from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner  # noqa: E402

POSE_MODEL = ROOT / "metadata" / "pose_landmarker_heavy.task"
MODEL_DIR = ROOT / "outputs" / "models" / "rgb_native_v4"


def _landmarker() -> PoseLandmarker:
    return PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def run(source: str | Path, output: str | Path, max_frames: int | None = None) -> dict:
    if str(source) == "0":
        raise ValueError("Hybrid offline demo refuses webcam source 0")
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open offline video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    runner = RGBNativeV4Runner(MODEL_DIR)
    smoother = TemporalSmoother(runner.labels, window_size=5, persistence_frames=3)
    landmarker = _landmarker()
    records = []
    index = 0
    try:
        while max_frames is None or index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(index * 1000.0 / fps))
            row = {"frame": index, "timestamp_seconds": index / fps, "status": "POSE_LOST", "direction": None, "geometry": None, "indicators": []}
            if result.pose_landmarks and result.pose_world_landmarks:
                image_points = result.pose_landmarks[0]
                world_points = result.pose_world_landmarks[0]
                try:
                    prediction = runner.predict_world_landmarks(world_points, image_visibility=image_points, visibility_threshold=0.5)
                    geometry = calculate_geometry(world_points)
                    temporal = smoother.update(np.asarray([prediction["probabilities"].get(label, 0.0) for label in runner.labels]))
                    row.update({"status": "OK", "direction": prediction["label"], "direction_confidence": prediction["confidence"], "temporal": temporal, "geometry": geometry, "indicators": evaluate_indicators(geometry, prediction["label"], confidence=prediction["confidence"])})
                except ValueError as exc:
                    row.update({"status": "LOW_VISIBILITY_OR_INVALID", "detail": str(exc)})
            records.append(row)
            index += 1
    finally:
        landmarker.close()
        capture.release()
    payload = {"source": str(path), "frames": len(records), "fps": fps, "status": "EXPERIMENTAL_OFFLINE_HYBRID_ONLY", "records": records, "limitations": ["V4 is a two-class DriveNet pilot, not a five-class replacement.", "Geometry indicators are camera-relative and not clinical measurements.", "No webcam source was opened."]}
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="offline image/video path; source 0 is forbidden")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "validation" / "overnight" / "hybrid_v4_geometry_demo.json"))
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    payload = run(args.source, args.output, args.max_frames)
    print(json.dumps({"output": args.output, "frames": payload["frames"], "status": payload["status"]}, indent=2))


if __name__ == "__main__":
    main()
