"""Run a bounded MediaPipe RGB preflight on imported MVSP media.

This is a sample gate only.  It does not train, label, or process the full
dataset.  It never opens webcam source 0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.posture_ai.features.rgb_native_v4 import V4_CORE_LANDMARKS  # noqa: E402
from src.posture_ai.pose.landmarks import LANDMARK_INDEX  # noqa: E402

RAW = ROOT / "datasets" / "raw" / "mvsp"
OUT = ROOT / "outputs" / "validation" / "mvsp"
REPORT = ROOT / "reports" / "mvsp_mediapipe_preflight.md"
POSE_MODEL = ROOT / "metadata" / "pose_landmarker_heavy.task"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _make_landmarker() -> PoseLandmarker:
    return PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))


def _check_result(result: Any) -> dict[str, Any]:
    row = {"pose_detected": bool(result.pose_landmarks), "world_available": bool(result.pose_world_landmarks), "core_accepted": False, "core_visibility_min": None, "reason": "no_pose"}
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return row
    points = result.pose_landmarks[0]
    visibility = np.asarray([float(getattr(point, "visibility", np.nan)) for point in points], dtype=float)
    indices = [LANDMARK_INDEX[name] for name in V4_CORE_LANDMARKS]
    row["core_visibility_min"] = float(np.nanmin(visibility[indices])) if np.isfinite(visibility[indices]).any() else None
    if not np.isfinite(visibility[indices]).all():
        row["reason"] = "missing_core_visibility"
    elif np.any(visibility[indices] < 0.5):
        row["reason"] = "core_visibility_below_0.5"
    else:
        row["core_accepted"] = True
        row["reason"] = "accepted"
    return row


def run(root: Path = RAW, max_images: int = 24, max_video_frames: int = 24) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts)
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    videos = [path for path in files if path.suffix.lower() in VIDEO_SUFFIXES]
    selected_images = images[:max_images]
    records = []
    landmarker = _make_landmarker()
    started = time.perf_counter()
    try:
        for index, path in enumerate(selected_images):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            row = {"path": str(path), "kind": "image", "readable": frame is not None, "channels": None if frame is None else int(frame.shape[2]) if frame.ndim == 3 else 1, "resolution": None if frame is None else [int(frame.shape[1]), int(frame.shape[0])]}
            if frame is None:
                row.update({"pose_detected": False, "world_available": False, "core_accepted": False, "reason": "corrupt_or_unreadable"})
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                row.update(_check_result(result))
            records.append(row)
        for path in videos:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                records.append({"path": str(path), "kind": "video", "readable": False, "pose_detected": False, "world_available": False, "core_accepted": False, "reason": "video_open_failed"})
                continue
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            count = 0
            try:
                while count < max_video_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(count * 1000.0 / fps))
                    item = {"path": str(path), "kind": "video", "frame": count, "readable": True, "resolution": [int(frame.shape[1]), int(frame.shape[0])], "channels": int(frame.shape[2]) if frame.ndim == 3 else 1}
                    item.update(_check_result(result))
                    records.append(item)
                    count += 1
            finally:
                capture.release()
    finally:
        landmarker.close()
    images_checked = [row for row in records if row["kind"] == "image"]
    video_checked = [row for row in records if row["kind"] == "video"]
    payload = {"status": "NO_MEDIA_TO_PREFLIGHT" if not images and not videos else "PREFLIGHT_COMPLETE", "raw_root": str(root), "total_images": len(images), "total_videos": len(videos), "images_checked": len(images_checked), "video_frames_checked": len(video_checked), "records": records, "summary": {"pose_detection_rate": float(np.mean([row["pose_detected"] for row in records])) if records else 0.0, "world_availability_rate": float(np.mean([row["world_available"] for row in records])) if records else 0.0, "core_acceptance_rate": float(np.mean([row["core_accepted"] for row in records])) if records else 0.0, "unreadable_media": int(sum(not row["readable"] for row in records)), "elapsed_seconds": time.perf_counter() - started}, "limitations": ["Bounded representative preflight only; this is not a full-dataset result.", "No labels are inferred from the preflight.", "No webcam device 0 was opened."]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mediapipe_preflight.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text("\n".join(["# MVSP MediaPipe RGB preflight", "", f"Status: **{payload['status']}**", "", f"Images in raw tree: `{len(images)}`; checked: `{len(images_checked)}`", f"Videos in raw tree: `{len(videos)}`; checked frames: `{len(video_checked)}`", f"Pose detection rate in checked sample: `{100*payload['summary']['pose_detection_rate']:.2f}%`", f"World landmark availability: `{100*payload['summary']['world_availability_rate']:.2f}%`", f"Five-core acceptance: `{100*payload['summary']['core_acceptance_rate']:.2f}%`", f"Unreadable checked media: `{payload['summary']['unreadable_media']}`", "", "This is a representative preflight. Full feature extraction is not authorized by this script and V5 is not trained automatically."]) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(RAW))
    parser.add_argument("--max-images", type=int, default=24)
    parser.add_argument("--max-video-frames", type=int, default=24)
    args = parser.parse_args()
    payload = run(Path(args.raw), args.max_images, args.max_video_frames)
    print(json.dumps({"status": payload["status"], "images": payload["total_images"], "videos": payload["total_videos"], "summary": payload["summary"], "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
