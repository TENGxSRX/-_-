"""Offline image/video inference for the RGB-native V4 pilot.

The CLI intentionally refuses source ``0`` on this workstation.  It accepts
an image, an image directory, or a prerecorded video and uses World Landmarks
plus the exact V4 feature builder used during training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _landmarker(asset: Path, mode: RunningMode) -> PoseLandmarker:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(asset)),
        running_mode=mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def _predict(result, runner: RGBNativeV4Runner, frame_index: int) -> dict:
    record = {"frame": frame_index}
    if not result.pose_landmarks or not result.pose_world_landmarks:
        record["status"] = "no_pose_or_world_landmarks"
        return record
    try:
        prediction = runner.predict_world_landmarks(result.pose_world_landmarks[0], image_visibility=result.pose_landmarks[0])
    except ValueError as exc:
        record.update({"status": "core_visibility_or_schema_error", "detail": str(exc)})
        return record
    record.update({"status": "ok", **prediction})
    return record


def infer(source: str | Path, model_dir: str | Path, pose_model: str | Path, max_frames: int | None = None) -> list[dict]:
    if str(source) == "0":
        raise ValueError("Webcam source 0 is intentionally disabled on this machine")
    source_path = Path(source)
    runner = RGBNativeV4Runner(model_dir)
    records: list[dict] = []
    if source_path.is_file() and source_path.suffix.lower() in IMAGE_SUFFIXES:
        detector = _landmarker(Path(pose_model), RunningMode.IMAGE)
        try:
            frame = cv2.imread(str(source_path))
            if frame is None:
                raise ValueError(f"Could not read image: {source_path}")
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            records.append({"source": str(source_path), **_predict(detector.detect(image), runner, 0)})
        finally:
            detector.close()
        return records
    if source_path.is_dir():
        image_paths = sorted(p for p in source_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        detector = _landmarker(Path(pose_model), RunningMode.IMAGE)
        try:
            for index, path in enumerate(image_paths):
                if max_frames is not None and index >= max_frames:
                    break
                frame = cv2.imread(str(path))
                if frame is None:
                    records.append({"source": str(path), "frame": index, "status": "invalid_image"})
                    continue
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                records.append({"source": str(path), **_predict(detector.detect(image), runner, index)})
        finally:
            detector.close()
        return records
    if source_path.is_file() and source_path.suffix.lower() in VIDEO_SUFFIXES:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {source_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        detector = _landmarker(Path(pose_model), RunningMode.VIDEO)
        try:
            index = 0
            while max_frames is None or index < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                records.append({"source": str(source_path), **_predict(detector.detect_for_video(image, int(index * 1000.0 / fps)), runner, index)})
                index += 1
        finally:
            detector.close()
            capture.release()
        return records
    raise FileNotFoundError(f"Unsupported offline source: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model-dir", default=str(ROOT / "outputs" / "models" / "rgb_native_v4"))
    parser.add_argument("--pose-model", default=str(ROOT / "metadata" / "pose_landmarker_heavy.task"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    records = infer(args.source, args.model_dir, args.pose_model, args.max_frames)
    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else "")
    if args.output == "-":
        print(payload, end="")
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
