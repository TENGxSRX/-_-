"""Run a sampled MediaPipe World Landmark pre-flight on local videos.

This script measures pose extraction only. It never uses a classifier to decide
the participant-instruction label and never modifies raw video.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.posture_ai.data.local_rgb_v1 import LOCAL_MANIFEST_COLUMNS, LOCAL_VIDEO_SUFFIXES  # noqa: E402
from src.posture_ai.data.rgb_native_v5 import sample_frame_indices  # noqa: E402
from src.posture_ai.features.rgb_native_v5 import V5_CORE_INDICES  # noqa: E402
from src.posture_ai.inference.pipeline import create_pose_landmarker, extract_pose_result  # noqa: E402

RAW_ROOT = ROOT / "datasets" / "raw" / "local_rgb_v1"
MANIFEST_PATH = ROOT / "datasets" / "processed" / "local_rgb_v1" / "manifest.csv"
CONFIG_PATH = ROOT / "configs" / "local_collection.yaml"
POSE_MODEL = ROOT / "metadata" / "pose_landmarker_heavy.task"
OUTPUT_PATH = ROOT / "outputs" / "metrics" / "local_rgb_v1_pose_preflight.json"


def _read_manifest() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(rows: list[dict[str, str]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOCAL_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _pose_xyz(landmarks) -> np.ndarray | None:
    if landmarks is None or len(landmarks) != 33:
        return None
    xyz = np.asarray([[float(item.x), float(item.y), float(item.z)] for item in landmarks], dtype=np.float64)
    return xyz if np.isfinite(xyz).all() else None


def preflight(*, target_fps: float = 2.0, exclude_initial_seconds: float = 2.0, write_manifest: bool = True) -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    visibility_threshold = float(config.get("core_visibility_threshold", 0.5))
    minimum_pose_rate = float(config.get("minimum_pose_detection_rate", 0.90))
    minimum_core_rate = float(config.get("minimum_core_acceptance_rate", 0.85))
    minimum_duration = float(config.get("minimum_duration_seconds", 20.0))
    rows = _read_manifest()
    row_by_path = {row["video_path"].replace("\\", "/"): row for row in rows}
    video_paths = sorted(
        path for path in RAW_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in LOCAL_VIDEO_SUFFIXES
    ) if RAW_ROOT.exists() else []
    results: list[dict] = []
    if video_paths and not POSE_MODEL.exists():
        raise FileNotFoundError(f"MediaPipe Pose model asset not found: {POSE_MODEL}")
    for path in video_paths:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        capture = cv2.VideoCapture(str(path))
        opened = bool(capture.isOpened())
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
        duration = frame_count / fps if fps > 0 else 0.0
        if not opened or fps <= 0 or frame_count <= 0:
            capture.release()
            results.append({"video_path": relative, "status": "FAIL", "sampled_frames": 0, "pose_detected_frames": 0, "core_accepted_frames": 0, "pose_detection_rate": 0.0, "core_acceptance_rate": 0.0, "errors": ["unreadable_or_invalid_video"]})
            continue
        indices = set(int(index) for index in sample_frame_indices(frame_count, fps, target_fps) if index >= int(np.ceil(exclude_initial_seconds * fps)))
        detector = create_pose_landmarker(POSE_MODEL, running_mode="video")
        sampled = detected = core_accepted = 0
        quality_values: list[float] = []
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index in indices:
                    sampled += 1
                    result = extract_pose_result(frame, detector, timestamp_ms=int(frame_index * 1000.0 / fps))
                    if result.pose_landmarks and result.pose_world_landmarks:
                        detected += 1
                        image_landmarks = result.pose_landmarks[0]
                        visibility = np.asarray([float(item.visibility) for item in image_landmarks], dtype=float)
                        core_visibility = visibility[list(V5_CORE_INDICES)]
                        quality_values.append(float(np.mean(core_visibility)))
                        world_xyz = _pose_xyz(result.pose_world_landmarks[0])
                        if world_xyz is not None and np.isfinite(world_xyz).all() and np.all(core_visibility >= visibility_threshold):
                            core_accepted += 1
                frame_index += 1
        finally:
            detector.close()
            capture.release()
        pose_rate = detected / sampled if sampled else 0.0
        core_rate = core_accepted / sampled if sampled else 0.0
        errors: list[str] = []
        if duration < minimum_duration:
            errors.append("SHORTER_THAN_CONFIGURED_MINIMUM")
        if pose_rate < minimum_pose_rate:
            errors.append("POSE_DETECTION_BELOW_CONFIG")
        if core_rate < minimum_core_rate:
            errors.append("CORE_ACCEPTANCE_BELOW_CONFIG")
        status = "PASS" if not errors else "WARNING"
        record = {
            "video_path": relative,
            "status": status,
            "sampled_frames": sampled,
            "pose_detected_frames": detected,
            "core_accepted_frames": core_accepted,
            "pose_detection_rate": pose_rate,
            "core_acceptance_rate": core_rate,
            "mean_core_visibility": float(np.mean(quality_values)) if quality_values else 0.0,
            "duration_seconds": duration,
            "fps": fps,
            "errors": errors,
            "sampling_fps": target_fps,
            "exclude_initial_seconds": exclude_initial_seconds,
        }
        results.append(record)
        row = row_by_path.get(relative)
        if row is not None:
            row.update({
                "pose_detected_frames": str(detected),
                "pose_detection_rate": f"{pose_rate:.6f}",
                "core_accepted_frames": str(core_accepted),
                "core_acceptance_rate": f"{core_rate:.6f}",
                "status": status,
                "notes": (row.get("notes", "") + f" preflight_sampled_frames={sampled};" + ";".join(errors)).strip(),
            })
    report_status = "WAITING_FOR_LOCAL_RGB_DATA" if not video_paths else ("PASS" if all(item["status"] == "PASS" for item in results) else "WARNING")
    report = {
        "status": report_status,
        "videos_checked": len(results),
        "config": {
            "minimum_pose_detection_rate": minimum_pose_rate,
            "minimum_core_acceptance_rate": minimum_core_rate,
            "core_visibility_threshold": visibility_threshold,
            "minimum_duration_seconds": minimum_duration,
            "sampling_fps": target_fps,
            "exclude_initial_seconds": exclude_initial_seconds,
        },
        "videos": results,
        "raw_media_modified": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if write_manifest and rows:
        _write_manifest(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--exclude-initial-seconds", type=float, default=2.0)
    parser.add_argument("--no-write-manifest", action="store_true")
    args = parser.parse_args()
    report = preflight(target_fps=args.target_fps, exclude_initial_seconds=args.exclude_initial_seconds, write_manifest=not args.no_write_manifest)
    print(json.dumps({"status": report["status"], "videos_checked": report["videos_checked"], "output": str(OUTPUT_PATH)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
