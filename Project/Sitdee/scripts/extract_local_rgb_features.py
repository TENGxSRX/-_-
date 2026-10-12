"""Extract the frozen RGB-native V4/V5 features from local videos.

No classifier is loaded or trained here. The label in each output row is the
controlled participant-instruction label encoded by the validated filename.
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

from src.posture_ai.data.local_rgb_v1 import LOCAL_VIDEO_SUFFIXES, validate_video_path  # noqa: E402
from src.posture_ai.data.rgb_native_v5 import sample_frame_indices  # noqa: E402
from src.posture_ai.features.rgb_native_v5 import (  # noqa: E402
    V5_CORE_INDICES,
    V5_FEATURE_FAMILIES,
    build_rgb_native_v5_features,
    v5_feature_names,
)
from src.posture_ai.inference.pipeline import create_pose_landmarker, extract_pose_result  # noqa: E402

RAW_ROOT = ROOT / "datasets" / "raw" / "local_rgb_v1"
OUTPUT_ROOT = ROOT / "datasets" / "processed" / "local_rgb_v1" / "features"
CONFIG_PATH = ROOT / "configs" / "local_collection.yaml"
POSE_MODEL = ROOT / "metadata" / "pose_landmarker_heavy.task"
FRAME_COLUMNS = (
    "frame_sample_id",
    "recording_sample_id",
    "participant_id",
    "session_id",
    "video_path",
    "project_label",
    "frame_index",
    "timestamp_seconds",
    "pose_detected",
    "pose_quality",
    "core_accepted",
    "status",
)


def _world_xyz(landmarks) -> np.ndarray | None:
    if landmarks is None or len(landmarks) != 33:
        return None
    values = np.asarray([[float(item.x), float(item.y), float(item.z)] for item in landmarks], dtype=np.float64)
    return values if np.isfinite(values).all() else None


def _write_frame_manifest(rows: list[dict[str, str]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "frame_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def extract(*, target_fps: float = 2.0, exclude_initial_seconds: float = 2.0, max_frames_per_video: int | None = None) -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    visibility_threshold = float(config.get("core_visibility_threshold", 0.5))
    video_paths = sorted(
        path for path in RAW_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in LOCAL_VIDEO_SUFFIXES
    ) if RAW_ROOT.exists() else []
    feature_values: dict[str, list[np.ndarray]] = {family: [] for family in V5_FEATURE_FAMILIES}
    feature_row_indices: list[int] = []
    frame_rows: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    videos_processed = 0
    for path in video_paths:
        parsed, errors = validate_video_path(path, RAW_ROOT)
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        if parsed is None or errors:
            invalid.append({"video_path": relative, "errors": "; ".join(errors)})
            continue
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            invalid.append({"video_path": relative, "errors": "unreadable_video"})
            continue
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total_frames <= 0:
            capture.release()
            invalid.append({"video_path": relative, "errors": "invalid_fps_or_frame_count"})
            continue
        indices = [int(index) for index in sample_frame_indices(total_frames, fps, target_fps) if index >= int(np.ceil(exclude_initial_seconds * fps))]
        if max_frames_per_video is not None:
            indices = indices[:max_frames_per_video]
        index_set = set(indices)
        detector = create_pose_landmarker(POSE_MODEL, running_mode="video")
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index in index_set:
                    recording_id = f"{parsed['participant']}_{parsed['session']}_{parsed['posture']}_01"
                    frame_id = f"{recording_id}__F{frame_index:06d}"
                    result = extract_pose_result(frame, detector, timestamp_ms=int(frame_index * 1000.0 / fps))
                    detected = bool(result.pose_landmarks and result.pose_world_landmarks)
                    quality = ""
                    core_accepted = False
                    status = "POSE_NOT_DETECTED"
                    if detected:
                        image_landmarks = result.pose_landmarks[0]
                        visibility = np.asarray([float(item.visibility) for item in image_landmarks], dtype=float)
                        core_visibility = visibility[list(V5_CORE_INDICES)]
                        quality = f"{float(np.mean(core_visibility)):.6f}"
                        world_xyz = _world_xyz(result.pose_world_landmarks[0])
                        core_accepted = world_xyz is not None and np.all(core_visibility >= visibility_threshold)
                        status = "FEATURES_EXTRACTED" if core_accepted else "CORE_REJECTED"
                        if core_accepted:
                            frame_row_index = len(frame_rows)
                            for family in V5_FEATURE_FAMILIES:
                                values, names = build_rgb_native_v5_features(world_xyz, family)
                                if names != v5_feature_names(family):
                                    raise ValueError(f"Feature schema mismatch for {family}")
                                feature_values[family].append(values[0].astype(np.float32, copy=False))
                            feature_row_indices.append(frame_row_index)
                    frame_rows.append(
                        {
                            "frame_sample_id": frame_id,
                            "recording_sample_id": recording_id,
                            "participant_id": parsed["participant"],
                            "session_id": parsed["session"],
                            "video_path": relative,
                            "project_label": parsed["posture"],
                            "frame_index": str(frame_index),
                            "timestamp_seconds": f"{frame_index / fps:.6f}",
                            "pose_detected": "1" if detected else "0",
                            "pose_quality": quality,
                            "core_accepted": "1" if core_accepted else "0",
                            "status": status,
                        }
                    )
                frame_index += 1
        finally:
            detector.close()
            capture.release()
        videos_processed += 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_frame_manifest(frame_rows)
    arrays = {
        family: np.asarray(values, dtype=np.float32).reshape(-1, len(v5_feature_names(family)))
        for family, values in feature_values.items()
    }
    arrays["frame_manifest_row_indices"] = np.asarray(feature_row_indices, dtype=np.int64)
    np.savez_compressed(OUTPUT_ROOT / "features.npz", **arrays)
    (OUTPUT_ROOT / "feature_config.json").write_text(
        json.dumps(
            {
                "pipeline": "local_rgb_v1",
                "coordinate_representation": "MediaPipe Pose World Landmarks",
                "transformation": "hip midpoint translation; frozen RGB-native V4/V5 feature definitions",
                "core_landmarks": ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"],
                "visibility_threshold": visibility_threshold,
                "sampling_fps": target_fps,
                "exclude_initial_seconds": exclude_initial_seconds,
                "families": {family: v5_feature_names(family) for family in V5_FEATURE_FAMILIES},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    status = "WAITING_FOR_LOCAL_RGB_DATA" if not video_paths else ("EXTRACTED_WITH_WARNINGS" if invalid or any(row["status"] != "FEATURES_EXTRACTED" for row in frame_rows) else "EXTRACTED")
    report = {
        "status": status,
        "videos_found": len(video_paths),
        "videos_processed": videos_processed,
        "sampled_frames": len(frame_rows),
        "feature_eligible_frames": len(feature_row_indices),
        "invalid_videos": invalid,
        "output": str(OUTPUT_ROOT),
        "raw_media_modified": False,
    }
    (OUTPUT_ROOT / "run_status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--exclude-initial-seconds", type=float, default=2.0)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    args = parser.parse_args()
    report = extract(target_fps=args.target_fps, exclude_initial_seconds=args.exclude_initial_seconds, max_frames_per_video=args.max_frames_per_video)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
