"""Offline ERGONOMIC POSTURE ENGINE V1 runner.

The runner accepts images, image directories, and prerecorded videos. It writes
neutral telemetry and optional annotated video; it never opens webcam device 0.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from src.posture_ai.ergonomics.geometry import calculate_geometry
from src.posture_ai.ergonomics.indicators import evaluate_indicators
from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig, TemporalRiskEngine
from src.posture_ai.ergonomics.session import SessionAccumulator, write_jsonl, write_summary
from src.posture_ai.pose.landmarks import PoseLandmarks
from src.posture_ai.pose.validation import PoseValidationError

from .pipeline import IMAGE_SUFFIXES, VIDEO_SUFFIXES, InferenceRunner, create_pose_landmarker, extract_pose_landmarks


SKELETON_CONNECTIONS = (
    (0, 11), (0, 12), (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (27, 29), (27, 31), (28, 30), (28, 32),
)


def _image_paths(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return []


def _draw_annotation(frame: np.ndarray, landmarks, record: dict) -> np.ndarray:
    output = frame.copy()
    if landmarks is not None:
        height, width = output.shape[:2]
        points = PoseLandmarks.from_any(landmarks).points
        coords: dict[int, tuple[int, int]] = {}
        for index, point in enumerate(points):
            if point is None or not point.present:
                continue
            x = int(np.clip(point.x, 0.0, 1.0) * max(width - 1, 1))
            y = int(np.clip(point.y, 0.0, 1.0) * max(height - 1, 1))
            coords[index] = (x, y)
            cv2.circle(output, (x, y), 3, (0, 220, 255), -1)
        for left, right in SKELETON_CONNECTIONS:
            if left in coords and right in coords:
                cv2.line(output, coords[left], coords[right], (255, 180, 0), 2)

    direction = record.get("direction") or "NO POSE"
    confidence = record.get("direction_confidence")
    confidence_text = "n/a" if confidence is None else f"{100.0 * float(confidence):.0f}%"
    duration = float(record.get("continuous_duration", 0.0) or 0.0)
    risk = record.get("risk_level", "POSE_LOST")
    indicators = ", ".join(record.get("risk_indicators", [])) or "none"
    lines = [
        f"SOURCE LABEL: {record.get('source_label') or 'UNLABELED'}",
        f"POSTURE: {direction}",
        f"CONFIDENCE: {confidence_text}",
        f"DURATION: {duration:.1f} sec",
        f"STATUS: {risk}",
        f"INDICATORS: {indicators}",
    ]
    y = 28
    for line in lines:
        cv2.putText(output, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(output, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 25
    return output


def _safe_record(engine: TemporalRiskEngine, timestamp: float, *, status: str, geometry: dict | None = None, landmarks=None) -> dict:
    state = engine.update(timestamp, pose_visible=False, geometry=geometry or {})
    return {"frame": None, "status": status, "adapter_status": "REAL_IMAGE_ADAPTER_NOT_VALIDATED", **state}


def run_posture_engine(
    source: str | Path,
    model_dir: str | Path,
    pose_model: str | Path,
    *,
    config_path: str | Path = "configs/posture_engine.yaml",
    max_frames: int | None = None,
    annotated_video: str | Path | None = None,
    event_log: str | Path | None = None,
    session_report: str | Path | None = None,
    performance_report: str | Path | None = None,
    source_label: str | None = None,
) -> dict:
    """Run V1 on offline media and return paths, telemetry, and performance."""

    source_text = str(source)
    if source_text == "0":
        raise ValueError("Webcam source 0 is intentionally disabled in this offline V1 runner")
    source_path = Path(source)
    config = PostureEngineConfig.from_yaml(config_path)
    runner = InferenceRunner(model_dir)
    engine = TemporalRiskEngine(config, runner.labels)
    session = SessionAccumulator(frame_interval_seconds=config.frame_interval_seconds)
    records: list[dict] = []
    total_pose_seconds = 0.0
    total_model_seconds = 0.0
    total_wall_start = time.perf_counter()
    process = psutil.Process(os.getpid())
    rss_start_bytes = process.memory_info().rss
    frame_count = 0
    pose_detected_count = 0
    valid_count = 0
    adapter_execution_count = 0
    low_visibility_count = 0
    low_confidence_count = 0
    missing_landmark_total = 0
    visible_landmark_total = 0
    visibility_sum = 0.0
    visibility_value_count = 0
    writer = None
    video_fps = config.frame_interval_seconds ** -1
    is_video = source_path.is_file() and source_path.suffix.lower() in VIDEO_SUFFIXES

    def process_frame(frame: np.ndarray, frame_index: int, timestamp: float) -> tuple[dict, object | None]:
        nonlocal total_pose_seconds, total_model_seconds, pose_detected_count, valid_count, adapter_execution_count
        nonlocal low_visibility_count, low_confidence_count, missing_landmark_total, visible_landmark_total
        nonlocal visibility_sum, visibility_value_count
        base = {
            "frame": frame_index,
            "timestamp_seconds": timestamp,
            "source": str(source_path),
            "source_label": source_label or "UNLABELED",
            "adapter_status": "REAL_IMAGE_ADAPTER_NOT_VALIDATED",
        }
        if frame is None or frame.size == 0:
            record = {**base, **engine.update(timestamp, pose_visible=False)}
            record["status"] = "invalid_image"
            return record, None
        pose_start = time.perf_counter()
        landmarks = extract_pose_landmarks(frame, landmarker, timestamp_ms=int(timestamp * 1000.0) if is_video else None)
        total_pose_seconds += time.perf_counter() - pose_start
        if landmarks is None:
            record = {**base, **engine.update(timestamp, pose_visible=False)}
            record["status"] = "no_pose_detected"
            return record, None
        pose_detected_count += 1
        canonical_pose = PoseLandmarks.from_any(landmarks)
        missing_landmark_total += len(canonical_pose.missing_names())
        visibility_values = canonical_pose.visibility_array()
        finite_visibility = visibility_values[np.isfinite(visibility_values)]
        if finite_visibility.size:
            visibility_sum += float(finite_visibility.sum())
            visibility_value_count += int(finite_visibility.size)
            visible_landmark_total += int(np.count_nonzero(finite_visibility >= config.minimum_pose_visibility))
        geometry = calculate_geometry(landmarks)
        try:
            model_start = time.perf_counter()
            prediction = runner.predict_landmarks(landmarks, visibility_threshold=config.minimum_pose_visibility)
            total_model_seconds += time.perf_counter() - model_start
        except (ValueError, PoseValidationError) as exc:
            low_visibility_count += 1
            record = {**base, **engine.update(timestamp, pose_visible=False, geometry=geometry if "geometry" in locals() else {})}
            record["status"] = "low_visibility_or_invalid_landmarks"
            record["detail"] = str(exc)
            return record, landmarks
        adapter_execution_count += 1
        if prediction["confidence"] < config.minimum_confidence:
            low_confidence_count += 1
            record = {**base, **engine.update(timestamp, pose_visible=False, geometry=geometry)}
            record.update({"status": "low_model_confidence", "direction_candidate": prediction["label"], "direction_candidate_confidence": prediction["confidence"]})
            return record, landmarks
        pose = PoseLandmarks.from_any(landmarks)
        indicators = evaluate_indicators(geometry, prediction["label"], confidence=prediction["confidence"], thresholds=config.indicator_thresholds)
        state = engine.update(
            timestamp,
            pose_visible=True,
            probabilities=np.asarray([prediction["probabilities"][label] for label in runner.labels], dtype=float),
            confidence=prediction["confidence"],
            geometry=geometry,
            indicators=indicators,
        )
        valid_count += 1
        record = {
            **base,
            "status": "ok",
            "model_label": prediction["label"],
            "model_display_label": prediction["display_label"],
            "model_feature_count": prediction["feature_count"],
            **state,
        }
        return record, pose

    if is_video:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source_path}")
        video_fps = float(capture.get(cv2.CAP_PROP_FPS) or (1.0 / config.frame_interval_seconds))
        video_fps = video_fps if video_fps > 0 else 30.0
        session.frame_interval_seconds = 1.0 / video_fps
        video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if annotated_video is None:
            annotated_video = Path("outputs/videos") / f"{source_path.stem}_analyzed.mp4"
        annotated_video = Path(annotated_video)
        annotated_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(annotated_video), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (video_width, video_height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create annotated video: {annotated_video}")
        landmarker = create_pose_landmarker(pose_model, running_mode="video")
        try:
            while max_frames is None or frame_count < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_count / video_fps
                record, landmarks = process_frame(frame, frame_count, timestamp)
                writer.write(_draw_annotation(frame, landmarks, record))
                records.append(record)
                session.add(record)
                frame_count += 1
        finally:
            landmarker.close()
            capture.release()
            writer.release()
    else:
        image_paths = _image_paths(source_path)
        if not image_paths:
            raise FileNotFoundError(f"Source is not a supported image, directory, or prerecorded video: {source}")
        landmarker = create_pose_landmarker(pose_model, running_mode="image")
        try:
            for frame_index, image_path in enumerate(image_paths):
                if max_frames is not None and frame_count >= max_frames:
                    break
                frame = cv2.imread(str(image_path))
                record, landmarks = process_frame(frame, frame_index, frame_index * config.frame_interval_seconds)
                record["source"] = str(image_path)
                records.append(record)
                session.add(record)
                frame_count += 1
        finally:
            landmarker.close()

    elapsed = time.perf_counter() - total_wall_start
    rss_end_bytes = process.memory_info().rss
    inferred_duration = frame_count / video_fps if is_video else frame_count * config.frame_interval_seconds
    summary = session.summary(total_duration_seconds=inferred_duration)
    performance = {
        "source": str(source_path),
        "frames": frame_count,
        "pose_detected_frames": pose_detected_count,
        "valid_pose_frames": valid_count,
        "pose_detection_success_rate": pose_detected_count / frame_count if frame_count else 0.0,
        "accepted_pose_rate_after_visibility_and_confidence": valid_count / frame_count if frame_count else 0.0,
        "adapter_execution_frames": adapter_execution_count,
        "adapter_execution_success_rate_given_detected": adapter_execution_count / pose_detected_count if pose_detected_count else 0.0,
        "low_visibility_rejection_frames": low_visibility_count,
        "low_model_confidence_frames": low_confidence_count,
        "missing_landmark_frequency_per_detected_landmark": missing_landmark_total / (pose_detected_count * 33) if pose_detected_count else 0.0,
        "mean_landmark_visibility": visibility_sum / visibility_value_count if visibility_value_count else None,
        "visible_landmark_fraction_at_config_threshold": visible_landmark_total / (visibility_value_count or 1),
        "video_fps_input": video_fps if is_video else None,
        "wall_seconds": elapsed,
        "complete_pipeline_fps": frame_count / elapsed if elapsed > 0 else 0.0,
        "rss_start_mb": rss_start_bytes / (1024.0 ** 2),
        "rss_end_mb": rss_end_bytes / (1024.0 ** 2),
        "rss_delta_mb": (rss_end_bytes - rss_start_bytes) / (1024.0 ** 2),
        "mean_pose_extraction_ms": 1000.0 * total_pose_seconds / frame_count if frame_count else 0.0,
        "mean_adapter_classifier_ms": 1000.0 * total_model_seconds / valid_count if valid_count else 0.0,
        "note": "Offline benchmark only; no classification accuracy is calculated for unlabeled real images.",
    }

    stem = source_path.stem if source_path.suffix else source_path.name
    event_log = Path(event_log or (Path("outputs/logs") / f"{stem}_events.jsonl"))
    session_report = Path(session_report or (Path("outputs/session_reports") / f"{stem}_summary.json"))
    performance_report = Path(performance_report or (Path("outputs/metrics") / f"{stem}_performance.json"))
    write_jsonl(event_log, records)
    write_summary(session_report, summary)
    write_summary(performance_report, performance)
    return {
        "records": records,
        "summary": summary,
        "performance": performance,
        "event_log": str(event_log),
        "session_report": str(session_report),
        "performance_report": str(performance_report),
        "annotated_video": None if annotated_video is None else str(annotated_video),
    }
