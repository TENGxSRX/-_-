"""Offline MediaPipe-to-model inference pipeline.

The implementation supports image files, image directories, video files, and
future webcam source ``0``. This module never opens a camera unless the caller
explicitly supplies the string ``0`` to ``infer_source``.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from src.posture_ai.pose.adapter import adapt_mediapipe_to_multiposture
from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner
from src.posture_ai.inference.rgb_native_v5 import RGBNativeV5Runner


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def create_pose_landmarker(model_path: str | Path, *, running_mode: str = "image", min_detection_confidence: float = 0.5, min_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
    mode = RunningMode.VIDEO if running_mode == "video" else RunningMode.IMAGE
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=mode,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return PoseLandmarker.create_from_options(options)


def extract_pose_result(frame_bgr: np.ndarray, landmarker, *, timestamp_ms: int | None = None):
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Empty image frame")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if timestamp_ms is None:
        result = landmarker.detect(image)
    else:
        result = landmarker.detect_for_video(image, timestamp_ms)
    return result


def extract_pose_landmarks(frame_bgr: np.ndarray, landmarker, *, timestamp_ms: int | None = None):
    result = extract_pose_result(frame_bgr, landmarker, timestamp_ms=timestamp_ms)
    if not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]


class TemporalSmoother:
    """Rolling probability smoother with configurable persistence."""

    def __init__(self, labels, *, window_size: int = 5, persistence_frames: int = 3, alpha: float | None = None):
        if window_size < 1 or persistence_frames < 1:
            raise ValueError("window_size and persistence_frames must be positive")
        self.labels = list(labels)
        self.window_size = window_size
        self.persistence_frames = persistence_frames
        self.alpha = alpha
        self.history: deque[np.ndarray] = deque(maxlen=window_size)
        self.candidate = None
        self.candidate_count = 0
        self.stable_label = None

    def update(self, probabilities: np.ndarray) -> dict:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.shape != (len(self.labels),):
            raise ValueError(f"Expected probability vector of shape {(len(self.labels),)}, got {probabilities.shape}")
        if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
            raise ValueError("Probabilities must be finite and have a positive sum")
        probabilities = probabilities / probabilities.sum()
        self.history.append(probabilities)
        if self.alpha is not None and len(self.history) > 1:
            smoothed = self.history[0].copy()
            for current in list(self.history)[1:]:
                smoothed = self.alpha * current + (1.0 - self.alpha) * smoothed
        else:
            smoothed = np.mean(np.vstack(self.history), axis=0)
        candidate = self.labels[int(np.argmax(smoothed))]
        if candidate == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = candidate
            self.candidate_count = 1
        if self.stable_label is None or self.candidate_count >= self.persistence_frames:
            self.stable_label = candidate
        return {
            "label": self.stable_label,
            "candidate_label": candidate,
            "confidence": float(np.max(smoothed)),
            "probabilities": {label: float(smoothed[i]) for i, label in enumerate(self.labels)},
            "persistence_frames": self.candidate_count,
        }


class InferenceRunner:
    """Load an exported package and run a single frame through it."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "model.joblib")
        self.feature_config = json.loads((self.model_dir / "feature_config.json").read_text(encoding="utf-8"))
        self.labels = json.loads((self.model_dir / "labels.json").read_text(encoding="utf-8"))["source_labels"]
        self.expected_feature_names = self.feature_config["columns"]

    def predict_landmarks(self, landmarks, *, visibility_threshold: float = 0.5) -> dict:
        adapted = adapt_mediapipe_to_multiposture(
            landmarks,
            visibility_threshold=visibility_threshold,
            model_dir=self.model_dir,
        )
        features, names = adapted.values, list(adapted.names)
        if names != self.expected_feature_names:
            raise ValueError("Feature adapter output order does not match the exported model schema")
        prediction = self.model.predict(features)[0]
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
        else:
            decision = np.asarray(self.model.decision_function(features))[0]
            decision = np.atleast_1d(decision)
            probabilities = np.exp(decision - np.max(decision))
            probabilities = probabilities / probabilities.sum()
        probability_by_label = {str(label): float(probabilities[i]) for i, label in enumerate(self.model.classes_)}
        return {
            "label": str(prediction),
            "display_label": self._display_label(str(prediction)),
            "confidence": float(np.max(probabilities)),
            "probabilities": probability_by_label,
            "feature_config": self.feature_config["name"],
            "feature_count": int(features.shape[1]),
            "adapter_status": "REAL_IMAGE_ADAPTER_NOT_VALIDATED",
            "adapter_source_convention": adapted.source_convention,
        }

    def _display_label(self, label: str) -> str:
        return {"TUP": "Upright", "TLF": "Lean forward", "TLB": "Lean backward", "TLL": "Lean left", "TLR": "Lean right"}.get(label, label)


def _iter_image_paths(source: Path):
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        yield source
    elif source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield path


def infer_source(source: str | Path, model_dir: str | Path, pose_model: str | Path, *, min_visibility: float = 0.5, min_confidence: float = 0.5, window_size: int = 5, persistence_frames: int = 3, max_frames: int | None = None, backend: str | None = None) -> list[dict]:
    """Run offline inference and return JSON-serializable predictions."""
    source_text = str(source)
    if backend == "rgb_native_v4":
        runner = RGBNativeV4Runner(model_dir)
    elif backend == "rgb_native_v5":
        runner = RGBNativeV5Runner(model_dir)
    else:
        runner = InferenceRunner(model_dir)
    predictions: list[dict] = []
    smoother = TemporalSmoother(runner.labels, window_size=window_size, persistence_frames=persistence_frames)

    if source_text == "0":
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Webcam source 0 could not be opened")
        landmarker = create_pose_landmarker(pose_model, running_mode="video")
        try:
            frame_index = 0
            while max_frames is None or frame_index < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                predictions.append(_predict_frame(frame, frame_index, runner, landmarker, smoother, min_confidence, min_visibility))
                frame_index += 1
        finally:
            landmarker.close()
            capture.release()
        return predictions

    path = Path(source)
    image_paths = list(_iter_image_paths(path))
    if image_paths:
        landmarker = create_pose_landmarker(pose_model, running_mode="image")
        try:
            for index, image_path in enumerate(image_paths):
                if max_frames is not None and index >= max_frames:
                    break
                frame = cv2.imread(str(image_path))
                record = _predict_frame(frame, index, runner, landmarker, None, min_confidence, min_visibility)
                record["source"] = str(image_path)
                predictions.append(record)
        finally:
            landmarker.close()
        return predictions

    if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {path}")
        landmarker = create_pose_landmarker(pose_model, running_mode="video")
        try:
            frame_index = 0
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            while max_frames is None or frame_index < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_ms = int(frame_index * 1000.0 / fps)
                record = _predict_frame(frame, frame_index, runner, landmarker, smoother, min_confidence, min_visibility, timestamp_ms=timestamp_ms)
                record["source"] = str(path)
                predictions.append(record)
                frame_index += 1
        finally:
            landmarker.close()
            capture.release()
        return predictions
    raise FileNotFoundError(f"Source is not a supported image, directory, video, or explicit webcam source 0: {source}")


def _predict_frame(frame, frame_index, runner, landmarker, smoother, min_confidence, min_visibility, *, timestamp_ms=None):
    record = {"frame": frame_index}
    if frame is None:
        record["status"] = "invalid_image"
        return record
    result = extract_pose_result(frame, landmarker, timestamp_ms=timestamp_ms) if hasattr(runner, "predict_world_landmarks") else None
    if result is not None:
        if not result.pose_landmarks or not result.pose_world_landmarks:
            record["status"] = "no_world_pose_detected"
            return record
        landmarks = result.pose_landmarks[0]
        world_landmarks = result.pose_world_landmarks[0]
    else:
        landmarks = extract_pose_landmarks(frame, landmarker, timestamp_ms=timestamp_ms)
        world_landmarks = None
    if landmarks is None:
        record["status"] = "no_pose_detected"
        return record
    try:
        if world_landmarks is not None:
            prediction = runner.predict_world_landmarks(
                world_landmarks,
                image_visibility=landmarks,
                visibility_threshold=min_visibility,
            )
        else:
            prediction = runner.predict_landmarks(landmarks, visibility_threshold=min_visibility)
    except ValueError as exc:
        record["status"] = "low_visibility_or_invalid_landmarks"
        record["detail"] = str(exc)
        return record
    if prediction["confidence"] < min_confidence:
        record.update({"status": "low_model_confidence", **prediction})
        return record
    record.update({"status": "ok", **prediction})
    if smoother is not None:
        record["temporal"] = smoother.update(np.asarray([prediction["probabilities"][label] for label in runner.labels], dtype=float))
    return record
