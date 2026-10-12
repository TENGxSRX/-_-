"""Pose backend interface for MediaPipe now and alternate backends later."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .landmarks import PoseLandmarks


class PoseBackend(Protocol):
    """Minimal contract required by the ergonomic engine."""

    def detect(self, frame_bgr: np.ndarray, *, timestamp_ms: int | None = None) -> PoseLandmarks | None:
        ...

    def close(self) -> None:
        ...


class MediaPipePoseBackend:
    """Adapter around the pinned MediaPipe Tasks Pose Landmarker asset."""

    def __init__(self, model_path, *, running_mode: str = "image"):
        from src.posture_ai.inference.pipeline import create_pose_landmarker

        self._landmarker = create_pose_landmarker(model_path, running_mode=running_mode)

    def detect(self, frame_bgr: np.ndarray, *, timestamp_ms: int | None = None) -> PoseLandmarks | None:
        from src.posture_ai.inference.pipeline import extract_pose_landmarks

        result = extract_pose_landmarks(frame_bgr, self._landmarker, timestamp_ms=timestamp_ms)
        return None if result is None else PoseLandmarks.from_any(result)

    def close(self) -> None:
        self._landmarker.close()
