"""MediaPipe Tasks adapter that preserves every pose returned by the detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from src.posture_ai.inference.pipeline import extract_pose_result

from .tracking import PoseDetection


def create_multi_pose_landmarker(
    model_path: str | Path,
    *,
    max_people: int = 3,
    running_mode: str = "video",
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> PoseLandmarker:
    """Create MediaPipe Tasks Pose Landmarker with an explicit pose count."""

    if int(max_people) < 1:
        raise ValueError("max_people must be at least 1")
    mode_name = str(running_mode).lower()
    if mode_name == "image":
        mode = RunningMode.IMAGE
    elif mode_name == "video":
        mode = RunningMode.VIDEO
    else:
        raise ValueError("running_mode must be 'image' or 'video' for the synchronous prototype")
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=mode,
        num_poses=int(max_people),
        min_pose_detection_confidence=float(min_detection_confidence),
        min_pose_presence_confidence=float(min_presence_confidence),
        min_tracking_confidence=float(min_tracking_confidence),
    )
    return PoseLandmarker.create_from_options(options)


class MultiPersonMediaPipeBackend:
    """Return one `PoseDetection` for each pose in a Tasks result."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        max_people: int = 3,
        running_mode: str = "video",
        visibility_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= float(visibility_threshold) <= 1.0:
            raise ValueError("visibility_threshold must be between 0 and 1")
        self.max_people = int(max_people)
        self.visibility_threshold = float(visibility_threshold)
        self.running_mode = str(running_mode).lower()
        self.landmarker = create_multi_pose_landmarker(
            model_path,
            max_people=self.max_people,
            running_mode=self.running_mode,
        )

    def detect(self, frame_bgr, *, timestamp_ms: int | None = None) -> list[PoseDetection]:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise ValueError("Empty image frame")
        result = extract_pose_result(frame_bgr, self.landmarker, timestamp_ms=timestamp_ms)
        image_poses = list(result.pose_landmarks or [])
        world_poses = list(result.pose_world_landmarks or [])
        detections = []
        for index, image_landmarks in enumerate(image_poses[: self.max_people]):
            world_landmarks: Any | None = world_poses[index] if index < len(world_poses) else None
            detections.append(
                PoseDetection.from_mediapipe(
                    image_landmarks,
                    world_landmarks,
                    visibility_threshold=self.visibility_threshold,
                )
            )
        return detections

    def close(self) -> None:
        self.landmarker.close()
