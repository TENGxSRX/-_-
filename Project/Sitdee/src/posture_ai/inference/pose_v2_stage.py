"""Optional, budgeted Pose V2 telemetry stage for long sessions.

This module deliberately stops at pose evidence.  It does not load or call
the frozen V4 classifier and it never creates a posture label.  A single
global scheduler is shared by every track so ``max_pose_per_second`` is not
multiplied by the number of people in the frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.posture_ai.detection.yolo_optional import PersonBox, crop_frame
from src.posture_ai.pose.landmarks import PoseLandmarks


CORE_LANDMARK_NAMES = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")


POSE_OK = "POSE_OK"
POSE_CORE_INVALID = "POSE_CORE_INVALID"
POSE_UNAVAILABLE = "POSE_UNAVAILABLE"
POSE_PENDING = "POSE_PENDING"
POSE_AMBIGUOUS = "POSE_AMBIGUOUS"


def _padded_box(box: PersonBox, padding_fraction: float) -> PersonBox:
    padding = max(0.0, float(padding_fraction))
    return PersonBox(
        box.x1 - box.width * padding, box.y1 - box.height * padding,
        box.x2 + box.width * padding, box.y2 + box.height * padding,
        box.confidence, box.class_id, box.track_id,
    )


def _box_iou(first: PersonBox, second: PersonBox) -> float:
    left, top = max(first.x1, second.x1), max(first.y1, second.y1)
    right, bottom = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _pixel_crop_box(box: PersonBox, padding: float, width: int, height: int) -> PersonBox:
    """Match crop_frame's clamping and outward integer rounding exactly."""
    clipped = _padded_box(box, padding).clipped(width, height)
    return PersonBox(
        math.floor(clipped.x1), math.floor(clipped.y1),
        math.ceil(clipped.x2), math.ceil(clipped.y2),
        box.confidence, box.class_id, box.track_id,
    )


def _crop_contaminated(crop: PersonBox, boxes: list[PersonBox]) -> bool:
    # Conservative safety rule: any positive-area intersection can expose a
    # second person to MediaPipe. Edge contact alone does not contaminate.
    return any(
        (other.track_id is None or other.track_id != crop.track_id)
        and min(crop.x2, other.x2) > max(crop.x1, other.x1)
        and min(crop.y2, other.y2) > max(crop.y1, other.y1)
        for other in boxes
    )


def _ambiguous_track_ids(boxes: list[PersonBox], threshold: float) -> set[int]:
    ambiguous: set[int] = set()
    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            if _box_iou(first, second) >= threshold:
                ambiguous.update((int(first.track_id), int(second.track_id)))
    return ambiguous


@dataclass
class _TrackPoseEvidence:
    last_seen_at: float = -1e12
    last_attempt_at: float | None = None
    last_success_at: float | None = None
    last_pose_quality: float | None = None
    last_visibility: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class PoseV2Result:
    """One anonymous pose-stage observation; no posture classification."""

    track_id: int
    bbox: tuple[float, float, float, float]
    bbox_confidence: float
    pose_state: str
    sampling_reason: str
    pose_available: bool
    core_valid: bool
    pose_quality: float | None = None
    landmark_visibility: dict[str, float | None] = field(default_factory=dict)
    pose_age_seconds: float | None = None
    feature_schema_version: str = "POSE_V2_RAW_MEDIAPIPE_CROP_WORLD_V1"
    world_landmarks_xyz: list[list[float]] | None = None
    image_landmarks_xyz: list[list[float]] | None = None
    detail: str | None = None

    def to_record(self, *, session_id: str, timestamp_seconds: float, timestamp_utc: str, frame_index: int) -> dict[str, Any]:
        return {
            "schema_version": "pose_v2_telemetry_v1",
            "record_type": "POSE_V2_OBSERVATION",
            "session_id": str(session_id),
            "timestamp_seconds": float(timestamp_seconds),
            "timestamp_utc": str(timestamp_utc),
            "frame_index": int(frame_index),
            "track_id": int(self.track_id),
            "bbox": [float(value) for value in self.bbox],
            "bbox_confidence": float(self.bbox_confidence),
            "pose_state": self.pose_state,
            "sampling_reason": self.sampling_reason,
            "pose_available": bool(self.pose_available),
            "core_valid": bool(self.core_valid),
            "pose_quality": None if self.pose_quality is None else float(self.pose_quality),
            "landmark_visibility": dict(self.landmark_visibility),
            "feature_schema_version": self.feature_schema_version,
            "pose_age_seconds": None if self.pose_age_seconds is None else float(self.pose_age_seconds),
            "world_landmarks_xyz": self.world_landmarks_xyz,
            "image_landmarks_xyz": self.image_landmarks_xyz,
            "detail": self.detail,
            "v4_prediction": None,
            "label_status": "NOT_A_LABEL",
            "raw_video_recorded": False,
        }


class MediaPipeCropPoseBackend:
    """Lazy single-pose MediaPipe backend for the optional crop stage."""

    def __init__(self, model_path: str, *, visibility_threshold: float = 0.5, core_landmark_names: tuple[str, ...] = CORE_LANDMARK_NAMES):
        # Importing this module does not require MediaPipe.  The dependency is
        # constructed only when --enable-pose-v2 is explicitly supplied.
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

        self.visibility_threshold = float(visibility_threshold)
        if not core_landmark_names or any(name not in CORE_LANDMARK_NAMES for name in core_landmark_names):
            raise ValueError("Unsupported core landmark names")
        self.core_landmark_names = core_landmark_names
        self._cv2 = cv2
        self._mp = mp
        self._landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        ))

    def detect(self, crop: Any) -> list[Any]:
        if crop is None or getattr(crop, "size", 0) == 0:
            raise ValueError("Empty pose crop")
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=self._cv2.cvtColor(crop, self._cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect(image)
        image_items = list(result.pose_landmarks or [])
        world_items = list(result.pose_world_landmarks or [])
        if not image_items:
            return []
        image_pose = PoseLandmarks.from_any(image_items[0])
        world_pose = None if not world_items else PoseLandmarks.from_any(world_items[0])
        visibility = [image_pose.by_name(name).visibility for name in self.core_landmark_names]
        core_valid = all(
            value is not None and math.isfinite(float(value)) and float(value) >= self.visibility_threshold
            and image_pose.by_name(name) is not None and image_pose.by_name(name).present
            for name, value in zip(self.core_landmark_names, visibility)
        )
        quality = float(sum(max(0.0, min(1.0, float(value or 0.0))) for value in visibility) / len(visibility))
        return [type("PoseDetectionEvidence", (), {
            "landmarks": image_pose,
            "world_landmarks": world_pose,
            "pose_quality": quality,
            "core_valid": core_valid,
        })()]

    def close(self) -> None:
        self._landmarker.close()


class PoseV2Stage:
    """Run optional MediaPipe pose evidence under one global crop budget."""

    def __init__(
        self,
        pose_backend: Any,
        *,
        max_pose_per_second: float = 2.0,
        crop_padding: float = 0.10,
        min_crop_width: int = 80,
        min_crop_height: int = 120,
        ambiguity_iou: float = 0.25,
        visibility_threshold: float = 0.5,
        expire_after_seconds: float = 2.0,
    ) -> None:
        if not math.isfinite(float(max_pose_per_second)) or float(max_pose_per_second) <= 0.0:
            raise ValueError("max_pose_per_second must be positive and finite")
        if min_crop_width < 1 or min_crop_height < 1:
            raise ValueError("minimum crop dimensions must be positive")
        if not 0.0 <= float(visibility_threshold) <= 1.0:
            raise ValueError("visibility_threshold must be between 0 and 1")
        self.pose_backend = pose_backend
        self.max_pose_per_second = float(max_pose_per_second)
        self.interval_seconds = 1.0 / self.max_pose_per_second
        self.crop_padding = max(0.0, float(crop_padding))
        self.min_crop_width = int(min_crop_width)
        self.min_crop_height = int(min_crop_height)
        self.ambiguity_iou = float(ambiguity_iou)
        self.visibility_threshold = float(visibility_threshold)
        self.expire_after_seconds = float(expire_after_seconds)
        self._next_allowed_at = -1e12
        self._tracks: dict[int, _TrackPoseEvidence] = {}
        self.attempts = 0
        self.successes = 0
        self.counts: dict[str, int] = {}

    def _state(self, track_id: int, now: float) -> _TrackPoseEvidence:
        state = self._tracks.setdefault(int(track_id), _TrackPoseEvidence())
        state.last_seen_at = now
        return state

    def _age(self, state: _TrackPoseEvidence, now: float) -> float | None:
        if state.last_success_at is None:
            return None
        return max(0.0, now - state.last_success_at)

    def _record_count(self, state: str) -> None:
        self.counts[state] = self.counts.get(state, 0) + 1

    @staticmethod
    def _landmark_xyz(landmarks: Any) -> list[list[float]] | None:
        if landmarks is None:
            return None
        result: list[list[float]] = []
        for point in landmarks.points:
            if point is None or not point.present:
                result.append([])
            else:
                result.append([float(point.x), float(point.y), float(point.z)])
        return result

    @staticmethod
    def _visibility(landmarks: Any) -> dict[str, float | None]:
        output: dict[str, float | None] = {}
        for name in CORE_LANDMARK_NAMES:
            point = landmarks.by_name(name) if landmarks is not None else None
            output[name] = None if point is None or point.visibility is None else float(point.visibility)
        return output

    def _result(self, box: PersonBox, state: _TrackPoseEvidence, now: float, pose_state: str, reason: str, *, sampled: bool = False, pose_quality: float | None = None, visibility: dict[str, float | None] | None = None, core_valid: bool = False, detection: PoseDetection | None = None, detail: str | None = None) -> PoseV2Result:
        self._record_count(pose_state)
        age = self._age(state, now)
        return PoseV2Result(
            track_id=int(box.track_id), bbox=(box.x1, box.y1, box.x2, box.y2),
            bbox_confidence=float(box.confidence), pose_state=pose_state,
            sampling_reason=reason, pose_available=pose_state == POSE_OK,
            core_valid=bool(core_valid), pose_quality=pose_quality,
            landmark_visibility=dict(visibility or {}), pose_age_seconds=age,
            world_landmarks_xyz=self._landmark_xyz(detection.world_landmarks) if detection is not None else None,
            image_landmarks_xyz=self._landmark_xyz(detection.landmarks) if detection is not None else None,
            detail=detail,
        )

    def _expire(self, now: float) -> None:
        for track_id, state in list(self._tracks.items()):
            if now - state.last_seen_at > self.expire_after_seconds:
                del self._tracks[track_id]

    def process(self, frame: Any, selected_boxes: list[PersonBox], timestamp_seconds: float, *, all_boxes: list[PersonBox] | None = None) -> list[PoseV2Result]:
        """Process selected boxes while checking ambiguity against all boxes.

        ``all_boxes`` is the current full tracked-person set.  It is used only
        for safety/overlap checks; unselected boxes never consume pose budget
        and never produce pose records. Safety checks use the exact padded,
        clamped, integer crop passed to MediaPipe, before spending a token.
        """

        now = float(timestamp_seconds)
        if not math.isfinite(now):
            raise ValueError("timestamp_seconds must be finite")
        tracked = [box for box in selected_boxes if box.track_id is not None]
        safety_boxes = all_boxes if all_boxes is not None else selected_boxes
        self._expire(now)
        states = {int(box.track_id): self._state(int(box.track_id), now) for box in tracked}
        ambiguous_all = _ambiguous_track_ids([box for box in safety_boxes if box.track_id is not None], self.ambiguity_iou)
        height, width = frame.shape[:2]
        crop_boxes = {int(box.track_id): _pixel_crop_box(box, self.crop_padding, width, height) for box in tracked}
        results: dict[int, PoseV2Result] = {}

        for box in tracked:
            track_id = int(box.track_id)
            if _crop_contaminated(crop_boxes[track_id], safety_boxes):
                ambiguous_all.add(track_id)
                results[track_id] = self._result(crop_boxes[track_id], states[track_id], now, POSE_AMBIGUOUS, "PADDED_CROP_REJECTED", detail="PADDED_CROP_PERSON_CONTAMINATION")
            elif track_id in ambiguous_all:
                results[track_id] = self._result(box, states[track_id], now, POSE_AMBIGUOUS, "OVERLAP_REJECTED", detail="OVERLAPPING_PERSON_BOXES")

        eligible = [box for box in tracked if int(box.track_id) not in ambiguous_all]
        if eligible and now >= self._next_allowed_at:
            # Never-attempted tracks win first; thereafter oldest attempt/success
            # wins.  The track ID is only a deterministic tie-breaker.
            chosen = min(
                eligible,
                key=lambda box: (
                    min(value for value in (states[int(box.track_id)].last_attempt_at, states[int(box.track_id)].last_success_at) if value is not None) if any(value is not None for value in (states[int(box.track_id)].last_attempt_at, states[int(box.track_id)].last_success_at)) else -1e12,
                    int(box.track_id),
                ),
            )
            chosen_id = int(chosen.track_id)
            state = states[chosen_id]
            state.last_attempt_at = now
            self.attempts += 1
            self._next_allowed_at = max(now, self._next_allowed_at) + self.interval_seconds
            crop, clipped = crop_frame(frame, crop_boxes[chosen_id])
            chosen = PersonBox(clipped.x1, clipped.y1, clipped.x2, clipped.y2, chosen.confidence, chosen.class_id, chosen.track_id)
            if crop.shape[1] < self.min_crop_width or crop.shape[0] < self.min_crop_height:
                results[chosen_id] = self._result(chosen, state, now, POSE_UNAVAILABLE, "SAMPLED", sampled=True, detail="CROP_TOO_SMALL")
            else:
                try:
                    detections = self.pose_backend.detect(crop)
                except (RuntimeError, TypeError, ValueError) as exc:
                    detections = []
                    detail = f"POSE_ERROR:{type(exc).__name__}"
                else:
                    detail = None
                detection = detections[0] if detections else None
                if detection is None:
                    results[chosen_id] = self._result(chosen, state, now, POSE_UNAVAILABLE, "SAMPLED", sampled=True, detail=detail or "NO_POSE")
                else:
                    visibility = self._visibility(detection.landmarks)
                    state.last_pose_quality = float(detection.pose_quality)
                    state.last_visibility = visibility
                    if not bool(detection.core_valid):
                        results[chosen_id] = self._result(chosen, state, now, POSE_CORE_INVALID, "SAMPLED", sampled=True, pose_quality=float(detection.pose_quality), visibility=visibility, detail="CORE_LANDMARKS_INVALID", detection=detection)
                    elif detection.world_landmarks is None:
                        results[chosen_id] = self._result(chosen, state, now, POSE_UNAVAILABLE, "SAMPLED", sampled=True, pose_quality=float(detection.pose_quality), visibility=visibility, core_valid=True, detail="WORLD_LANDMARKS_UNAVAILABLE", detection=detection)
                    else:
                        state.last_success_at = now
                        self.successes += 1
                        results[chosen_id] = self._result(chosen, state, now, POSE_OK, "SAMPLED", sampled=True, pose_quality=float(detection.pose_quality), visibility=visibility, core_valid=True, detection=detection)

        for box in eligible:
            track_id = int(box.track_id)
            if track_id in results:
                continue
            state = states[track_id]
            results[track_id] = self._result(box, state, now, POSE_PENDING, "GLOBAL_BUDGET_EXHAUSTED")
        return [results[int(box.track_id)] for box in selected_boxes if box.track_id is not None]

    def summary(self) -> dict[str, Any]:
        return {
            "status": "ENABLED",
            "max_pose_per_second_global": self.max_pose_per_second,
            "attempts": int(self.attempts),
            "successes": int(self.successes),
            "counts": dict(self.counts),
            "tracked_state_count": len(self._tracks),
            "feature_schema_version": "POSE_V2_RAW_MEDIAPIPE_CROP_WORLD_V1",
            "v4_called": False,
        }

    def close(self) -> None:
        close = getattr(self.pose_backend, "close", None)
        if callable(close):
            close()


__all__ = [
    "MediaPipeCropPoseBackend", "POSE_AMBIGUOUS", "POSE_CORE_INVALID", "POSE_OK",
    "POSE_PENDING", "POSE_UNAVAILABLE", "PoseV2Result", "PoseV2Stage",
]
