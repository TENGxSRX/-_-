"""Safe data contracts for a future anonymous multi-person layer."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.posture_ai.pose.landmarks import PoseLandmarks


_SUBJECT_ID = re.compile(r"PERSON_[0-9]{2,4}\Z")
_TRACKING_STATES = frozenset({"NEW", "ACTIVE", "TEMPORARILY_LOST", "EXPIRED"})


def _pair(value: tuple[float, float] | list[float] | None, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _box(value: tuple[float, float, float, float] | list[float] | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError("bbox must contain exactly four values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("bbox must contain finite values")
    if result[2] < result[0] or result[3] < result[1]:
        raise ValueError("bbox must be ordered as left, top, right, bottom")
    return result


def anonymous_subject_id(index: int) -> str:
    """Create a non-biometric, frame/session-local subject identifier."""

    value = int(index)
    if value < 0:
        raise ValueError("subject index must be non-negative")
    return f"PERSON_{value + 1:02d}"


@dataclass(frozen=True)
class PoseSubject:
    """One anonymous pose candidate and its independently maintained state."""

    subject_id: str
    landmarks: PoseLandmarks
    pose_quality: float
    v4_result: Mapping[str, Any] | None = None
    geometry: Mapping[str, Any] = field(default_factory=dict)
    temporal_state: Mapping[str, Any] = field(default_factory=dict)
    tracking_state: str = "ACTIVE"
    world_landmarks: PoseLandmarks | None = None
    bbox: tuple[float, float, float, float] | None = None
    center: tuple[float, float] | None = None
    core_valid: bool = False
    v4_prediction: str | None = None
    v4_probabilities: Mapping[str, float] | None = None
    last_seen_at: float | None = None
    core_visibility: Mapping[str, float | None] = field(default_factory=dict)
    hierarchical_state: Mapping[str, Any] = field(default_factory=dict)
    activity_state: str = "UNKNOWN"
    activity_raw_state: str = "UNKNOWN"
    activity_confidence: float = 0.0
    activity_method: str = "NOT_EVALUATED"
    activity_evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _SUBJECT_ID.fullmatch(str(self.subject_id)):
            raise ValueError("subject_id must be an anonymous PERSON_XX identifier")
        if not isinstance(self.landmarks, PoseLandmarks):
            raise TypeError("landmarks must be a PoseLandmarks instance")
        quality = float(self.pose_quality)
        if not np.isfinite(quality) or quality < 0.0 or quality > 1.0:
            raise ValueError("pose_quality must be finite and between 0 and 1")
        state = str(self.tracking_state)
        if state not in _TRACKING_STATES:
            raise ValueError(f"tracking_state must be one of {sorted(_TRACKING_STATES)}")
        last_seen = None if self.last_seen_at is None else float(self.last_seen_at)
        if last_seen is not None and (not np.isfinite(last_seen) or last_seen < 0.0):
            raise ValueError("last_seen_at must be finite and non-negative")
        probabilities = None
        if self.v4_probabilities is not None:
            probabilities = {str(key): float(value) for key, value in self.v4_probabilities.items()}
            if not all(np.isfinite(value) and value >= 0.0 for value in probabilities.values()):
                raise ValueError("v4_probabilities must contain finite, non-negative values")
        object.__setattr__(self, "subject_id", str(self.subject_id))
        object.__setattr__(self, "pose_quality", quality)
        object.__setattr__(self, "v4_result", None if self.v4_result is None else dict(self.v4_result))
        object.__setattr__(self, "geometry", dict(self.geometry))
        object.__setattr__(self, "temporal_state", dict(self.temporal_state))
        object.__setattr__(self, "tracking_state", state)
        object.__setattr__(self, "world_landmarks", None if self.world_landmarks is None else PoseLandmarks.from_any(self.world_landmarks))
        object.__setattr__(self, "bbox", _box(self.bbox))
        object.__setattr__(self, "center", _pair(self.center, "center"))
        object.__setattr__(self, "core_valid", bool(self.core_valid))
        object.__setattr__(self, "v4_prediction", None if self.v4_prediction is None else str(self.v4_prediction))
        object.__setattr__(self, "v4_probabilities", probabilities)
        object.__setattr__(self, "last_seen_at", last_seen)
        visibility = {}
        for key, value in dict(self.core_visibility).items():
            visibility[str(key)] = None if value is None else float(value)
        object.__setattr__(self, "core_visibility", visibility)
        object.__setattr__(self, "hierarchical_state", dict(self.hierarchical_state))
        activity_state = str(self.activity_state).upper()
        activity_raw_state = str(self.activity_raw_state).upper()
        if activity_state not in {"SITTING", "STANDING", "UNKNOWN"} or activity_raw_state not in {"SITTING", "STANDING", "UNKNOWN"}:
            raise ValueError("activity states must be SITTING, STANDING, or UNKNOWN")
        confidence = float(self.activity_confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("activity_confidence must be between 0 and 1")
        object.__setattr__(self, "activity_state", activity_state)
        object.__setattr__(self, "activity_raw_state", activity_raw_state)
        object.__setattr__(self, "activity_confidence", confidence)
        object.__setattr__(self, "activity_method", str(self.activity_method))
        object.__setattr__(self, "activity_evidence", dict(self.activity_evidence))

    def to_dict(self) -> dict[str, Any]:
        """Return a telemetry-safe summary without face or identity data."""

        return {
            "subject_id": self.subject_id,
            "tracking_state": self.tracking_state,
            "core_valid": self.core_valid,
            "pose_quality": self.pose_quality,
            "landmark_count": len(self.landmarks),
            "world_landmark_count": 0 if self.world_landmarks is None else len(self.world_landmarks),
            "bbox": self.bbox,
            "center": self.center,
            "last_seen_at": self.last_seen_at,
            "v4_prediction": self.v4_prediction,
            "v4_probabilities": self.v4_probabilities,
            "v4_result": self.v4_result,
            "core_visibility": dict(self.core_visibility),
            "hierarchical_state": dict(self.hierarchical_state),
            "activity_state": self.activity_state,
            "activity_raw_state": self.activity_raw_state,
            "activity_confidence": self.activity_confidence,
            "activity_method": self.activity_method,
            "activity_evidence": dict(self.activity_evidence),
            "geometry": self.geometry,
            "temporal_state": self.temporal_state,
        }


@dataclass(frozen=True)
class MultiPersonFrameResult:
    """A frame containing zero or more anonymous pose tracks."""

    timestamp_seconds: float
    subjects: tuple[PoseSubject, ...] = ()
    camera_id: str | None = None
    tracking_status: str = "NOT_IMPLEMENTED"
    events: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_seconds)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        subjects = tuple(self.subjects)
        if any(not isinstance(item, PoseSubject) for item in subjects):
            raise TypeError("subjects must contain PoseSubject values")
        ids = [item.subject_id for item in subjects]
        if len(ids) != len(set(ids)):
            raise ValueError("subject_id values must be unique within a frame")
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(self, "camera_id", None if self.camera_id is None else str(self.camera_id))
        object.__setattr__(self, "tracking_status", str(self.tracking_status))
        object.__setattr__(self, "events", tuple(dict(event) for event in self.events))

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_seconds": self.timestamp_seconds,
            "camera_id": self.camera_id,
            "tracking_status": self.tracking_status,
            "subject_count": len(self.subjects),
            "subjects": [subject.to_dict() for subject in self.subjects],
            "events": [dict(event) for event in self.events],
        }
