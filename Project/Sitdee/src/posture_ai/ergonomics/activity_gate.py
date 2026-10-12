"""Conservative experimental sitting/standing gate.

This module is deliberately not a trained activity classifier.  It only emits
SITTING/STANDING when both lower-body sides provide visible, geometrically
consistent evidence.  Otherwise it emits UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.posture_ai.pose.landmarks import PoseLandmarks


ACTIVITY_STATES = frozenset({"SITTING", "STANDING", "UNKNOWN"})
ACTIVITY_METHOD = "GEOMETRY_EXPERIMENTAL"


@dataclass(frozen=True)
class ActivityGateConfig:
    """Explicit, deployment-tunable heuristics; not clinical thresholds."""

    visibility_threshold: float = 0.5
    standing_knee_angle_min_deg: float = 155.0
    sitting_knee_angle_max_deg: float = 140.0
    vertical_order_margin: float = 0.02
    transition_confirmations: int = 3

    def __post_init__(self) -> None:
        for name in ("visibility_threshold", "vertical_order_margin"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0.0 < float(self.sitting_knee_angle_max_deg) < float(self.standing_knee_angle_min_deg) <= 180.0:
            raise ValueError("activity angle thresholds must satisfy 0 < sitting < standing <= 180")
        if int(self.transition_confirmations) < 1:
            raise ValueError("transition_confirmations must be positive")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "ActivityGateConfig":
        values = dict(mapping or {})
        return cls(
            visibility_threshold=float(values.get("visibility_threshold", cls.visibility_threshold)),
            standing_knee_angle_min_deg=float(values.get("standing_knee_angle_min_deg", cls.standing_knee_angle_min_deg)),
            sitting_knee_angle_max_deg=float(values.get("sitting_knee_angle_max_deg", cls.sitting_knee_angle_max_deg)),
            vertical_order_margin=float(values.get("vertical_order_margin", cls.vertical_order_margin)),
            transition_confirmations=int(values.get("transition_confirmations", cls.transition_confirmations)),
        )


@dataclass(frozen=True)
class ActivityEvidence:
    raw_state: str
    confidence: float
    reason: str
    method: str = ACTIVITY_METHOD
    metrics: dict[str, Any] = field(default_factory=dict)
    missing_landmarks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.raw_state not in ACTIVITY_STATES:
            raise ValueError("raw_state must be SITTING, STANDING, or UNKNOWN")
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "missing_landmarks", tuple(str(item) for item in self.missing_landmarks))

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_state": self.raw_state,
            "confidence": self.confidence,
            "reason": self.reason,
            "method": self.method,
            "metrics": dict(self.metrics),
            "missing_landmarks": list(self.missing_landmarks),
        }


def _point(pose: PoseLandmarks, name: str, threshold: float) -> np.ndarray | None:
    point = pose.by_name(name)
    if point is None or not point.present:
        return None
    if point.visibility is not None and float(point.visibility) < threshold:
        return None
    values = np.asarray([point.x, point.y], dtype=float)
    return values if np.isfinite(values).all() else None


def _knee_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> float | None:
    first = hip - knee
    second = ankle - knee
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-9:
        return None
    cosine = float(np.dot(first, second) / denominator)
    cosine = max(-1.0, min(1.0, cosine))
    return float(np.degrees(np.arccos(cosine)))


def classify_activity_geometry(pose: PoseLandmarks, config: ActivityGateConfig | None = None) -> ActivityEvidence:
    """Classify only when both lower-body sides support the same hypothesis."""

    settings = config or ActivityGateConfig()
    required = (
        "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    )
    points = {name: _point(pose, name, settings.visibility_threshold) for name in required}
    missing = tuple(name for name, value in points.items() if value is None)
    if missing:
        return ActivityEvidence("UNKNOWN", 0.0, "LOW_VISIBILITY_OR_MISSING_LOWER_BODY", missing_landmarks=missing)

    angles = []
    vertical_orders = []
    for side in ("left", "right"):
        angle = _knee_angle(points[f"{side}_hip"], points[f"{side}_knee"], points[f"{side}_ankle"])
        if angle is None:
            return ActivityEvidence("UNKNOWN", 0.0, "DEGENERATE_KNEE_GEOMETRY")
        angles.append(angle)
        hip_y = float(points[f"{side}_hip"][1])
        knee_y = float(points[f"{side}_knee"][1])
        ankle_y = float(points[f"{side}_ankle"][1])
        vertical_orders.append(knee_y > hip_y + settings.vertical_order_margin and ankle_y > knee_y + settings.vertical_order_margin)

    median_angle = float(np.median(angles))
    angle_agreement = abs(float(angles[0]) - float(angles[1]))
    if median_angle >= settings.standing_knee_angle_min_deg and all(vertical_orders) and angle_agreement <= 20.0:
        confidence = min(1.0, 0.60 + (median_angle - settings.standing_knee_angle_min_deg) / 50.0)
        return ActivityEvidence(
            "STANDING", confidence, "EXTENDED_KNEES_AND_VERTICAL_LEG_ORDER",
            metrics={"knee_angles_deg": angles, "median_knee_angle_deg": median_angle, "vertical_orders": vertical_orders},
        )
    if median_angle <= settings.sitting_knee_angle_max_deg and angle_agreement <= 25.0:
        confidence = min(1.0, 0.60 + (settings.sitting_knee_angle_max_deg - median_angle) / 100.0)
        return ActivityEvidence(
            "SITTING", confidence, "FLEXED_KNEES_WITH_BILATERAL_EVIDENCE",
            metrics={"knee_angles_deg": angles, "median_knee_angle_deg": median_angle, "vertical_orders": vertical_orders},
        )
    return ActivityEvidence(
        "UNKNOWN", 0.0, "AMBIGUOUS_LOWER_BODY_GEOMETRY",
        metrics={"knee_angles_deg": angles, "median_knee_angle_deg": median_angle, "vertical_orders": vertical_orders},
    )


class ActivityGate:
    """Per-person hysteresis for the experimental activity evidence."""

    def __init__(self, config: ActivityGateConfig | None = None) -> None:
        self.config = config or ActivityGateConfig()
        self.state = "UNKNOWN"
        self.raw_state = "UNKNOWN"
        self.confidence = 0.0
        self.last_evidence = ActivityEvidence("UNKNOWN", 0.0, "NOT_EVALUATED")
        self.candidate_state: str | None = None
        self.candidate_count = 0
        self.last_timestamp: float | None = None

    def reset(self) -> None:
        self.__init__(self.config)

    def update(self, timestamp_seconds: float, pose: PoseLandmarks) -> dict[str, Any]:
        timestamp = float(timestamp_seconds)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        self.last_timestamp = timestamp
        evidence = classify_activity_geometry(pose, self.config)
        previous = self.state
        self.raw_state = evidence.raw_state
        self.confidence = evidence.confidence
        self.last_evidence = evidence
        if evidence.raw_state == "UNKNOWN":
            self.state = "UNKNOWN"
            self.candidate_state = None
            self.candidate_count = 0
        elif self.state == "UNKNOWN":
            if self.candidate_state == evidence.raw_state:
                self.candidate_count += 1
            else:
                self.candidate_state = evidence.raw_state
                self.candidate_count = 1
            if self.candidate_count >= self.config.transition_confirmations:
                self.state = evidence.raw_state
                self.candidate_state = None
                self.candidate_count = 0
        elif evidence.raw_state == self.state:
            self.candidate_state = None
            self.candidate_count = 0
        else:
            if self.candidate_state == evidence.raw_state:
                self.candidate_count += 1
            else:
                self.candidate_state = evidence.raw_state
                self.candidate_count = 1
            if self.candidate_count >= self.config.transition_confirmations:
                self.state = evidence.raw_state
                self.candidate_state = None
                self.candidate_count = 0
            else:
                # A transition is uncertain; pause posture monitoring safely.
                self.state = "UNKNOWN"
        return {
            "state": self.state,
            "raw_state": evidence.raw_state,
            "previous_state": previous,
            "changed": self.state != previous,
            "confidence": evidence.confidence,
            "method": ACTIVITY_METHOD,
            "reason": evidence.reason,
            "metrics": dict(evidence.metrics),
            "missing_landmarks": list(evidence.missing_landmarks),
            "candidate_state": self.candidate_state,
            "candidate_count": self.candidate_count,
            "timestamp_seconds": timestamp,
        }


__all__ = [
    "ACTIVITY_METHOD", "ACTIVITY_STATES", "ActivityEvidence", "ActivityGate", "ActivityGateConfig",
    "classify_activity_geometry",
]
