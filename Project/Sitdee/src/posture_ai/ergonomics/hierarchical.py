"""Versioned, evidence-gated hierarchical posture decision contract.

This is a decision layer, not a claim that the project already owns SIT/STAND
or sitting-quality classifiers.  Until those models are trained on trusted
labels, missing evidence produces ``UNCERTAIN``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


HIERARCHICAL_SCHEMA_VERSION = "cctv_hierarchy_v1"
ACTIVITY_LABELS = frozenset({"SITTING", "STANDING"})
SITTING_QUALITY_LABELS = frozenset({"CORRECT", "INCORRECT"})
SUBTYPE_LABELS = frozenset({"FORWARD", "BACKWARD", "LEFT", "RIGHT", "MIXED"})


def _confidence(value: float | None) -> float:
    if value is None:
        return 0.0
    result = float(value)
    if not np.isfinite(result):
        return 0.0
    return max(0.0, min(1.0, result))


@dataclass(frozen=True)
class HierarchicalEvidence:
    pose_valid: bool
    activity_label: str | None = None
    activity_confidence: float | None = None
    sitting_quality_label: str | None = None
    sitting_quality_confidence: float | None = None
    subtype_label: str | None = None
    subtype_confidence: float | None = None
    v4_prediction: str | None = None
    v4_probabilities: Mapping[str, float] = field(default_factory=dict)
    geometry: Mapping[str, Any] = field(default_factory=dict)
    quality_reason: str | None = None

    def __post_init__(self) -> None:
        activity = None if self.activity_label is None else str(self.activity_label).upper()
        quality = None if self.sitting_quality_label is None else str(self.sitting_quality_label).upper()
        subtype = None if self.subtype_label is None else str(self.subtype_label).upper()
        if activity is not None and activity not in ACTIVITY_LABELS:
            raise ValueError(f"activity_label must be one of {sorted(ACTIVITY_LABELS)}")
        if quality is not None and quality not in SITTING_QUALITY_LABELS:
            raise ValueError(f"sitting_quality_label must be one of {sorted(SITTING_QUALITY_LABELS)}")
        if subtype is not None and subtype not in SUBTYPE_LABELS:
            raise ValueError(f"subtype_label must be one of {sorted(SUBTYPE_LABELS)}")
        object.__setattr__(self, "pose_valid", bool(self.pose_valid))
        object.__setattr__(self, "activity_label", activity)
        object.__setattr__(self, "sitting_quality_label", quality)
        object.__setattr__(self, "subtype_label", subtype)
        object.__setattr__(self, "v4_prediction", None if self.v4_prediction is None else str(self.v4_prediction))
        object.__setattr__(self, "v4_probabilities", {str(key): _confidence(value) for key, value in self.v4_probabilities.items()})
        object.__setattr__(self, "geometry", dict(self.geometry))


@dataclass(frozen=True)
class HierarchicalDecision:
    state: str
    activity: str
    sitting_quality: str | None
    subtype: str | None
    confidence: float
    reason: str
    raw_evidence: HierarchicalEvidence
    schema_version: str = HIERARCHICAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state,
            "activity": self.activity,
            "sitting_quality": self.sitting_quality,
            "subtype": self.subtype,
            "confidence": self.confidence,
            "reason": self.reason,
            "raw_activity_label": self.raw_evidence.activity_label,
            "raw_sitting_quality_label": self.raw_evidence.sitting_quality_label,
            "raw_subtype_label": self.raw_evidence.subtype_label,
            "v4_prediction": self.raw_evidence.v4_prediction,
            "geometry": dict(self.raw_evidence.geometry),
        }


def decide_hierarchical_posture(evidence: HierarchicalEvidence, *, min_confidence: float = 0.60) -> HierarchicalDecision:
    """Apply the hierarchy without inventing missing stage outputs."""

    threshold = _confidence(min_confidence)
    if not evidence.pose_valid:
        return HierarchicalDecision("UNCERTAIN", "UNCERTAIN", None, None, 0.0, evidence.quality_reason or "POSE_INVALID", evidence)
    activity_confidence = _confidence(evidence.activity_confidence)
    if evidence.activity_label is None or activity_confidence < threshold:
        return HierarchicalDecision("UNCERTAIN", "UNCERTAIN", None, None, activity_confidence, "ACTIVITY_EVIDENCE_INSUFFICIENT", evidence)
    if evidence.activity_label == "STANDING":
        return HierarchicalDecision("STANDING", "STANDING", None, None, activity_confidence, "STANDING_ACTIVITY_CONFIRMED", evidence)
    quality_confidence = _confidence(evidence.sitting_quality_confidence)
    if evidence.sitting_quality_label is None or quality_confidence < threshold:
        return HierarchicalDecision("UNCERTAIN", "SITTING", None, None, quality_confidence, "SITTING_QUALITY_EVIDENCE_INSUFFICIENT", evidence)
    if evidence.sitting_quality_label == "CORRECT":
        return HierarchicalDecision("SITTING_CORRECT", "SITTING", "CORRECT", None, quality_confidence, "SITTING_CORRECT_CONFIRMED", evidence)
    subtype_confidence = _confidence(evidence.subtype_confidence)
    if evidence.subtype_label is None or subtype_confidence < threshold:
        return HierarchicalDecision("SITTING_INCORRECT_UNCERTAIN", "SITTING", "INCORRECT", None, quality_confidence, "SUBTYPE_EVIDENCE_INSUFFICIENT", evidence)
    return HierarchicalDecision(
        f"SITTING_INCORRECT_{evidence.subtype_label}",
        "SITTING",
        "INCORRECT",
        evidence.subtype_label,
        min(quality_confidence, subtype_confidence),
        "SITTING_SUBTYPE_CONFIRMED",
        evidence,
    )


class HierarchicalTemporalEngine:
    """Per-person persistence layer that still exposes the raw decision."""

    def __init__(self, *, min_consecutive: int = 3, min_confidence: float = 0.60):
        if int(min_consecutive) < 1:
            raise ValueError("min_consecutive must be positive")
        self.min_consecutive = int(min_consecutive)
        self.min_confidence = float(min_confidence)
        self.current: HierarchicalDecision | None = None
        self.candidate: HierarchicalDecision | None = None
        self.candidate_count = 0
        self.last_timestamp: float | None = None

    def update(self, timestamp_seconds: float, evidence: HierarchicalEvidence) -> dict[str, Any]:
        timestamp = float(timestamp_seconds)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        self.last_timestamp = timestamp
        raw = decide_hierarchical_posture(evidence, min_confidence=self.min_confidence)
        if self.current is None:
            self.current = raw
            self.candidate = None
            self.candidate_count = 0
        elif raw.state == self.current.state:
            self.candidate = None
            self.candidate_count = 0
        elif self.candidate is not None and raw.state == self.candidate.state:
            self.candidate_count += 1
            if self.candidate_count >= self.min_consecutive:
                self.current = raw
                self.candidate = None
                self.candidate_count = 0
        else:
            self.candidate = raw
            self.candidate_count = 1
        return {
            "timestamp_seconds": timestamp,
            "raw_decision": raw.to_dict(),
            "committed_decision": self.current.to_dict(),
            "candidate_state": None if self.candidate is None else self.candidate.state,
            "candidate_count": self.candidate_count,
            "raw_outputs_exposed": True,
        }


__all__ = [
    "ACTIVITY_LABELS",
    "HIERARCHICAL_SCHEMA_VERSION",
    "HierarchicalDecision",
    "HierarchicalEvidence",
    "HierarchicalTemporalEngine",
    "SITTING_QUALITY_LABELS",
    "SUBTYPE_LABELS",
    "decide_hierarchical_posture",
]
