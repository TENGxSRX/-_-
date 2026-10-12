"""Contracts for target-locked, human-labelled CCTV collection V2.

Rows in this schema are training candidates only when a human selected the
label *and* the target lock reports an unambiguous locked target.  Runtime
telemetry is intentionally written elsewhere and never becomes a label.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from src.posture_ai.features.canonical_cctv import build_canonical_cctv_features
from src.posture_ai.multiperson.target_lock import TargetLockSnapshot
from src.posture_ai.pose.landmarks import LANDMARK_NAMES, PoseLandmarks


TARGET_LOCK_COLLECTION_SCHEMA_VERSION = "controlled_cctv_target_lock_v2"
TARGET_LOCK_EXTRA_SAMPLE_COLUMNS = (
    "source_label",
    "canonical_label",
    "label_source",
    "target_lock_state",
    "target_identity_mode",
    "target_yolo_track_id",
    "target_runtime_subject_id",
    "target_bbox_json",
    "target_association_state",
    "target_association_score",
    "target_pause_reason",
    "yolo_enabled",
    "yolo_model",
    "yolo_tracker",
    "yolo_detection_count",
    "yolo_tracked_count",
    "association_strategy",
    "collection_mode",
    "inference_frame_width",
    "inference_frame_height",
    "inference_scale",
    "world_landmarks_json",
    "cctv_canonical_status",
    "cctv_canonical_features_json",
    "cctv_canonical_metadata_json",
)


def serialise_landmarks(landmarks: PoseLandmarks | None) -> list[dict[str, float | str | None]]:
    """Return a fixed-order, non-image representation of MediaPipe landmarks."""

    if landmarks is None:
        return []
    values = PoseLandmarks.from_any(landmarks)
    rows: list[dict[str, float | str | None]] = []
    for name, point in zip(LANDMARK_NAMES, values.points):
        rows.append({
            "name": name,
            "x": None if point is None or point.x is None else float(point.x),
            "y": None if point is None or point.y is None else float(point.y),
            "z": None if point is None or point.z is None else float(point.z),
            "visibility": None if point is None or point.visibility is None else float(point.visibility),
        })
    return rows


def canonical_cctv_diagnostic(landmarks: PoseLandmarks | None) -> dict[str, Any]:
    """Compute cctv_canonical_v1 only when it is mechanically valid.

    This is a stored diagnostic representation.  It is not passed to V4 and
    must not be treated as a validated new model input.
    """

    if landmarks is None:
        return {"status": "UNAVAILABLE_NO_WORLD_LANDMARKS", "features": [], "metadata": {}}
    try:
        result = build_canonical_cctv_features(
            landmarks,
            coordinate_type="mediapipe_world",
            scale_mode="torso_length",
        )
    except (TypeError, ValueError) as exc:
        return {"status": f"UNAVAILABLE_{type(exc).__name__.upper()}", "features": [], "metadata": {"detail": str(exc)}}
    values = np.asarray(result.values, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        return {"status": "UNAVAILABLE_NONFINITE", "features": [], "metadata": dict(result.metadata)}
    return {
        "status": "AVAILABLE_DIAGNOSTIC_ONLY",
        "features": [float(value) for value in values],
        "metadata": dict(result.metadata),
    }


def target_lock_collection_gate(
    snapshot: TargetLockSnapshot,
    *,
    human_label: str | None,
    label_source: str,
    quality_accepted: bool,
) -> dict[str, Any]:
    """Explain whether one row may become a human-labelled training sample."""

    reasons: list[str] = []
    if str(label_source).upper() != "HUMAN":
        reasons.append("LABEL_SOURCE_NOT_HUMAN")
    if not human_label:
        reasons.append("NO_HUMAN_LABEL")
    if snapshot.state != "TARGET_LOCKED":
        reasons.append("TARGET_NOT_LOCKED")
    if snapshot.auto_paused or not snapshot.can_collect:
        reasons.append("TARGET_AUTO_PAUSED")
    if not quality_accepted:
        reasons.append("POSE_OR_FEATURE_QUALITY_REJECTED")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "label_is_human": str(label_source).upper() == "HUMAN",
        "training_data_is_separate_from_runtime_telemetry": True,
    }


__all__ = [
    "TARGET_LOCK_COLLECTION_SCHEMA_VERSION",
    "TARGET_LOCK_EXTRA_SAMPLE_COLUMNS",
    "canonical_cctv_diagnostic",
    "serialise_landmarks",
    "target_lock_collection_gate",
]
