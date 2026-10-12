"""Evidence-based routing of CCTV failures to the right subsystem."""

from __future__ import annotations

from typing import Any, Mapping


PERSON_DETECTION_FAILURE = "PERSON_DETECTION_FAILURE"
POSE_FAILURE = "POSE_FAILURE"
CORE_FAILURE = "CORE_FAILURE"
TRACKING_FAILURE = "TRACKING_FAILURE"
ACTIVITY_CLASSIFIER_FAILURE = "ACTIVITY_CLASSIFIER_FAILURE"
POSTURE_CLASSIFIER_FAILURE = "POSTURE_CLASSIFIER_FAILURE"
SUBTYPE_FAILURE = "SUBTYPE_FAILURE"
TEMPORAL_INSTABILITY = "TEMPORAL_INSTABILITY"


def classify_frame_failure(
    *,
    expected_count: int | None,
    detected_count: int,
    core_valid: bool,
    tracker_state: str | None = None,
    hierarchy: Mapping[str, Any] | None = None,
) -> str | None:
    """Return the first actionable failure, or ``None`` for a usable frame."""

    detected = int(detected_count)
    if detected <= 0:
        return PERSON_DETECTION_FAILURE
    if expected_count is not None and detected < int(expected_count):
        return PERSON_DETECTION_FAILURE
    if not core_valid:
        return CORE_FAILURE
    if tracker_state and str(tracker_state).upper() in {"LOST", "UNSTABLE", "UNMATCHED"}:
        return TRACKING_FAILURE
    evidence = dict(hierarchy or {})
    state = str(evidence.get("state", "")).upper()
    reason = str(evidence.get("reason", "")).upper()
    if state in {"UNCERTAIN", ""}:
        if "ACTIVITY" in reason:
            return ACTIVITY_CLASSIFIER_FAILURE
        if "QUALITY" in reason:
            return POSTURE_CLASSIFIER_FAILURE
        if "SUBTYPE" in reason:
            return SUBTYPE_FAILURE
    if bool(evidence.get("temporal_unstable")):
        return TEMPORAL_INSTABILITY
    return None


__all__ = [
    "PERSON_DETECTION_FAILURE", "POSE_FAILURE", "CORE_FAILURE", "TRACKING_FAILURE",
    "ACTIVITY_CLASSIFIER_FAILURE", "POSTURE_CLASSIFIER_FAILURE", "SUBTYPE_FAILURE",
    "TEMPORAL_INSTABILITY", "classify_frame_failure",
]
