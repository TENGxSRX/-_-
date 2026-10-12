"""Manual-label contract for the joint activity/posture pilot."""

from __future__ import annotations


BEHAVIOR_COLLECTION_SCHEMA_VERSION = "behavior_partial_pose_collection_v1"
ACTIVITY_LABELS = ("SITTING", "STANDING")
SITTING_POSTURE_LABELS = ("TUP", "TLB")
POSTURE_NONE = "NONE"
BEHAVIOR_EXTRA_SAMPLE_COLUMNS = (
    "human_activity_label", "human_posture_label", "label_source", "target_subject_id", "track_id", "pose_index",
    "expected_people_count", "distance_band", "camera_position_id", "lower_body_visibility_json", "partial_quality_json",
    "activity_feature_schema_version", "activity_feature_variant", "v4_diagnostic_status",
)


def validate_behavior_labels(activity_label: str, posture_label: str) -> tuple[str, str]:
    """Accept only explicit human activity/posture combinations."""

    activity = str(activity_label).upper()
    posture = str(posture_label).upper()
    if activity not in ACTIVITY_LABELS:
        raise ValueError(f"activity_label must be one of {ACTIVITY_LABELS}")
    if activity == "STANDING" and posture != POSTURE_NONE:
        raise ValueError("STANDING requires posture_label=NONE")
    if activity == "SITTING" and posture not in SITTING_POSTURE_LABELS:
        raise ValueError(f"SITTING requires posture_label in {SITTING_POSTURE_LABELS}")
    return activity, posture


__all__ = [
    "ACTIVITY_LABELS", "BEHAVIOR_COLLECTION_SCHEMA_VERSION", "BEHAVIOR_EXTRA_SAMPLE_COLUMNS", "POSTURE_NONE",
    "SITTING_POSTURE_LABELS", "validate_behavior_labels",
]
