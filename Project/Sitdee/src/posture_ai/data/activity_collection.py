"""Schema constants for future human-labelled sit/stand collection."""

from __future__ import annotations


ACTIVITY_COLLECTION_SCHEMA_VERSION = "controlled_activity_v1"
ACTIVITY_LABELS = ("SITTING", "STANDING")
ACTIVITY_EXTRA_SAMPLE_COLUMNS = (
    "human_activity_label",
    "label_source",
    "target_subject_id",
    "track_id",
    "pose_index",
    "expected_people_count",
    "activity_gate_state",
    "activity_gate_method",
    "canonical_schema_version",
    "canonical_coordinate_type",
    "canonical_scale_mode",
)


__all__ = ["ACTIVITY_COLLECTION_SCHEMA_VERSION", "ACTIVITY_LABELS", "ACTIVITY_EXTRA_SAMPLE_COLUMNS"]
