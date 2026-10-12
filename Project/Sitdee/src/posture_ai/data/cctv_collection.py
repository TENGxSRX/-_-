"""Manual-label contracts for controlled multi-person CCTV collection.

This module contains only validation and keyboard-to-label rules.  It does
not identify people, infer ground truth, or turn a model prediction into a
label.  ``PERSON_XX`` values are anonymous, short-lived runtime track IDs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


CCTV_COLLECTION_SCHEMA_VERSION = "controlled_cctv_ground_truth_v2"
CCTV_MANUAL_LABELS = ("TUP", "TLB", "TLF", "TLL", "TLR")
CCTV_NON_SAMPLE_ACTIONS = ("IGNORE", "UNCERTAIN")
CCTV_EXTRA_SAMPLE_COLUMNS = (
    "expected_people_count",
    "target_subject_id",
    "target_seat_id",
    "target_position_id",
    "distance_band",
    "pose_index",
    "track_id",
    "nose_visibility",
    "left_shoulder_visibility",
    "right_shoulder_visibility",
    "left_hip_visibility",
    "right_hip_visibility",
    "human_posture_label",
    "label_source",
)
CCTV_LABEL_KEYS = {
    "1": "TUP",
    "2": "TLB",
    "3": "TLF",
    "4": "TLL",
    "5": "TLR",
    "0": "IGNORE",
    "U": "UNCERTAIN",
}
_PERSON_ID = re.compile(r"^PERSON_[0-9]{2,4}$")


def validate_person_id(value: str) -> str:
    """Require an anonymous runtime ID and reject names/biometric IDs."""

    result = str(value).strip().upper()
    if not _PERSON_ID.fullmatch(result):
        raise ValueError("person_id must be an anonymous PERSON_XX runtime ID")
    return result


def validate_manual_label(value: str, *, allow_actions: bool = True) -> str:
    """Validate a human-selected posture label or explicit non-sample action."""

    result = str(value).strip().upper()
    allowed = CCTV_MANUAL_LABELS + (CCTV_NON_SAMPLE_ACTIONS if allow_actions else ())
    if result not in allowed:
        raise ValueError(f"manual label must be one of {allowed}")
    return result


def label_for_key(key: str) -> str | None:
    """Map one UI key to a manual action; unknown keys are ignored."""

    return CCTV_LABEL_KEYS.get(str(key).upper())


@dataclass(frozen=True)
class ManualPersonAssignment:
    """One explicit operator assignment for one anonymous track."""

    person_id: str
    label: str
    operator: str = "ANONYMOUS_OPERATOR"

    def __post_init__(self) -> None:
        object.__setattr__(self, "person_id", validate_person_id(self.person_id))
        object.__setattr__(self, "label", validate_manual_label(self.label))
        operator = str(self.operator).strip()
        if not operator or operator.lower() in {"name", "face", "identity"}:
            raise ValueError("operator must be a non-identifying label or code")
        object.__setattr__(self, "operator", operator[:80])

    @property
    def recordable(self) -> bool:
        return self.label in CCTV_MANUAL_LABELS

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "person_id": self.person_id,
            "human_label": self.label,
            "operator": self.operator,
            "recordable": self.recordable,
            "label_source": "MANUAL_OPERATOR_ONLY",
            "model_prediction_is_not_ground_truth": True,
        }


__all__ = [
    "CCTV_COLLECTION_SCHEMA_VERSION",
    "CCTV_LABEL_KEYS",
    "CCTV_MANUAL_LABELS",
    "CCTV_NON_SAMPLE_ACTIONS",
    "CCTV_EXTRA_SAMPLE_COLUMNS",
    "ManualPersonAssignment",
    "label_for_key",
    "validate_manual_label",
    "validate_person_id",
]
