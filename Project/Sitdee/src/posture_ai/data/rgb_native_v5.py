"""Manifest, sampling, and leakage contracts for the RGB-native V5 track."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

V5_PROJECT_LABELS = ("TUP", "TLF", "TLB", "TLL", "TLR")
V5_OPTIONAL_LABEL = "SLOUCH"
V5_MANIFEST_COLUMNS = (
    "sample_id",
    "source_dataset",
    "source_url",
    "source_file",
    "participant_id",
    "session_id",
    "video_id",
    "frame_index",
    "timestamp",
    "source_label",
    "project_label",
    "mapping_status",
    "license",
    "camera_view",
    "pose_detected",
    "pose_quality",
    "notes",
)
VALID_MAPPING_STATUSES = ("EXACT", "COMPATIBLE", "AMBIGUOUS", "INCOMPATIBLE")


def sample_frame_indices(frame_count: int, fps: float, target_fps: float = 1.0) -> np.ndarray:
    """Return deterministic, non-adjacent frame indices for a video."""
    if frame_count < 0 or fps <= 0 or target_fps <= 0:
        raise ValueError("frame_count must be non-negative and fps/target_fps must be positive")
    if frame_count == 0:
        return np.empty(0, dtype=np.int64)
    stride = max(1, int(round(float(fps) / float(target_fps))))
    values = np.arange(0, frame_count, stride, dtype=np.int64)
    return values[values < frame_count]


def validate_manifest_rows(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        missing = [name for name in V5_MANIFEST_COLUMNS if name not in row]
        if missing:
            raise ValueError(f"V5 manifest row is missing columns: {missing}")
        status = str(row["mapping_status"])
        if status not in VALID_MAPPING_STATUSES:
            raise ValueError(f"Unknown V5 mapping status: {status}")
        project_label = str(row["project_label"] or "")
        if project_label and project_label not in (*V5_PROJECT_LABELS, V5_OPTIONAL_LABEL):
            raise ValueError(f"Unknown V5 project label: {project_label}")


def group_overlap(groups_a: Iterable[object], groups_b: Iterable[object]) -> set[str]:
    left = {str(value) for value in groups_a}
    right = {str(value) for value in groups_b}
    return left.intersection(right)


def dataset_class_confounds(source_datasets: Sequence[object], labels: Sequence[object]) -> dict:
    """Measure whether source identity nearly determines the project label."""
    if len(source_datasets) != len(labels):
        raise ValueError("source_datasets and labels must have equal length")
    by_source: dict[str, set[str]] = {}
    by_label: dict[str, set[str]] = {}
    for source, label in zip(source_datasets, labels):
        source_text = str(source)
        label_text = str(label)
        by_source.setdefault(source_text, set()).add(label_text)
        by_label.setdefault(label_text, set()).add(source_text)
    one_label_per_source = all(len(values) <= 1 for values in by_source.values())
    one_source_per_label = all(len(values) <= 1 for values in by_label.values())
    return {
        "by_source": {key: sorted(values) for key, values in sorted(by_source.items())},
        "by_label": {key: sorted(values) for key, values in sorted(by_label.items())},
        "one_label_per_source": one_label_per_source,
        "one_source_per_label": one_source_per_label,
        "severe": bool(one_label_per_source and one_source_per_label and len(by_label) > 1),
    }


__all__ = [
    "VALID_MAPPING_STATUSES",
    "V5_MANIFEST_COLUMNS",
    "V5_OPTIONAL_LABEL",
    "V5_PROJECT_LABELS",
    "dataset_class_confounds",
    "group_overlap",
    "sample_frame_indices",
    "validate_manifest_rows",
]
