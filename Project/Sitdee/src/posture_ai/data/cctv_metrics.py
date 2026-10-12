"""Ground-truth-first metrics for controlled CCTV evaluation.

All functions accept plain mappings so they can be used with JSONL/CSV
records and unit-tested without a camera.  Missing observations are reported
as unavailable rather than being converted into classifier errors.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _rate(values: Sequence[bool]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=float)))


def detection_frame_metrics(expected_people_count: int, detected_people_count: int, core_valid_people_count: int) -> dict[str, Any]:
    """Calculate one frame's count outcomes without judging posture."""

    expected = int(expected_people_count)
    detected = int(detected_people_count)
    core_valid = int(core_valid_people_count)
    if expected < 1 or expected > 4:
        raise ValueError("expected_people_count must be between 1 and 4")
    if detected < 0 or core_valid < 0:
        raise ValueError("detected and core-valid counts must not be negative")
    return {
        "expected_people_count": expected,
        "detected_people_count": detected,
        "core_valid_people_count": core_valid,
        "exact_count": detected == expected,
        "undercount": detected < expected,
        "overcount": detected > expected,
        "all_people_core_valid": detected == expected and core_valid == expected,
    }


def aggregate_detection_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate exact/under/over-count and complete-core rates."""

    frames = [
        detection_frame_metrics(
            int(record["expected_people_count"]),
            int(record["detected_people_count"]),
            int(record.get("core_valid_people_count", 0)),
        )
        for record in records
    ]
    if not frames:
        return {"frame_count": 0, "status": "NO_GROUND_TRUTH_FRAMES"}
    return {
        "frame_count": len(frames),
        "status": "MEASURED",
        "exact_count_rate": float(np.mean([item["exact_count"] for item in frames])),
        "undercount_rate": float(np.mean([item["undercount"] for item in frames])),
        "overcount_rate": float(np.mean([item["overcount"] for item in frames])),
        "all_people_core_valid_rate": float(np.mean([item["all_people_core_valid"] for item in frames])),
        "mean_expected_people": float(np.mean([item["expected_people_count"] for item in frames])),
        "mean_detected_people": float(np.mean([item["detected_people_count"] for item in frames])),
        "max_detected_people": int(max(item["detected_people_count"] for item in frames)),
    }


def detection_metrics_by_scenario(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        expected = int(record["expected_people_count"])
        grouped[str(expected)].append(record)
    return {key: aggregate_detection_metrics(value) for key, value in sorted(grouped.items(), key=lambda item: int(item[0]))}


CORE_VISIBILITY_FIELDS = (
    "nose_visibility",
    "left_shoulder_visibility",
    "right_shoulder_visibility",
    "left_hip_visibility",
    "right_hip_visibility",
)


def per_person_core_metrics(records: Iterable[Mapping[str, Any]], *, group_fields: Sequence[str] = ()) -> dict[str, Any]:
    """Report target pose/core rates by participant and supplied view groups."""

    rows = list(records)
    if not rows:
        return {"status": "NO_GROUND_TRUTH_PERSON_ROWS", "groups": {}}
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    fields = tuple(group_fields) or ("subject_id",)
    for row in rows:
        groups[tuple(str(row.get(field, "UNKNOWN")) for field in fields)].append(row)
    output: dict[str, Any] = {}
    for key, values in groups.items():
        item: dict[str, Any] = {
            "row_count": len(values),
            "pose_detected_rate": _rate([bool(int(_number(row.get("pose_detected", 0)))) for row in values]),
            "core_valid_rate": _rate([bool(int(_number(row.get("core_valid", 0)))) for row in values]),
        }
        for field in CORE_VISIBILITY_FIELDS:
            observed = [_number(row.get(field), default=np.nan) >= 0.5 for row in values if np.isfinite(_number(row.get(field), default=np.nan))]
            item[field + "_valid_rate"] = _rate(observed)
        output["/".join(key)] = item
    return {"status": "MEASURED", "group_fields": list(fields), "groups": output}


def frozen_v4_metrics(y_true: Sequence[str], y_pred: Sequence[str], *, labels: Sequence[str] = ("TUP", "TLB")) -> dict[str, Any]:
    """Evaluate V4 only on rows with human labels and valid features."""

    truth = [str(value) for value in y_true]
    pred = [str(value) for value in y_pred]
    if len(truth) != len(pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not truth:
        return {"status": "NO_VALID_LABELLED_V4_ROWS"}
    label_list = [str(label) for label in labels]
    return {
        "status": "MEASURED",
        "sample_count": len(truth),
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=label_list, average="macro", zero_division=0)),
        "precision_per_class": {label: float(value) for label, value in zip(label_list, precision_score(truth, pred, labels=label_list, average=None, zero_division=0))},
        "recall_per_class": {label: float(value) for label, value in zip(label_list, recall_score(truth, pred, labels=label_list, average=None, zero_division=0))},
        "f1_per_class": {label: float(value) for label, value in zip(label_list, f1_score(truth, pred, labels=label_list, average=None, zero_division=0))},
        "confusion_matrix_labels": label_list,
        "confusion_matrix": confusion_matrix(truth, pred, labels=label_list).tolist(),
    }


def temporal_stability_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure raw V4 flips within continuous human-labelled segments."""

    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        prediction = row.get("v4_prediction")
        truth = row.get("human_posture_label", row.get("human_label"))
        if prediction and truth:
            key = (
                str(row.get("session_id", "")),
                str(row.get("clip_id", "")),
                str(row.get("target_subject_id", row.get("track_id", row.get("subject_id", "")))),
                str(truth),
            )
            grouped[key].append(row)
    if not grouped:
        return {"status": "NO_LABELLED_TEMPORAL_SEQUENCE"}
    sample_count = 0
    transitions = 0
    switches = 0
    false_transitions = 0
    correct_runs: list[int] = []
    confidence: list[float] = []
    confidence_diffs: list[float] = []
    for segment in grouped.values():
        usable = sorted(segment, key=lambda row: _number(row.get("timestamp_seconds")))
        predictions = [str(row["v4_prediction"]) for row in usable]
        truth = [str(row.get("human_posture_label", row.get("human_label"))) for row in usable]
        segment_switches = sum(first != second for first, second in zip(predictions, predictions[1:]))
        transitions += max(0, len(usable) - 1)
        switches += segment_switches
        false_transitions += sum(
            first != second and first == truth[index] and second != truth[index + 1]
            for index, (first, second) in enumerate(zip(predictions, predictions[1:]))
        )
        correct = [prediction == label for prediction, label in zip(predictions, truth)]
        current = 0
        for value in correct:
            current = current + 1 if value else 0
            correct_runs.append(current)
        segment_confidence = [_number(row.get("v4_confidence"), default=np.nan) for row in usable]
        segment_confidence = [value for value in segment_confidence if np.isfinite(value)]
        confidence.extend(segment_confidence)
        if len(segment_confidence) > 1:
            confidence_diffs.extend(np.diff(segment_confidence).tolist())
        sample_count += len(usable)
    return {
        "status": "MEASURED",
        "segment_count": len(grouped),
        "sample_count": sample_count,
        "raw_prediction_flip_rate": float(switches / max(transitions, 1)),
        "raw_prediction_switches": switches,
        "false_transition_count": false_transitions,
        "mean_confidence": None if not confidence else float(np.mean(confidence)),
        "median_confidence": None if not confidence else float(np.median(confidence)),
        "confidence_jitter": None if not confidence_diffs else float(np.std(confidence_diffs)),
        "longest_correct_run": int(max(correct_runs, default=0)),
    }


def tracking_metrics(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize anonymous track lifecycle; ID switches need manual GT."""

    seen: set[str] = set()
    first_seen: dict[str, float] = {}
    last_seen: dict[str, float] = {}
    temporary_lost = 0
    expired = 0
    for event in events:
        timestamp = _number(event.get("timestamp_seconds"))
        for subject in event.get("subjects", []) or []:
            subject_id = str(subject.get("subject_id", ""))
            if not subject_id:
                continue
            if subject_id not in seen:
                seen.add(subject_id)
                first_seen[subject_id] = timestamp
            last_seen[subject_id] = timestamp
            state = str(subject.get("tracking_state", ""))
            temporary_lost += int(state == "TEMPORARILY_LOST")
            expired += int(state == "EXPIRED")
    durations = [last_seen[key] - first_seen[key] for key in seen]
    return {
        "status": "MEASURED" if events else "NO_TRACK_EVENTS",
        "new_track_count": len(seen),
        "temporarily_lost_observations": temporary_lost,
        "expired_observations": expired,
        "average_continuous_track_duration_seconds": None if not durations else float(np.mean(durations)),
        "id_switch_evidence": "NOT_MEASURED_NO_MANUAL_TRACK_ID_GROUND_TRUTH",
        "face_or_biometric_identity_used": False,
    }


__all__ = [
    "CORE_VISIBILITY_FIELDS",
    "aggregate_detection_metrics",
    "detection_frame_metrics",
    "detection_metrics_by_scenario",
    "frozen_v4_metrics",
    "per_person_core_metrics",
    "temporal_stability_metrics",
    "tracking_metrics",
]
