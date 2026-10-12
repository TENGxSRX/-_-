"""Leakage-safe preparation for a future partial-pose Activity AI V1.

Nothing in this module treats a model output, geometry result, or tracking
state as a target.  Training requires explicit manual labels from the
behavior pilot collection contract and groups folds by anonymous participant.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.posture_ai.data.behavior_collection import ACTIVITY_LABELS, BEHAVIOR_COLLECTION_SCHEMA_VERSION, validate_behavior_labels
from src.posture_ai.data.cctv_metrics import frozen_v4_metrics, temporal_stability_metrics
from src.posture_ai.features.activity_partial_pose_v1 import (
    ACTIVITY_PARTIAL_POSE_V1_SCHEMA_VERSION,
    FEATURE_VARIANTS,
    activity_feature_names,
)


def load_behavior_rows(root: str | Path) -> list[dict[str, str]]:
    """Load only verified, manually labelled rows; reject schema ambiguities."""

    path = Path(root) / "samples.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows, start=2):
        if row.get("schema_version") != BEHAVIOR_COLLECTION_SCHEMA_VERSION:
            raise ValueError(f"row {index}: unexpected collection schema")
        if row.get("label_source") != "MANUAL_OPERATOR_ONLY":
            raise ValueError(f"row {index}: labels must be MANUAL_OPERATOR_ONLY")
        activity, posture = validate_behavior_labels(row.get("human_activity_label", ""), row.get("human_posture_label", ""))
        if row.get("human_label") != activity:
            raise ValueError(f"row {index}: human_label and human_activity_label disagree")
        if row.get("activity_feature_schema_version") != ACTIVITY_PARTIAL_POSE_V1_SCHEMA_VERSION:
            raise ValueError(f"row {index}: unexpected activity feature schema")
        if not row.get("subject_id") or not row.get("session_id"):
            raise ValueError(f"row {index}: participant/session group is required")
    return rows


def activity_training_readiness(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    groups = sorted({str(row.get("subject_id", "")) for row in values if row.get("subject_id")})
    labels = Counter(str(row.get("human_activity_label", "")) for row in values)
    per_participant = defaultdict(Counter)
    for row in values:
        per_participant[str(row.get("subject_id", ""))][str(row.get("human_activity_label", ""))] += 1
    missing_classes = [label for label in ACTIVITY_LABELS if labels[label] == 0]
    if not values:
        status = "READY_FOR_BEHAVIOR_DATA_COLLECTION"
    elif missing_classes or len(groups) < 3:
        status = "DATA_PRESENT_NOT_READY_FOR_PARTICIPANT_GROUPED_TRAINING"
    else:
        status = "READY_FOR_GROUPED_ACTIVITY_TRAINING"
    return {
        "status": status,
        "sample_count": len(values),
        "participant_count": len(groups),
        "participants": groups,
        "class_distribution": dict(labels),
        "missing_activity_classes": missing_classes,
        "per_participant_class_distribution": {key: dict(value) for key, value in sorted(per_participant.items())},
        "split_policy": "participant-independent grouped validation; never random frame split",
        "model_outputs": ["SITTING", "STANDING"],
        "runtime_uncertain_policy": "If a future model confidence is below its frozen threshold, emit UNCERTAIN and pause sitting temporal risk.",
    }


def _matrix(rows: list[Mapping[str, Any]], variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    names = activity_feature_names(variant)
    try:
        matrix = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"feature data does not match {variant}") from exc
    if not np.isfinite(matrix).all():
        raise ValueError(f"{variant} contains non-finite values")
    labels = np.asarray([str(row["human_activity_label"]) for row in rows], dtype=object)
    groups = np.asarray([str(row["subject_id"]) for row in rows], dtype=object)
    return matrix, labels, groups, names


def _models() -> dict[str, Any]:
    return {
        "LogisticRegression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))]),
        "RandomForest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=250, l2_regularization=0.1, random_state=42),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(ACTIVITY_LABELS), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(ACTIVITY_LABELS), average="macro", zero_division=0)),
        "per_class": {
            label: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])}
            for index, label in enumerate(ACTIVITY_LABELS)
        },
    }


def evaluate_grouped_activity_models(rows: list[Mapping[str, Any]], *, variants: Iterable[str] = FEATURE_VARIANTS) -> dict[str, Any]:
    """Run participant-grouped CV only after readiness validates the labels."""

    readiness = activity_training_readiness(rows)
    if readiness["status"] != "READY_FOR_GROUPED_ACTIVITY_TRAINING":
        raise ValueError(readiness["status"])
    groups_count = int(readiness["participant_count"])
    splits = min(5, groups_count)
    results: dict[str, Any] = {}
    for variant in variants:
        matrix, labels, groups, names = _matrix(rows, str(variant))
        splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=42)
        variant_result: dict[str, Any] = {"feature_names": list(names), "models": {}}
        for model_name, model in _models().items():
            predicted = np.empty(len(rows), dtype=object)
            fold_groups: list[dict[str, list[str]]] = []
            for train_index, test_index in splitter.split(matrix, labels, groups):
                train_groups, test_groups = set(groups[train_index]), set(groups[test_index])
                if train_groups & test_groups:
                    raise RuntimeError("participant overlap in grouped fold")
                model.fit(matrix[train_index], labels[train_index])
                predicted[test_index] = model.predict(matrix[test_index])
                fold_groups.append({"train_groups": sorted(train_groups), "test_groups": sorted(test_groups)})
            variant_result["models"][model_name] = {"grouped_cv": _metrics(labels, predicted), "fold_groups": fold_groups}
        results[str(variant)] = variant_result
    return {"status": "GROUPED_ACTIVITY_EVALUATION_COMPLETED", "readiness": readiness, "variants": results}


def v4_cctv_baseline_readiness(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Prepare frozen-V4 evaluation strictly against sitting human labels."""

    sitting = [row for row in rows if row.get("human_activity_label") == "SITTING" and row.get("human_posture_label") in {"TUP", "TLB"}]
    usable = [row for row in sitting if row.get("v4_prediction") in {"TUP", "TLB"}]
    if not usable:
        return {"status": "NOT_EVALUATED_NO_VERIFIED_SITTING_V4_ROWS", "labelled_sitting_rows": len(sitting), "usable_v4_rows": 0}
    truth = [str(row["human_posture_label"]) for row in usable]
    prediction = [str(row["v4_prediction"]) for row in usable]
    return {
        "status": "READY_TO_EVALUATE_FROZEN_V4_ON_CCTV_GROUND_TRUTH",
        "metrics": frozen_v4_metrics(truth, prediction),
        "temporal": temporal_stability_metrics(usable),
        "by_participant": {
            participant: frozen_v4_metrics([str(row["human_posture_label"]) for row in values], [str(row["v4_prediction"]) for row in values])
            for participant, values in _group_by_participant(usable).items()
        },
    }


def behavior_stability_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Prepare behavior-quality metrics without treating missing outputs as errors.

    Future runtime telemetry may add ``activity_prediction`` and
    ``activity_confidence``.  Until an Activity AI exists, this function
    explicitly reports the relevant metrics as unavailable rather than
    producing a proxy score from the geometry gate.
    """

    rows = list(records)
    predicted = [
        row for row in rows
        if str(row.get("human_activity_label", "")) in ACTIVITY_LABELS
        and str(row.get("activity_prediction", "")) in {*ACTIVITY_LABELS, "UNCERTAIN"}
    ]
    if not predicted:
        return {
            "activity": {"status": "NOT_EVALUATED_NO_ACTIVITY_AI_PREDICTIONS", "required_metrics": ["SITTING precision/recall/F1", "STANDING precision/recall/F1", "macro F1", "UNCERTAIN rate", "activity flip rate", "transition latency"]},
            "posture": {"status": "NOT_EVALUATED_NO_VERIFIED_POSTURE_PREDICTIONS", "required_metrics": ["TUP/TLB precision/recall/F1", "macro F1", "prediction flip rate", "confidence when wrong"]},
            "temporal": {"status": "NOT_RECORDED_UNTIL_RUNTIME_TELEMETRY_EXISTS", "required_metrics": ["false warning count", "warning onset delay", "incorrect risk accumulation", "pose-loss interruptions"]},
        }
    truth = np.asarray([str(row["human_activity_label"]) for row in predicted], dtype=object)
    raw_prediction = np.asarray([str(row["activity_prediction"]) for row in predicted], dtype=object)
    certain = raw_prediction != "UNCERTAIN"
    metrics = {"status": "MEASURED", "uncertain_rate": float(np.mean(~certain))}
    if np.any(certain):
        metrics.update(_metrics(truth[certain], raw_prediction[certain]))
    else:
        metrics.update({"accuracy": None, "balanced_accuracy": None, "macro_f1": None, "per_class": {}})
    per_sequence: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predicted:
        per_sequence[(str(row.get("subject_id", "")), str(row.get("session_id", "")))].append(row)
    flips = comparisons = transitions = resolved_transitions = 0
    latencies: list[float] = []
    for sequence in per_sequence.values():
        ordered = sorted(sequence, key=lambda row: float(row.get("timestamp_seconds", 0.0)))
        prior_prediction = None
        prior_truth = None
        for index, row in enumerate(ordered):
            current_prediction = str(row["activity_prediction"])
            current_truth = str(row["human_activity_label"])
            if prior_prediction is not None:
                comparisons += 1
                flips += int(current_prediction != prior_prediction)
            if prior_truth is not None and current_truth != prior_truth:
                transitions += 1
                for later in ordered[index:]:
                    if str(later["activity_prediction"]) == current_truth:
                        latencies.append(max(0.0, float(later.get("timestamp_seconds", 0.0)) - float(row.get("timestamp_seconds", 0.0))))
                        resolved_transitions += 1
                        break
            prior_prediction, prior_truth = current_prediction, current_truth
    metrics.update({
        "activity_flip_rate": None if not comparisons else float(flips / comparisons),
        "activity_flip_count": flips,
        "transition_count": transitions,
        "resolved_transition_count": resolved_transitions,
        "transition_latency_seconds_median": None if not latencies else float(np.median(latencies)),
    })
    posture_rows = [row for row in rows if str(row.get("human_posture_label", "")) in {"TUP", "TLB"} and str(row.get("v4_prediction", "")) in {"TUP", "TLB"}]
    posture = v4_cctv_baseline_readiness(posture_rows) if posture_rows else {"status": "NOT_EVALUATED_NO_VERIFIED_POSTURE_PREDICTIONS"}
    temporal_rows = [row for row in rows if row.get("temporal_risk_level")]
    temporal = {"status": "NOT_RECORDED_UNTIL_RUNTIME_TELEMETRY_EXISTS"} if not temporal_rows else {
        "status": "MEASURED_RUNTIME_TELEMETRY", "warning_count": sum(str(row.get("temporal_risk_level")) in {"WARNING", "HIGH_DURATION_RISK"} for row in temporal_rows),
        "pose_loss_interruptions": sum(str(row.get("temporal_risk_level")) == "POSE_LOST" for row in temporal_rows),
        "false_warning_count": "REQUIRES_HUMAN_REVIEW", "warning_onset_delay": "REQUIRES_HUMAN_TRANSITION_TIMESTAMPS",
    }
    return {"activity": metrics, "posture": posture, "temporal": temporal}


def _group_by_participant(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["subject_id"])].append(row)
    return grouped


def export_activity_candidate(evaluation: Mapping[str, Any], rows: list[Mapping[str, Any]], output_dir: str | Path) -> Path:
    """Fit/export one explicitly selected grouped-CV candidate after evaluation."""

    candidates = []
    for variant, details in dict(evaluation["variants"]).items():
        for name, output in dict(details["models"]).items():
            candidates.append((float(output["grouped_cv"]["macro_f1"]), str(variant), str(name)))
    _score, variant, model_name = max(candidates, key=lambda item: item[0])
    matrix, labels, _groups, names = _matrix(rows, variant)
    model = _models()[model_name]
    model.fit(matrix, labels)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination / "model.joblib")
    (destination / "feature_config.json").write_text(json.dumps({"schema_version": ACTIVITY_PARTIAL_POSE_V1_SCHEMA_VERSION, "variant": variant, "feature_names": list(names), "uncertain_probability_threshold": 0.60, "person_id_feature": False}, indent=2), encoding="utf-8")
    (destination / "metadata.json").write_text(json.dumps({"model_type": model_name, "selection_metric": "participant_grouped_cv_macro_f1", "evaluation": evaluation, "source": "manual behavior_partial_pose_collection_v1 only", "limitations": ["No V4 modification", "UNCERTAIN is runtime confidence handling, not a training label"]}, indent=2), encoding="utf-8")
    return destination


__all__ = [
    "activity_training_readiness", "behavior_stability_metrics", "evaluate_grouped_activity_models", "export_activity_candidate", "load_behavior_rows", "v4_cctv_baseline_readiness",
]
