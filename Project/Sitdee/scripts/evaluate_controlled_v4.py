"""Evaluate frozen V4 against human-labelled controlled samples only.

This command never trains or tunes a model. It refuses to produce metrics when
the controlled dataset is absent, malformed, or contains unsupported labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.data.controlled_collection import CONTROLLED_LABELS  # noqa: E402


class ControlledDataNotReady(RuntimeError):
    pass


def _read_controlled_rows(data_root: Path, feature_names: list[str]) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    path = data_root / "samples.csv"
    if not path.exists():
        raise ControlledDataNotReady("WAITING_FOR_VERIFIED_HUMAN_LABELS: samples.csv does not exist")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ControlledDataNotReady("WAITING_FOR_VERIFIED_HUMAN_LABELS: samples.csv contains no accepted samples")
    required = {"human_label", "subject_id", "session_id", "sample_id", *feature_names}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ControlledDataNotReady("controlled sample schema is missing: " + ", ".join(missing))
    labels = np.asarray([str(row["human_label"]).upper() for row in rows], dtype=object)
    unsupported = sorted(set(labels) - set(CONTROLLED_LABELS))
    if unsupported:
        raise ControlledDataNotReady("unsupported human labels: " + ", ".join(unsupported))
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ControlledDataNotReady("duplicate sample_id values found")
    try:
        features = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ControlledDataNotReady("controlled feature values are not numeric") from exc
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ControlledDataNotReady("controlled feature values contain non-finite numbers")
    groups = np.asarray([str(row["subject_id"]).upper() for row in rows], dtype=object)
    if len(set(groups)) < 2:
        raise ControlledDataNotReady("at least two participant groups are required for leakage-safe diagnostics")
    return rows, features, labels, groups


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, probability_tlb: np.ndarray | None = None) -> dict[str, object]:
    y_true_bin = (y_true == "TLB").astype(int)
    y_pred_bin = (y_pred == "TLB").astype(int)
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=list(CONTROLLED_LABELS), average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=list(CONTROLLED_LABELS), average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(CONTROLLED_LABELS), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(CONTROLLED_LABELS), average="weighted", zero_division=0)),
        "sensitivity_tlb": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "specificity_tup": float(recall_score(1 - y_true_bin, 1 - y_pred_bin, zero_division=0)),
        "confusion_matrix_labels_tup_tlb": confusion_matrix(y_true, y_pred, labels=list(CONTROLLED_LABELS)).tolist(),
        "support": {label: int(np.sum(y_true == label)) for label in CONTROLLED_LABELS},
    }
    if len(np.unique(y_true_bin)) == 2:
        score = probability_tlb if probability_tlb is not None else y_pred_bin
        result["roc_auc_tlb"] = float(roc_auc_score(y_true_bin, score))
        result["pr_auc_tlb"] = float(average_precision_score(y_true_bin, score))
    else:
        result["roc_auc_tlb"] = None
        result["pr_auc_tlb"] = None
    return result


def evaluate(data_root: str | Path, model_dir: str | Path, report_path: str | Path, json_path: str | Path) -> dict[str, object]:
    model_path = Path(model_dir)
    feature_config = json.loads((model_path / "feature_config.json").read_text(encoding="utf-8"))
    feature_names = list(feature_config["columns"])
    rows, features, labels, groups = _read_controlled_rows(Path(data_root), feature_names)
    model = joblib.load(model_path / "model.joblib")
    predictions = np.asarray([str(value) for value in model.predict(features)], dtype=object)
    probability_tlb = None
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        classes = [str(value) for value in model.classes_]
        if "TLB" in classes:
            probability_tlb = probabilities[:, classes.index("TLB")]
    result: dict[str, object] = {
        "status": "FROZEN_V4_EVALUATED_ON_HUMAN_LABELS",
        "model_dir": str(model_path),
        "data_root": str(data_root),
        "sample_count": int(len(rows)),
        "participant_count": int(len(set(groups))),
        "participants": sorted(set(str(value) for value in groups)),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "human_label_source": "manual_operator_selection; never copied from V4",
        "overall": _metrics(labels, predictions, probability_tlb),
        "logged_v4_prediction_match_rate": float(np.mean([str(row.get("v4_prediction", "")) == pred for row, pred in zip(rows, predictions)])),
        "grouped_diagnostic": [],
        "limitations": [
            "This evaluates a frozen model; it does not train or tune.",
            "Participant-grouped diagnostics are descriptive and require the collection protocol to be followed.",
            "Results do not establish medical safety or universal real-world performance.",
        ],
    }
    for group in sorted(set(groups)):
        mask = groups == group
        probability = probability_tlb[mask] if probability_tlb is not None else None
        result["grouped_diagnostic"].append({"held_out_subject": str(group), "samples": int(np.sum(mask)), "metrics": _metrics(labels[mask], predictions[mask], probability)})  # type: ignore[union-attr]

    json_target = Path(json_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    overall = result["overall"]
    report = "\n".join([
        "# Frozen V4 evaluation on controlled human-labelled RGB data",
        "",
        f"Status: **{result['status']}**",
        f"Samples: {result['sample_count']} | participants: {result['participant_count']}",
        "",
        "ป้ายกำกับที่ใช้คำนวณคือ `human_label` จากผู้ควบคุมการเก็บข้อมูลเท่านั้น ไม่ใช่ผล V4",
        "",
        "## Overall frozen-model metrics",
        "",
        "```json",
        json.dumps(overall, ensure_ascii=False, indent=2),
        "```",
        "",
        f"รายงาน JSON: `{json_target}`",
        "",
        "ยังไม่ใช่ผลฝึก V5 และไม่ใช่ medical/health score",
    ])
    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen V4 on controlled manually labelled samples; never trains")
    parser.add_argument("--data-root", default="data/controlled_rgb_v5")
    parser.add_argument("--model", default="outputs/models/rgb_native_v4")
    parser.add_argument("--report", default="reports/v4_controlled_realworld_evaluation_th.md")
    parser.add_argument("--json", default="outputs/metrics/v4_controlled_realworld_evaluation.json")
    args = parser.parse_args()
    try:
        result = evaluate(args.data_root, args.model, args.report, args.json)
    except (ControlledDataNotReady, FileNotFoundError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
