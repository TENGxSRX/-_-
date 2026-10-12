"""Train an isolated five-class webcam Human-GT experiment.

This is deliberately separate from the live demo and rgb_native_v5.  It uses
only explicit Human-GT records, groups by segment, and calibrates UNKNOWN on
development-session out-of-fold predictions only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.posture_ai.features.upper_body_posture_v2 import FEATURES, SCHEMA, build_features

LABELS = ("TUP", "TLF", "TLB", "TLL", "TLR")
FEATURE_SCHEMA = "upper_body_posture_v2:NO_HIP"
HUMAN_GT_ROOT = ROOT / "datasets/derived/webcam_human_gt_v1"
DEFAULT_OUTPUT = ROOT / "outputs/models/webcam_p001_fiveclass_exp_v1"
DEFAULT_HOLDOUT_SESSION = "SESSION_20260919T035749Z_P001"
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_human_gt(root: Path = HUMAN_GT_ROOT) -> tuple[list[dict], dict]:
    accepted: list[dict] = []
    rejected_counts: Counter[str] = Counter()
    rejected_files = sorted(root.glob("SESSION_*/rejected_evidence.jsonl"))
    for path in rejected_files:
        for record in _jsonl(path):
            rejected_counts[str(record.get("reason", "UNKNOWN"))] += 1

    expected_features = tuple(FEATURES["NO_HIP"])
    invalid: Counter[str] = Counter()
    source_files: list[dict] = []
    for path in sorted(root.glob("SESSION_*/accepted_observations.jsonl")):
        source_files.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
        for record in _jsonl(path):
            reason = None
            if record.get("record_type") != "ACCEPTED_HUMAN_GT_OBSERVATION":
                reason = "WRONG_RECORD_TYPE"
            elif record.get("label_source") != "HUMAN" or record.get("label_status") != "HUMAN_GROUND_TRUTH":
                reason = "NON_HUMAN_LABEL_PROVENANCE"
            elif record.get("human_label") not in LABELS:
                reason = "UNKNOWN_LABEL"
            elif record.get("feature_schema_version") != FEATURE_SCHEMA:
                reason = "FEATURE_SCHEMA_MISMATCH"
            elif tuple(record.get("feature_names", ())) != expected_features:
                reason = "FEATURE_ORDER_MISMATCH"
            elif record.get("pose_state") != "POSE_OK" or not record.get("pose_valid") or not record.get("core_valid"):
                reason = "INVALID_POSE_STATE"
            elif record.get("crop_safety_rejected"):
                reason = "CROP_SAFETY_REJECTED"
            else:
                values = np.asarray(record.get("features"), dtype=float)
                if values.shape != (len(expected_features),) or not np.isfinite(values).all():
                    reason = "INVALID_FEATURE_VALUES"
            if reason:
                invalid[reason] += 1
            else:
                accepted.append(record)
    if invalid:
        raise ValueError(f"INVALID_HUMAN_GT_RECORDS: {dict(invalid)}")
    if not accepted:
        raise ValueError("NO_VERIFIED_HUMAN_GT")
    audit = {
        "accepted_records": len(accepted),
        "rejected_evidence_records": int(sum(rejected_counts.values())),
        "rejected_reasons": dict(rejected_counts),
        "source_files": source_files,
        "participants": sorted({str(r["participant_id"]) for r in accepted}),
        "sessions": sorted({str(r["session_id"]) for r in accepted}),
        "segments": sorted({str(r["segment_id"]) for r in accepted}),
        "labels": dict(Counter(str(r["human_label"]) for r in accepted)),
        "schema": FEATURE_SCHEMA,
        "label_source": sorted({str(r["label_source"]) for r in accepted}),
        "duplicate_feature_rows": len(accepted) - len({tuple(r["features"]) for r in accepted}),
    }
    return accepted, audit


def feature_parity(records: list[dict]) -> dict:
    differences = []
    for record in records:
        xyz = np.asarray(record["world_landmarks_xyz"], dtype=float)
        rebuilt = build_features(xyz, "NO_HIP")[0]
        recorded = np.asarray(record["features"], dtype=float)
        differences.append(float(np.max(np.abs(rebuilt - recorded))))
    return {
        "schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURES["NO_HIP"]),
        "records_checked": len(records),
        "max_abs_difference": max(differences),
        "exact_within_1e-9": bool(max(differences) <= 1e-9),
        "decision": "RUNTIME_FEATURE_PARITY_CONFIRMED" if max(differences) <= 1e-9 else "PARITY_FAILED",
    }


def models() -> dict[str, object]:
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=42),
        ),
        "RBF_SVM": make_pipeline(
            StandardScaler(),
            SVC(C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42),
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, learning_rate=0.08,
            l2_regularization=1.0, early_stopping=False, random_state=42,
        ),
    }


def known_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...] = LABELS) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else 0.0,
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "labels": list(labels),
    }


def apply_unknown(probabilities: np.ndarray, classes: np.ndarray, threshold: float) -> np.ndarray:
    best = probabilities.argmax(axis=1)
    confidence = probabilities[np.arange(len(probabilities)), best]
    prediction = classes[best].astype(object)
    prediction[confidence < threshold] = "UNKNOWN"
    return prediction


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, threshold: float) -> dict:
    prediction = apply_unknown(probabilities, classes, threshold)
    accepted = prediction != "UNKNOWN"
    accepted_metrics = known_metrics(y_true[accepted], prediction[accepted]) if accepted.any() else None
    full = known_metrics(y_true, prediction)
    return {
        "threshold": threshold,
        "coverage": float(accepted.mean()),
        "unknown_count": int((~accepted).sum()),
        "unknown_percentage": float((~accepted).mean() * 100.0),
        "end_to_end_known_class_metrics": full,
        "accepted_only_metrics": accepted_metrics,
    }


def fit_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    unique_groups = np.unique(groups)
    cv = GroupKFold(n_splits=min(5, len(unique_groups)))
    probabilities = np.full((len(y), len(LABELS)), np.nan, dtype=float)
    folds = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups), 1):
        assert set(groups[train_idx]).isdisjoint(set(groups[valid_idx]))
        fitted = clone(model).fit(X[train_idx], y[train_idx])
        fold_classes = np.asarray(fitted.classes_)
        probabilities[valid_idx] = fitted.predict_proba(X[valid_idx])[:, [list(fold_classes).index(label) for label in LABELS]]
        folds.append({
            "fold": fold,
            "train_segments": sorted(set(groups[train_idx])),
            "validation_segments": sorted(set(groups[valid_idx])),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(valid_idx)),
        })
    assert np.isfinite(probabilities).all()
    return probabilities, np.asarray(LABELS), folds


def choose_threshold(y: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> tuple[float, list[dict]]:
    table = [threshold_metrics(y, probabilities, classes, threshold) for threshold in THRESHOLDS]
    best_score = max(item["end_to_end_known_class_metrics"]["macro_f1"] for item in table)
    eligible = [item for item in table if item["end_to_end_known_class_metrics"]["macro_f1"] >= best_score - 1e-12]
    # Tie-break toward the stricter refusal threshold.
    return max(eligible, key=lambda item: item["threshold"])["threshold"], table


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def markdown_report(report: dict) -> str:
    selected = report["selected"]
    test = report["test"]
    lines = [
        "# Webcam P001 Five-Class Experimental Model V1",
        "",
        "สถานะ: **EXPERIMENTAL_ONLY — ไม่ใช่ V5 production และยังไม่ได้ live-validate**",
        "",
        "## Data and split",
        "",
        f"- Accepted Human GT: `{report['data']['accepted_records']}` records, `{len(report['data']['segments'])}` segments, participant `{report['data']['participants']}`",
        f"- Classes: `{report['data']['labels']}`",
        f"- Feature schema: `{report['feature_parity']['schema']}`",
        f"- Feature parity: `{report['feature_parity']['decision']}`, max absolute difference `{report['feature_parity']['max_abs_difference']:.3g}`",
        f"- Development session: `{report['split']['development_session']}`; test session: `{report['split']['holdout_session']}`",
        f"- Development: `{report['split']['development_rows']}` rows / `{len(report['split']['development_segments'])}` segments",
        f"- Test: `{report['split']['test_rows']}` rows / `{len(report['split']['test_segments'])}` segments",
        "- All validation folds group by segment; no segment appears in both train and validation.",
        "- Test session was not used for model or UNKNOWN-threshold selection.",
        "",
        "## Model selection",
        "",
        f"- Selected model: **{selected['model']}**",
        f"- Selected UNKNOWN threshold: `{selected['unknown_threshold']}`",
        "- Selection used development-session GroupKFold out-of-fold results only.",
        "",
        "| Model | Development OOF macro-F1 | Development OOF balanced accuracy | Threshold | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in report["model_comparison"].items():
        chosen = item["threshold_selection"]
        lines.append(f"| {name} | {item['raw_oof']['macro_f1']:.4f} | {item['raw_oof']['balanced_accuracy']:.4f} | {chosen['threshold']:.2f} | {chosen['coverage'] * 100:.1f}% |")
    lines += [
        "",
        "## Untouched session holdout",
        "",
        f"- Raw classifier metrics: `{json.dumps(test['raw_metrics'], ensure_ascii=False)}`",
        f"- With UNKNOWN refusal: `{json.dumps(test['threshold_metrics'], ensure_ascii=False)}`",
        "- This is same-participant, same-camera-session evidence only; it is not participant-independent generalization.",
        "- Confidence intervals are not reported as reliable because the holdout contains only five segments from P001.",
        "",
        "## Current UNKNOWN/OOD diagnosis",
        "",
        "- The current live V2 artifact was trained on a different source-domain distribution.",
        "- Webcam Human GT features use the exact same six-feature runtime schema, so the new experiment has feature parity.",
        "- The old artifact's training ranges exclude most webcam values, especially shoulder yaw and nose/shoulder z; this explains the previous `POSSIBLE_OOD_EXPERIMENTAL` refusals.",
        "- This experiment does not weaken the old safety gate and does not replace the live demo model.",
        "",
        "## Limitations",
        "",
        "- One participant (`P001`) and two non-empty sessions.",
        "- No participant-independent test.",
        "- Left/right convention is consistent within this camera/person collection but not externally validated against a mirrored camera.",
        "- No live webcam test was run in this milestone.",
        "- Human GT is a controlled calibration dataset, not broad office CCTV evidence.",
    ]
    return "\n".join(lines) + "\n"


def run(output: Path = DEFAULT_OUTPUT, holdout_session: str = DEFAULT_HOLDOUT_SESSION) -> dict:
    records, data_audit = load_human_gt()
    parity = feature_parity(records)
    if not parity["exact_within_1e-9"]:
        raise ValueError("RUNTIME_FEATURE_PARITY_FAILED")
    sessions = sorted({r["session_id"] for r in records})
    if holdout_session not in sessions or len(sessions) != 2:
        raise ValueError("EXPECTED_TWO_SESSION_HOLDOUT_NOT_AVAILABLE")
    dev_records = [r for r in records if r["session_id"] != holdout_session]
    test_records = [r for r in records if r["session_id"] == holdout_session]
    X_dev = np.asarray([r["features"] for r in dev_records], dtype=float)
    y_dev = np.asarray([r["human_label"] for r in dev_records], dtype=object)
    g_dev = np.asarray([r["segment_id"] for r in dev_records], dtype=object)
    X_test = np.asarray([r["features"] for r in test_records], dtype=float)
    y_test = np.asarray([r["human_label"] for r in test_records], dtype=object)
    g_test = np.asarray([r["segment_id"] for r in test_records], dtype=object)
    if not set(y_dev) == set(LABELS) or not set(y_test) == set(LABELS):
        raise ValueError("ALL_CLASSES_REQUIRED_IN_DEVELOPMENT_AND_TEST")
    if not set(g_dev).isdisjoint(set(g_test)):
        raise AssertionError("SEGMENT_LEAKAGE")

    comparison = {}
    fitted_models = {}
    for name, estimator in models().items():
        oof, classes, folds = fit_oof(X_dev, y_dev, g_dev, estimator)
        raw_pred = classes[oof.argmax(axis=1)]
        threshold, threshold_table = choose_threshold(y_dev, oof, classes)
        comparison[name] = {
            "raw_oof": known_metrics(y_dev, raw_pred),
            "threshold_selection": next(item for item in threshold_table if item["threshold"] == threshold),
            "threshold_table": threshold_table,
            "folds": folds,
        }
        fitted_models[name] = clone(estimator).fit(X_dev, y_dev)
    selected_name = max(
        comparison,
        key=lambda name: (comparison[name]["raw_oof"]["macro_f1"], -list(models()).index(name)),
    )
    selected_threshold = comparison[selected_name]["threshold_selection"]["threshold"]
    selected_model = fitted_models[selected_name]
    test_classes = np.asarray(selected_model.classes_)
    test_probabilities = selected_model.predict_proba(X_test)
    test_raw_pred = test_classes[test_probabilities.argmax(axis=1)]
    test_threshold = threshold_metrics(y_test, test_probabilities, test_classes, selected_threshold)
    selected = {"model": selected_name, "unknown_threshold": selected_threshold,
                "feature_schema": FEATURE_SCHEMA, "features": list(FEATURES["NO_HIP"]),
                "development_oof_macro_f1": comparison[selected_name]["raw_oof"]["macro_f1"]}
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"VERSIONED_OUTPUT_ALREADY_EXISTS: {output}")
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "experiment": "webcam_p001_fiveclass_exp_v1",
        "model": selected_model,
        "labels": list(LABELS),
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURES["NO_HIP"]),
        "unknown_threshold": selected_threshold,
        "unknown_policy": "UNKNOWN when max predict_proba is below the development-session OOF-calibrated threshold",
    }
    joblib.dump(bundle, output / "model.joblib")
    split = {
        "development_session": sessions[0] if sessions[1] == holdout_session else sessions[1],
        "holdout_session": holdout_session,
        "development_rows": len(dev_records),
        "test_rows": len(test_records),
        "development_segments": sorted(set(g_dev)),
        "test_segments": sorted(set(g_test)),
        "no_segment_overlap": True,
        "participant_overlap_is_expected": True,
        "participant_independent": False,
    }
    report = {
        "experiment": "webcam_p001_fiveclass_exp_v1",
        "status": "EXPERIMENTAL_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "data": data_audit,
        "feature_parity": parity,
        "split": split,
        "model_comparison": comparison,
        "selected": selected,
        "test": {
            "raw_metrics": known_metrics(y_test, test_raw_pred),
            "threshold_metrics": test_threshold,
            "test_segment_count": len(set(g_test)),
            "test_segment_labels": {segment: str(y_test[np.flatnonzero(g_test == segment)[0]]) for segment in sorted(set(g_test))},
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
        "frozen_v4_sha256": "af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd",
        "limitations": [
            "One participant P001; no participant-independent generalization.",
            "Two non-empty sessions; five held-out segments only.",
            "No live webcam validation performed.",
            "Left/right convention is not independently validated under camera mirroring.",
            "This artifact is not rgb_native_v5 and is not connected to the production demo.",
        ],
    }
    write_json(output / "metrics.json", report)
    write_json(output / "metadata.json", {
        "experiment": report["experiment"], "status": report["status"], "source_commit": report["source_commit"],
        "dataset_root": str(HUMAN_GT_ROOT.relative_to(ROOT)), "source_files": data_audit["source_files"],
        "feature_schema": FEATURE_SCHEMA, "feature_names": list(FEATURES["NO_HIP"]),
        "model": selected, "split": split, "unknown_policy": bundle["unknown_policy"],
        "frozen_v4_sha256": report["frozen_v4_sha256"], "limitations": report["limitations"],
    })
    write_json(output / "labels.json", {"labels": list(LABELS), "unknown": "UNKNOWN", "label_source": "HUMAN"})
    write_json(output / "feature_config.json", {"schema": FEATURE_SCHEMA, "condition": "NO_HIP", "features": list(FEATURES["NO_HIP"])})
    write_json(output / "split_manifest.json", split)
    (ROOT / "reports/webcam_p001_fiveclass_exp_v1.md").write_text(markdown_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-session", default=DEFAULT_HOLDOUT_SESSION)
    args = parser.parse_args()
    result = run(args.output, args.holdout_session)
    print(json.dumps({"status": result["status"], "selected": result["selected"], "test": result["test"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
