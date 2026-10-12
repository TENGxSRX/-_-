"""Small fixed benchmark with participant-grouped CV and a predeclared holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.posture_ai.data.multiposture import load_multiposture, validate_multiposture
from src.posture_ai.features.upper_body_posture_v1 import SCHEMA, CORE_NAMES, FEATURE_NAMES, SUBTYPES, build_features
from src.posture_ai.inference.upper_body_posture_v1 import classify_features, verify_v4

LABELS = sorted(SUBTYPES)


def metrics(y, pred, groups, labels=LABELS):
    def scores(truth, predictions):
        binary_true, binary_pred = truth != "TUP", predictions != "TUP"
        confusion_labels = sorted(set(labels) | set(predictions))
        return {
            "samples": len(truth), "macro_f1": float(f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)),
            "per_class": classification_report(truth, predictions, labels=labels, output_dict=True, zero_division=0),
            "confusion_labels": confusion_labels, "confusion_matrix": confusion_matrix(truth, predictions, labels=confusion_labels).tolist(),
            "binary": {"correct_recall": float(recall_score(binary_true, binary_pred, pos_label=False, zero_division=0)),
                       "incorrect_recall": float(recall_score(binary_true, binary_pred, zero_division=0)),
                       "balanced_accuracy": float(balanced_accuracy_score(binary_true, binary_pred))},
        }
    result = scores(y, pred)
    result["participants"] = {str(g): scores(y[groups == g], pred[groups == g]) for g in np.unique(groups)}
    return result


def refusal_metrics(bundle, x, y, groups):
    records = [classify_features(bundle, row[None]) for row in x]
    accepted = np.array([r["output"] != "UNKNOWN" for r in records])
    inverse = {v: k for k, v in SUBTYPES.items()}
    pred = np.array([inverse.get(r["subtype"], "UNKNOWN") for r in records])
    return {
        "policy": "EXPERIMENTAL: training marginal min/max and unanimous final/grouped-fold models; no calibrated probability",
        "coverage": float(accepted.mean()), "unknown": int((~accepted).sum()),
        "reasons": pd.Series([r["reason"] for r in records]).value_counts().to_dict(),
        "coverage_per_class": {label: float(accepted[y == label].mean()) for label in LABELS if np.any(y == label)},
        "coverage_per_participant": {str(g): float(accepted[groups == g].mean()) for g in np.unique(groups)},
        "accepted_metrics": metrics(y[accepted], pred[accepted], groups[accepted], bundle["labels"]) if accepted.any() else None,
        "end_to_end_correct_recall": float(np.mean(pred[y == "TUP"] == "TUP")),
        "end_to_end_incorrect_recall": float(np.mean((pred[y != "TUP"] != "TUP") & (pred[y != "TUP"] != "UNKNOWN"))),
    }


def load_drivenet():
    directory = ROOT / "datasets/processed/rgb_native"
    frame = pd.read_csv(directory / "manifest.csv")
    frame = frame[frame.core_accepted & frame.mapped_project_label.isin(["TUP", "TLB"])]
    if not all(frame.mapped_project_label == frame.source_label.map({"SS": "TUP", "LB": "TLB"})):
        raise ValueError("DriveNet source-label join mismatch")
    core = np.load(directory / "features.npz")["world"][frame.feature_row.astype(int)].reshape(-1, 5, 3)[:, 1:]
    return frame, build_features(core)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("multiposture", "drivenet"), default="multiposture")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    v4_before = verify_v4()
    output = Path(args.output or f"outputs/models/upper_body_posture_v1_{args.dataset}_exp").resolve()
    if output == (ROOT / "outputs/models/rgb_native_v4").resolve() or (ROOT / "outputs/models/rgb_native_v4").resolve() in output.parents:
        parser.error("Frozen V4 output forbidden")
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "datasets/raw/multiposture/data.csv"
    if args.dataset == "multiposture":
        frame = load_multiposture(source)
        audit = validate_multiposture(frame)
        core = frame[[f"{name}_{axis}" for name in CORE_NAMES for axis in ("x", "y", "z")]].to_numpy().reshape(-1, 4, 3)
        x = build_features(core)
        y, groups = frame.upperbody_label.astype(str).to_numpy(), frame.subject.astype(str).to_numpy()
        labels, held_subjects, n_splits = LABELS, ["12", "13"], 5
    else:
        source = ROOT / "datasets/processed/rgb_native/features.npz"
        frame, x = load_drivenet()
        y, groups = frame.mapped_project_label.astype(str).to_numpy(), frame.participant_id.astype(str).to_numpy()
        labels, held_subjects, n_splits = ["TLB", "TUP"], ["3"], 2
        audit = {"rows": len(frame), "participants": sorted(set(groups)), "upper_labels": pd.Series(y).value_counts().to_dict(),
                 "provenance": "Verified source protocol SS/TUP and LB/TLB compatible mappings; not office ergonomic GT",
                 "manifest_sha256": hashlib.sha256((source.parent / "manifest.csv").read_bytes()).hexdigest()}
    if set(y) != set(labels):
        raise ValueError("Unverified or missing source classes")
    dev, holdout = ~np.isin(groups, held_subjects), np.isin(groups, held_subjects)
    if not holdout.any() or set(groups[dev]) & set(groups[holdout]):
        raise ValueError("Participant split invalid")
    xd, yd, gd = x[dev], y[dev], groups[dev]
    families = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, class_weight="balanced", random_state=42)),
        "RBF_SVM": make_pipeline(StandardScaler(), SVC(C=1, gamma="scale", class_weight="balanced")),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1, class_weight="balanced", early_stopping=False, random_state=42),
    }
    results, committees = {}, {}
    folds = list(GroupKFold(n_splits=n_splits).split(xd, yd, gd))
    for name, model in families.items():
        started = time.perf_counter()
        pred = np.empty(len(yd), dtype=object)
        fitted, fold_scores = [], []
        for train, valid in folds:
            assert not set(gd[train]) & set(gd[valid])
            estimator = clone(model).fit(xd[train], yd[train])
            pred[valid] = estimator.predict(xd[valid])
            fold_scores.append(float(f1_score(yd[valid], pred[valid], labels=labels, average="macro", zero_division=0)))
            fitted.append(estimator)
        results[name] = {"grouped_oof": metrics(yd, pred, gd, labels), "fold_macro_f1": fold_scores,
                         "fold_std": float(np.std(fold_scores)), "fit_seconds": time.perf_counter() - started}
        committees[name] = fitted
    # Fixed family order breaks exact ties toward simpler models. Holdout never selects.
    selected = max(families, key=lambda name: results[name]["grouped_oof"]["macro_f1"])
    model = clone(families[selected]).fit(xd, yd)
    bundle = {"schema": SCHEMA, "features": FEATURE_NAMES, "labels": labels, "model": model, "committee": committees[selected],
              "minimum": xd.min(axis=0), "maximum": xd.max(axis=0)}
    started = time.perf_counter()
    for _ in range(100):
        classify_features(bundle, xd[:1])
    inference_ms = (time.perf_counter() - started) * 10
    report = {"schema": SCHEMA, "status": "EXPERIMENTAL_SOURCE_DOMAIN_CANDIDATE", "source_audit": audit,
              "dataset": args.dataset, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "ground_truth": "HUMAN_GROUND_TRUTH: source-verified protocol labels; compatibility limits apply",
              "features": FEATURE_NAMES, "landmarks": CORE_NAMES, "subtypes": {label: SUBTYPES[label] for label in labels},
              "trained_at_utc": datetime.now(timezone.utc).isoformat(),
              "split": {"development_subjects": sorted(set(gd), key=int), "holdout_subjects": held_subjects,
                        "development_samples": int(dev.sum()), "holdout_samples": int(holdout.sum()),
                        "method": f"{n_splits}-fold GroupKFold on development; final fixed holdout; no frame-random split"},
              "development_class_counts": pd.Series(yd).value_counts().to_dict(),
              "holdout_class_counts": pd.Series(y[holdout]).value_counts().to_dict(),
              "candidates": results, "selected": selected, "holdout": metrics(y[holdout], model.predict(x[holdout]), groups[holdout], labels),
              "holdout_refusal": refusal_metrics(bundle, x[holdout], y[holdout], groups[holdout]),
              "inference_ms_per_person_including_committee": inference_ms,
              "v4_sha256": v4_before,
              "limitations": ["Processed source coordinates are not proven equivalent to crop MediaPipe world axes; no blind source fusion.",
                              "Direction-to-correctness is the requested product mapping, not independent ergonomic or medical GT.",
                              "Holdout is participant-disjoint for this candidate; historical experiments used this corpus, so results remain exploratory.",
                              "No SLOUCH GT. No verified office CCTV accuracy. Experimental rejection has no calibrated coverage guarantee."]}
    # Frozen candidate on a separate RGB source: diagnostic only, never used in fit/selection.
    rgb_dir = ROOT / "datasets/processed/rgb_native"
    if args.dataset == "multiposture" and (rgb_dir / "features.npz").exists():
        rgb, rx = load_drivenet()
        ry, rg = rgb.mapped_project_label.to_numpy(), rgb.participant_id.astype(str).to_numpy()
        report["rgb_transfer_diagnostic"] = {"source": "DriveNet v6 cached V4 extraction; SS/TUP and LB/TLB compatible only",
                                             "participants": sorted(set(rg)), "raw_metrics": metrics(ry, model.predict(rx), rg, ["TLB", "TUP"]),
                                             "refusal": refusal_metrics(bundle, rx, ry, rg), "training_rows": 0}
    assert verify_v4() == v4_before
    joblib.dump(bundle, output / "candidate.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "cv_macro_f1": {k: v["grouped_oof"]["macro_f1"] for k, v in results.items()},
                      "holdout_macro_f1": report["holdout"]["macro_f1"], "binary": report["holdout"]["binary"],
                      "holdout_coverage": report["holdout_refusal"]["coverage"], "inference_ms": inference_ms, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
