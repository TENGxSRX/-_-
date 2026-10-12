"""Exactly three feature strategies / three fixed families, grouped by person."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.train_upper_body_posture_v1 import metrics
from src.posture_ai.data.multiposture import load_multiposture, validate_multiposture
from src.posture_ai.features.upper_body_posture_v2 import CONDITIONS, FEATURES, SCHEMA, build_features, training_hip_dropout
from src.posture_ai.inference.upper_body_posture_v2 import classify_batch
from src.posture_ai.inference.upper_body_posture_v1 import verify_v4

OUTPUT = ROOT / "outputs/models/upper_body_posture_v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_hashes():
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "outputs/models").glob("upper_body_posture_v1*/*")) if p.is_file()} | {"v4": verify_v4()}


def load_sources():
    source = ROOT / "datasets/raw/multiposture/data.csv"
    frame = load_multiposture(source)
    audit = validate_multiposture(frame)
    names = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
    xyz = frame[[f"{n}_{axis}" for n in names for axis in "xyz"]].to_numpy().reshape(-1, 5, 3)
    yield "MultiPosture", xyz, frame.upperbody_label.astype(str).to_numpy(), frame.subject.astype(str).to_numpy(), ["12", "13"], 5, {"audit": audit, "source_sha256": digest(source)}
    directory = ROOT / "datasets/processed/rgb_native"
    frame = pd.read_csv(directory / "manifest.csv")
    frame = frame[frame.core_accepted & frame.mapped_project_label.isin(["TUP", "TLB"])]
    inventory = json.loads((directory / "source_manifest.json").read_text())["source_inventory"]
    by_file = {r["filename"]: r for r in inventory}
    for row in frame.itertuples():
        verified = by_file[row.source_filename]
        if (row.mapped_project_label != {"SS": "TUP", "LB": "TLB"}.get(row.source_label)
                or str(row.participant_id) != verified["participant_id"]
                or row.source_label != verified["source_label"] or row.feature_row < 0):
            raise ValueError("UNVERIFIED_DRIVENET_JOIN")
    xyz = np.load(directory / "features.npz")["world"][frame.feature_row.astype(int)].reshape(-1, 5, 3)
    yield "DriveNet", xyz, frame.mapped_project_label.to_numpy(), frame.participant_id.astype(str).to_numpy(), ["3"], 2, {"manifest_sha256": digest(directory / "manifest.csv"), "features_sha256": digest(directory / "features.npz"), "inventory_sha256": digest(directory / "source_manifest.json")}


def fit_training_fold(model, xyz, y, condition):
    # This function receives ONLY the already-sliced training fold.
    if condition == "HIP_DROPOUT_ROBUST":
        xyz, y = training_hip_dropout(xyz, y)
    x = build_features(xyz, condition)
    return clone(model).fit(x, y), x


def make_bundle(model, committee, training_x, labels, condition):
    minimum, maximum = training_x.min(axis=0), training_x.max(axis=0)
    if condition == "HIP_DROPOUT_ROBUST":
        observed = training_x[training_x[:, -1] == 1, 6:-1]
        if len(observed):
            minimum[6:-1], maximum[6:-1] = observed.min(axis=0), observed.max(axis=0)
    return {"schema": SCHEMA, "condition": condition, "features": FEATURES[condition], "labels": labels,
            "model": model, "committee": committee, "minimum": minimum, "maximum": maximum}


def refusal(bundle, x, y, groups):
    records = classify_batch(bundle, x)
    accepted = np.array([r["output"] != "UNKNOWN" for r in records])
    inverse = {"UPRIGHT": "TUP", "BACKWARD_OR_RECLINED": "TLB", "FORWARD_LEAN": "TLF", "LATERAL_LEFT": "TLL", "LATERAL_RIGHT": "TLR"}
    pred = np.array([inverse.get(r["subtype"], "UNKNOWN") for r in records])
    return {"coverage": float(accepted.mean()), "unknown": int((~accepted).sum()),
            "reasons": pd.Series([r["reason"] for r in records]).value_counts().to_dict(),
            "coverage_per_class": {label: float(accepted[y == label].mean()) for label in bundle["labels"]},
            "coverage_per_participant": {str(g): float(accepted[groups == g].mean()) for g in np.unique(groups)},
            "accepted_metrics": metrics(y[accepted], pred[accepted], groups[accepted], bundle["labels"]) if accepted.any() else None}


def main():
    before = frozen_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if any(OUTPUT.iterdir()):
        raise ValueError("VERSIONED_OUTPUT_ALREADY_EXISTS")
    families = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, class_weight="balanced", random_state=42)),
        "RBF_SVM": make_pipeline(StandardScaler(), SVC(C=1, gamma="scale", class_weight="balanced")),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1, class_weight="balanced", early_stopping=False, random_state=42),
    }
    report = {"schema": SCHEMA, "status": "EXPERIMENTAL", "starting_commit": "e83cbe6336f5716e07f12e9eb4a10b0277bc3f55",
              "features": FEATURES, "frozen_hashes": before, "datasets": {},
              "selection_rule": "DriveNet compatibility first; maximize min(normal,fully-hidden) development OOF F1; within 0.01 prefer LR then SVM then HGB, and NO_HIP then HIP_DROPOUT_ROBUST. Holdout/CCTV never select.",
              "policy": "EXPERIMENTAL: training-only min/max on observed features and unanimous final/grouped-fold models; visibility>=0.5; no relaxed thresholds",
              "stress_protocol": "Independent deterministic left/right/both-hip-masked copies used only for diagnostic inference; originals and labels untouched; no fitting/threshold changes using held-out data",
              "limitations": ["Separate coordinate domains, never pooled", "Historical hip/nose accepted cache selection bias", "Nose is a head anchor, not a neck or trunk measurement", "DriveNet supports only TUP/TLB; unsupported forward/lateral poses cannot be guaranteed UNKNOWN", "No CCTV accuracy without human GT"]}
    saved = {}
    for source, xyz, y, groups, held, splits, provenance in load_sources():
        labels = sorted(set(y))
        expected = ["TLB", "TUP"] if source == "DriveNet" else ["TLB", "TLF", "TLL", "TLR", "TUP"]
        if labels != expected or not np.isfinite(xyz).all():
            raise ValueError("INVALID_VERIFIED_SOURCE")
        dev = ~np.isin(groups, held)
        xd, yd, gd = xyz[dev], y[dev], groups[dev]
        folds = list(GroupKFold(splits).split(xd, yd, gd))
        dataset = {"participants": sorted(set(groups)), "samples": len(y), "class_counts": pd.Series(y).value_counts().to_dict(),
                   "provenance": provenance, "holdout_participants": held, "development_samples": int(dev.sum()), "holdout_samples": int((~dev).sum()),
                   "validation": f"{splits}-fold GroupKFold development, fixed participant holdout", "conditions": {}}
        for condition in CONDITIONS:
            results = dataset["conditions"][condition] = {}
            for name, model in families.items():
                predictions = {key: np.empty(len(yd), dtype=object) for key in ("normal", "left_hidden", "right_hidden", "both_hidden")}
                committee, fold_scores, fold_audit = [], [], []
                for train, valid in folds:
                    assert not set(gd[train]) & set(gd[valid])
                    valid_before = xd[valid].copy()
                    estimator, _ = fit_training_fold(model, xd[train], yd[train], condition)
                    committee.append(estimator)
                    for key, missing in (("normal", ()), ("left_hidden", (3,)), ("right_hidden", (4,)), ("both_hidden", (3, 4))):
                        if condition == "CURRENT_BASELINE" and missing:
                            predictions[key][valid] = "UNKNOWN"
                            continue
                        view = xd[valid].copy()
                        view[:, missing, :] = np.nan
                        predictions[key][valid] = estimator.predict(build_features(view, condition))
                    assert np.array_equal(valid_before, xd[valid])
                    fold_scores.append(metrics(yd[valid], predictions["normal"][valid], gd[valid], labels)["macro_f1"])
                    fold_audit.append({"train_participants": sorted(set(gd[train])), "valid_participants": sorted(set(gd[valid])),
                                       "original_training_rows": len(train), "fitted_training_rows": len(train) * (2 if condition == "HIP_DROPOUT_ROBUST" else 1), "validation_rows_unchanged": True})
                result = {"grouped_oof": metrics(yd, predictions["normal"], gd, labels), "fold_macro_f1": fold_scores,
                          "fold_std": float(np.std(fold_scores)), "fold_audit": fold_audit,
                          "missing_hips_oof": {k: ({"coverage": 0, "reason": "REQUIRED_HIPS_MISSING"} if condition == "CURRENT_BASELINE" else metrics(yd, p, gd, labels)) for k, p in predictions.items() if k != "normal"}}
                # Persist fixed training on development only; holdout remains untouched.
                fitted, training_x = fit_training_fold(model, xd, yd, condition)
                bundle = make_bundle(fitted, committee, training_x, labels, condition)
                if source == "DriveNet" and condition != "CURRENT_BASELINE":
                    saved[condition, name] = bundle
                result["holdout"] = {}
                for key, missing in (("normal", ()), ("left_hidden", (3,)), ("right_hidden", (4,)), ("both_hidden", (3, 4))):
                    if condition == "CURRENT_BASELINE" and missing:
                        result["holdout"][key] = {"coverage": 0, "reason": "REQUIRED_HIPS_MISSING"}
                        continue
                    view = xyz[~dev].copy()
                    view[:, missing, :] = np.nan
                    x = build_features(view, condition)
                    result["holdout"][key] = {"raw": metrics(y[~dev], fitted.predict(x), groups[~dev], labels),
                                             "refusal": refusal(bundle, x, y[~dev], groups[~dev])}
                results[name] = result
                print(source, condition, name, round(result["grouped_oof"]["macro_f1"], 4), flush=True)
        report["datasets"][source] = dataset
    choices = []
    for condition, name in saved:
        r = report["datasets"]["DriveNet"]["conditions"][condition][name]
        score = min(r["grouped_oof"]["macro_f1"], r["missing_hips_oof"]["both_hidden"]["macro_f1"])
        choices.append((score, condition, name))
    best = max(r[0] for r in choices)
    selected = min((r for r in choices if r[0] >= best - 0.01), key=lambda r: (list(families).index(r[2]), CONDITIONS.index(r[1])))
    score, condition, name = selected
    report["selected"] = {"dataset": "DriveNet", "condition": condition, "family": name, "robust_oof_score": score, "features": FEATURES[condition], "supported_labels": ["TLB", "TUP"]}
    assert before == frozen_hashes()
    joblib.dump(saved[condition, name], OUTPUT / "candidate.joblib")
    report["candidate_sha256"] = digest(OUTPUT / "candidate.joblib")
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["selected"], indent=2))


if __name__ == "__main__":
    main()
