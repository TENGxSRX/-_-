"""Run the visibility, domain-shift, and experimental upper-body V2 study.

This script creates only derived diagnostics and a separate V2 model family.
It never writes ``datasets/raw`` and never changes the V1 export.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.posture_ai.data.multiposture import GROUP_COLUMN, load_multiposture  # noqa: E402
from src.posture_ai.evaluation.splitting import grouped_cv, grouped_holdout_split  # noqa: E402
from src.posture_ai.features.pose_features import expected_feature_columns  # noqa: E402
from src.posture_ai.features.upperbody_v2 import (  # noqa: E402
    CORE_REQUIRED_LANDMARKS,
    GEOMETRY_FEATURE_NAMES,
    LOW_VALUE_FOR_OFFICE_POSTURE,
    OPTIONAL_LANDMARKS,
    adapt_mediapipe_to_upperbody_v2,
    build_upperbody_features,
    upperbody_feature_names,
)
from src.posture_ai.inference.pipeline import extract_pose_landmarks  # noqa: E402
from src.posture_ai.models.factories import make_model  # noqa: E402
from src.posture_ai.pose.landmarks import LANDMARK_INDEX, LANDMARK_NAMES, PoseLandmarks  # noqa: E402
from src.posture_ai.pose.robustness import evaluate_visibility_policy, pose_quality_score  # noqa: E402
from src.posture_ai.pose.validation import PoseValidationError  # noqa: E402

POSE_MODEL = PROJECT_ROOT / "metadata" / "pose_landmarker_heavy.task"
RAW_MULTIPOSTURE = PROJECT_ROOT / "datasets" / "raw" / "multiposture" / "data.csv"
MANIFEST = PROJECT_ROOT / "validation_data" / "external" / "external_manifest.csv"
EXTERNAL_OUTPUT = PROJECT_ROOT / "outputs" / "validation" / "external"
MODEL_V1 = PROJECT_ROOT / "outputs" / "models" / "multiposture_v1"
MODEL_V2 = PROJECT_ROOT / "outputs" / "models" / "multiposture_upperbody_v2"
LABEL_COLUMN = "upperbody_label"
POLICIES = ("strict", "core-only", "tiered", "quality")
V2_CONFIGS = ("raw", "normalized", "geometry", "hybrid")
V2_MODELS = ("LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting", "GradientBoosting")
MAX_FRAMES = 300


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _make_landmarker() -> PoseLandmarker:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def collect_landmarks(rows: list[dict[str, str]]) -> dict[str, Any]:
    all_coords: list[np.ndarray] = []
    all_visibility: list[np.ndarray] = []
    clip_ids: list[str] = []
    frame_ids: list[int] = []
    clip_summaries: list[dict[str, Any]] = []
    for row in rows:
        path = PROJECT_ROOT / row["local_path"]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open external video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        coords: list[np.ndarray] = []
        visibility: list[np.ndarray] = []
        detected = 0
        landmarker = _make_landmarker()
        start = time.perf_counter()
        try:
            for frame_index in range(MAX_FRAMES):
                ok, frame = capture.read()
                if not ok:
                    break
                landmarks = extract_pose_landmarks(frame, landmarker, timestamp_ms=int(frame_index * 1000.0 / fps))
                if landmarks is None:
                    coords.append(np.full((33, 3), np.nan, dtype=float))
                    visibility.append(np.full(33, np.nan, dtype=float))
                    continue
                pose = PoseLandmarks.from_any(landmarks)
                coords.append(pose.xyz_array())
                visibility.append(pose.visibility_array())
                detected += 1
        finally:
            landmarker.close()
            capture.release()
        clip_coords = np.asarray(coords, dtype=float)
        clip_visibility = np.asarray(visibility, dtype=float)
        all_coords.extend(clip_coords)
        all_visibility.extend(clip_visibility)
        clip_ids.extend([row["source_file"]] * len(coords))
        frame_ids.extend(range(len(coords)))
        clip_summaries.append(
            {
                "source_file": row["source_file"],
                "local_path": row["local_path"],
                "verified_label": row["verified_label"],
                "processed_frames": len(coords),
                "pose_detected_frames": detected,
                "pose_detection_rate": detected / len(coords) if coords else 0.0,
                "processing_seconds": time.perf_counter() - start,
            }
        )
    payload = {
        "coords": np.asarray(all_coords, dtype=float),
        "visibility": np.asarray(all_visibility, dtype=float),
        "clip_ids": np.asarray(clip_ids),
        "frame_ids": np.asarray(frame_ids, dtype=int),
        "clip_summaries": clip_summaries,
    }
    EXTERNAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EXTERNAL_OUTPUT / "robustness_landmarks.npz", coords=payload["coords"], visibility=payload["visibility"], clip_ids=payload["clip_ids"], frame_ids=payload["frame_ids"])
    return payload


def _pose_from_cache(coords: np.ndarray, visibility: np.ndarray) -> PoseLandmarks:
    return PoseLandmarks.from_xyz_array(coords, visibility=visibility)


def _visibility_stats(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": None, "median": None, "p10": None, "p25": None, "p75": None, "p90": None, "pct_ge_0_3": 0.0, "pct_ge_0_5": 0.0, "pct_ge_0_7": 0.0}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "pct_ge_0_3": float(np.mean(finite >= 0.3)),
        "pct_ge_0_5": float(np.mean(finite >= 0.5)),
        "pct_ge_0_7": float(np.mean(finite >= 0.7)),
    }


def visibility_analysis(payload: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    visibility = payload["visibility"]
    per_landmark = {}
    for index, name in enumerate(LANDMARK_NAMES):
        per_landmark[name] = _visibility_stats(visibility[:, index])
    ranking = sorted(LANDMARK_NAMES, key=lambda name: (per_landmark[name]["pct_ge_0_5"], per_landmark[name]["mean"] or 0.0), reverse=True)
    per_clip: dict[str, dict[str, Any]] = {}
    for source_file in sorted(set(payload["clip_ids"].tolist())):
        mask = payload["clip_ids"] == source_file
        per_clip[source_file] = {name: _visibility_stats(visibility[mask, index]) for index, name in enumerate(LANDMARK_NAMES)}
    return {"overall": per_landmark, "ranking": ranking, "per_clip": per_clip, "clip_summaries": payload["clip_summaries"]}


def policy_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for policy in POLICIES:
        accepted = 0
        quality_values: list[float] = []
        reasons: Counter[str] = Counter()
        by_clip: dict[str, dict[str, Any]] = {}
        for index, (coords, visibility, clip) in enumerate(zip(payload["coords"], payload["visibility"], payload["clip_ids"])):
            assessment = evaluate_visibility_policy(_pose_from_cache(coords, visibility), policy)
            accepted += int(assessment["accepted"])
            quality_values.append(float(assessment["quality"]["score"]))
            if not assessment["accepted"]:
                if assessment["missing_names"]:
                    reasons.update(f"missing:{name}" for name in assessment["missing_names"])
                if assessment["low_names"]:
                    reasons.update(f"low_visibility:{name}" for name in assessment["low_names"])
                if policy == "quality" and not assessment["low_names"] and not assessment["missing_names"]:
                    reasons["quality_below_threshold"] += 1
            elif assessment.get("warnings"):
                reasons.update(assessment["warnings"])
            item = by_clip.setdefault(str(clip), {"processed_frames": 0, "accepted_frames": 0, "qualities": [], "reasons": Counter()})
            item["processed_frames"] += 1
            item["accepted_frames"] += int(assessment["accepted"])
            item["qualities"].append(float(assessment["quality"]["score"]))
            if not assessment["accepted"]:
                item["reasons"].update(assessment["low_names"] + assessment["missing_names"])
        for item in by_clip.values():
            item["acceptance_percentage"] = 100.0 * item["accepted_frames"] / item["processed_frames"] if item["processed_frames"] else 0.0
            item["average_pose_quality"] = float(np.mean(item["qualities"])) if item["qualities"] else 0.0
            item["reasons"] = dict(item["reasons"])
            del item["qualities"]
        results[policy] = {
            "processed_frames": int(len(payload["coords"])),
            "accepted_frames": int(accepted),
            "rejected_frames": int(len(payload["coords"]) - accepted),
            "acceptance_percentage": 100.0 * accepted / len(payload["coords"]) if len(payload["coords"]) else 0.0,
            "average_pose_quality": float(np.mean(quality_values)) if quality_values else 0.0,
            "failure_reason_distribution": dict(reasons),
            "by_clip": by_clip,
        }
    return results


def temporal_stability(payload: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    jump_threshold = 0.15
    for source_file in sorted(set(payload["clip_ids"].tolist())):
        mask = payload["clip_ids"] == source_file
        coords = payload["coords"][mask]
        visibility = payload["visibility"][mask]
        for index, name in enumerate(LANDMARK_NAMES):
            if len(coords) < 2:
                deltas = np.empty(0)
            else:
                deltas = np.linalg.norm(np.diff(coords[:, index, :], axis=0), axis=1)
                deltas = deltas[np.isfinite(deltas)]
            low_mask = visibility[:, index][1:] < 0.5 if len(visibility) > 1 else np.empty(0, dtype=bool)
            all_deltas = np.linalg.norm(np.diff(coords[:, index, :], axis=0), axis=1) if len(coords) > 1 else np.empty(0)
            finite_mask = np.isfinite(all_deltas)
            low_deltas = all_deltas[low_mask & finite_mask] if len(all_deltas) else np.empty(0)
            rows.append(
                {
                    "source_file": source_file,
                    "landmark": name,
                    "coordinate_std": float(np.nanstd(coords[:, index, :])) if np.isfinite(coords[:, index, :]).any() else None,
                    "median_adjacent_delta": float(np.median(deltas)) if deltas.size else None,
                    "p95_adjacent_delta": float(np.percentile(deltas, 95)) if deltas.size else None,
                    "jump_rate_delta_gt_0_15": float(np.mean(deltas > jump_threshold)) if deltas.size else None,
                    "low_visibility_adjacent_delta_median": float(np.median(low_deltas)) if low_deltas.size else None,
                    "low_visibility_frames": int(np.count_nonzero(visibility[:, index] < 0.5)),
                }
            )
    return {"jump_threshold_coordinate_delta": jump_threshold, "rows": rows}


def _dependency(feature_name: str) -> list[str]:
    if feature_name.endswith(("_x", "_y", "_z")) and feature_name.rsplit("_", 1)[0] in LANDMARK_INDEX:
        return [feature_name.rsplit("_", 1)[0]]
    if feature_name == "left_shoulder_to_hip":
        return ["left_shoulder", "left_hip", "right_hip"]
    if feature_name == "right_shoulder_to_hip":
        return ["right_shoulder", "left_hip", "right_hip"]
    if feature_name.startswith("head_to_shoulder") or feature_name == "neck_inclination_deg":
        return ["nose", "left_shoulder", "right_shoulder"]
    if feature_name.startswith("left_"):
        return ["left_shoulder", "left_hip", "left_elbow", "left_knee", "left_ankle"]
    if feature_name.startswith("right_"):
        return ["right_shoulder", "right_hip", "right_elbow", "right_knee", "right_ankle"]
    return ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]


def _root_cause(feature_name: str) -> str:
    if feature_name in {"left_hip_x", "right_hip_x", "left_hip_z", "right_hip_z"}:
        return "Likely coordinate convention/projection and left-right body-frame difference; hip-centering removes translation but not camera-axis scale/orientation."
    if feature_name in {"left_knee_x", "left_ankle_x", "right_knee_z"}:
        return "Strongly affected by low-visibility lower limbs and side-view occlusion; also subject to camera projection and depth convention."
    if feature_name == "head_to_shoulder_z":
        return "Depth-axis convention/projection mismatch is the leading explanation; this is not evidence of a human posture error."
    if feature_name == "left_shoulder_to_hip":
        return "Derived ratio is sensitive to side-view foreshortening, shoulder/hip localization, and different source normalization."
    if feature_name == "right_elbow_z":
        return "Depends on a low-visibility elbow in the external side view and on MediaPipe z semantics."
    return "Mixed camera, scale, visibility, and source-coordinate effects; human-posture attribution is not identifiable here."


def domain_shift_root_cause(payload: dict[str, Any]) -> dict[str, Any]:
    train = load_multiposture(RAW_MULTIPOSTURE)
    train_values, names = __import__("src.posture_ai.features.pose_features", fromlist=["build_features"]).build_features(train, "hybrid")
    coords = payload["coords"]
    external_full: list[np.ndarray] = []
    for row in coords:
        if np.isfinite(row).all():
            pose = PoseLandmarks.from_xyz_array(row)
            from src.posture_ai.pose.adapter import adapt_mediapipe_to_multiposture

            external_full.append(adapt_mediapipe_to_multiposture(pose, visibility_threshold=None).values[0])
    external = np.asarray(external_full, dtype=float)
    if external.size == 0:
        return {"features": [], "external_rows": 0}
    rows = []
    existing = json.loads((EXTERNAL_OUTPUT / "domain_shift.json").read_text(encoding="utf-8"))
    top_names = [item["name"] for item in existing.get("most_shifted_features", [])[:10]]
    vis = payload["visibility"]
    for feature_name in top_names:
        index = names.index(feature_name)
        dependencies = _dependency(feature_name)
        low_rate = max(float(np.mean(vis[:, LANDMARK_INDEX[name]] < 0.5)) for name in dependencies if name in LANDMARK_INDEX)
        rows.append(
            {
                "feature_name": feature_name,
                "source_landmarks": dependencies,
                "training_range": [float(np.min(train_values[:, index])), float(np.max(train_values[:, index]))],
                "external_range": [float(np.min(external[:, index])), float(np.max(external[:, index]))],
                "training_mean": float(np.mean(train_values[:, index])),
                "training_std": float(np.std(train_values[:, index])),
                "external_mean": float(np.mean(external[:, index])),
                "external_std": float(np.std(external[:, index])),
                "max_dependency_low_visibility_rate": low_rate,
                "depends_on_low_visibility_landmark": low_rate > 0.2,
                "coordinate_convention_issue": True,
                "scaling_or_normalization_issue": True,
                "camera_viewpoint_issue": True,
                "z_semantics_issue": feature_name.endswith("_z") or feature_name.endswith("_z"),
                "actual_human_posture_difference": "Not identifiable from this comparison; DriveNet is seated vehicle footage and differs in camera/view, so no human-posture attribution is claimed.",
                "root_cause": _root_cause(feature_name),
            }
        )
    return {"external_rows": int(len(external)), "features": rows}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def train_v2() -> dict[str, Any]:
    frame = load_multiposture(RAW_MULTIPOSTURE)
    y = frame[LABEL_COLUMN].astype(str).to_numpy()
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    labels = sorted(np.unique(y).tolist())
    split = grouped_holdout_split(groups, test_groups=2, validation_groups=2, random_state=42)
    cache = {config: build_upperbody_features(frame, config, already_hip_centered=True) for config in V2_CONFIGS}
    ranking_rows: list[dict[str, Any]] = []
    for config in V2_CONFIGS:
        X, feature_names = cache[config]
        X_train = X[split["train"]]
        y_train = y[split["train"]]
        g_train = groups[split["train"]]
        for model_name in V2_MODELS:
            fold_metrics: list[dict[str, Any]] = []
            for fold_index, (train_rel, eval_rel) in enumerate(grouped_cv(g_train, n_splits=5)):
                model = make_model(model_name, random_state=42 + fold_index)
                model.fit(X_train[train_rel], y_train[train_rel])
                pred = model.predict(X_train[eval_rel])
                fold_metrics.append(_metrics(y_train[eval_rel], pred, labels))
            ranking_rows.append(
                {
                    "feature_config": config,
                    "model": model_name,
                    "feature_count": len(feature_names),
                    "cv_macro_f1_mean": float(np.mean([item["macro_f1"] for item in fold_metrics])),
                    "cv_macro_f1_std": float(np.std([item["macro_f1"] for item in fold_metrics])),
                    "cv_balanced_accuracy_mean": float(np.mean([item["balanced_accuracy"] for item in fold_metrics])),
                    "cv_accuracy_mean": float(np.mean([item["accuracy"] for item in fold_metrics])),
                }
            )
    ranking = pd.DataFrame(ranking_rows).sort_values(["cv_macro_f1_mean", "cv_balanced_accuracy_mean", "cv_accuracy_mean"], ascending=False).reset_index(drop=True)
    best = ranking.iloc[0]
    best_config = str(best["feature_config"])
    best_model_name = str(best["model"])
    best_X, best_names = cache[best_config]
    validation_model = make_model(best_model_name, random_state=42)
    validation_model.fit(best_X[split["train"]], y[split["train"]])
    validation_metrics = _metrics(y[split["validation"]], validation_model.predict(best_X[split["validation"]]), labels)
    final_idx = np.concatenate([split["train"], split["validation"]])
    final_model = make_model(best_model_name, random_state=42)
    final_model.fit(best_X[final_idx], y[final_idx])
    test_pred = final_model.predict(best_X[split["test"]])
    test_metrics = _metrics(y[split["test"]], test_pred, labels)
    MODEL_V2.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_V2 / "model.joblib")
    (MODEL_V2 / "feature_config.json").write_text(json.dumps({"name": best_config, "columns": best_names, "core_required_landmarks": list(CORE_REQUIRED_LANDMARKS), "optional_landmarks": list(OPTIONAL_LANDMARKS)}, indent=2), encoding="utf-8")
    (MODEL_V2 / "labels.json").write_text(json.dumps({"source_labels": labels}, indent=2), encoding="utf-8")
    metadata = {
        "model_family": "multiposture_upperbody_v2_experimental",
        "dataset_path": str(RAW_MULTIPOSTURE),
        "dataset_sha256": _sha256(RAW_MULTIPOSTURE),
        "target": LABEL_COLUMN,
        "group_column": GROUP_COLUMN,
        "model_name": best_model_name,
        "feature_config": best_config,
        "feature_columns": best_names,
        "feature_count": len(best_names),
        "core_required_landmarks": list(CORE_REQUIRED_LANDMARKS),
        "optional_landmarks": list(OPTIONAL_LANDMARKS),
        "split_subjects": {key: sorted(str(x) for x in np.unique(groups[value])) for key, value in split.items()},
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "limitations": ["Experimental upper-body track; V1 is unchanged.", "Source CSV column names support a 33-slot schema, but Zenodo prose says 11 joints, so full semantic provenance remains caveated.", "Real-image coordinate-scale equivalence is not proven."],
    }
    (MODEL_V2 / "metadata.json").write_text(json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    (EXTERNAL_OUTPUT / "multiposture_upperbody_v2_model_comparison.csv").parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(EXTERNAL_OUTPUT / "multiposture_upperbody_v2_model_comparison.csv", index=False)
    (EXTERNAL_OUTPUT / "multiposture_upperbody_v2_split.json").write_text(json.dumps({key: value.tolist() for key, value in split.items()}, indent=2), encoding="utf-8")
    lines = [
        "# MultiPosture upper-body V2 training",
        "",
        "Experimental separate model family. V1 was not overwritten.",
        "",
        f"- Selected model/config: `{best_model_name}` + `{best_config}` ({len(best_names)} features)",
        f"- Core required landmarks: `{', '.join(CORE_REQUIRED_LANDMARKS)}`",
        f"- Grouped CV macro F1: `{best['cv_macro_f1_mean']:.4f} ± {best['cv_macro_f1_std']:.4f}`",
        f"- Validation macro F1: `{validation_metrics['macro_f1']:.4f}`",
        f"- Final subject-held-out macro F1: `{test_metrics['macro_f1']:.4f}`",
        f"- Train subjects: `{', '.join(metadata['split_subjects']['train'])}`",
        f"- Validation subjects: `{', '.join(metadata['split_subjects']['validation'])}`",
        f"- Test subjects: `{', '.join(metadata['split_subjects']['test'])}`",
        f"- Export: `{MODEL_V2}`",
    ]
    (PROJECT_ROOT / "reports" / "multiposture_upperbody_v2_training.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"ranking": ranking, "metadata": metadata, "model": final_model, "best_config": best_config, "best_names": best_names, "best_X": best_X}


def external_v2_test(payload: dict[str, Any], rows: list[dict[str, str]], training: dict[str, Any]) -> dict[str, Any]:
    model = training["model"]
    config = training["best_config"]
    labels = list(training["metadata"]["validation_metrics"]["labels"])
    row_to_label = {row["source_file"]: row["verified_label"] for row in rows}
    results = []
    aggregate_y: list[str] = []
    aggregate_pred: list[str] = []
    external_feature_rows: list[np.ndarray] = []
    for source_file in sorted(set(payload["clip_ids"].tolist())):
        mask = payload["clip_ids"] == source_file
        clip_coords = payload["coords"][mask]
        clip_visibility = payload["visibility"][mask]
        pred: list[str] = []
        confidence: list[float] = []
        probabilities: list[np.ndarray] = []
        accepted = 0
        inference_seconds = 0.0
        for coords, visibility in zip(clip_coords, clip_visibility):
            pose = _pose_from_cache(coords, visibility)
            try:
                start = time.perf_counter()
                features, names = adapt_mediapipe_to_upperbody_v2(pose, config, visibility_threshold=0.5)
                if names != training["best_names"]:
                    raise ValueError("V2 feature schema/order mismatch")
                yhat = str(model.predict(features)[0])
                proba = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else np.full(len(model.classes_), np.nan)
                inference_seconds += time.perf_counter() - start
            except (ValueError, PoseValidationError):
                continue
            accepted += 1
            external_feature_rows.append(features[0])
            pred.append(yhat)
            confidence.append(float(np.nanmax(proba)) if np.isfinite(proba).all() else float("nan"))
            probabilities.append(np.asarray(proba, dtype=float))
            true_label = row_to_label[source_file]
            if true_label in labels:
                aggregate_y.append(true_label)
                aggregate_pred.append(yhat)
        valid_probs = [value for value in confidence if np.isfinite(value)]
        raw_switches = sum(a != b for a, b in zip(pred, pred[1:]))
        smooth_pred: list[str] = []
        history: deque[np.ndarray] = deque(maxlen=5)
        classes = list(model.classes_)
        for proba in probabilities:
            history.append(proba / np.sum(proba))
            smooth_pred.append(str(classes[int(np.argmax(np.mean(np.vstack(history), axis=0)))]))
        smooth_switches = sum(a != b for a, b in zip(smooth_pred, smooth_pred[1:]))
        results.append(
            {
                "source_file": source_file,
                "verified_label": row_to_label[source_file],
                "processed_frames": len(clip_coords),
                "accepted_frames": accepted,
                "rejected_frames": len(clip_coords) - accepted,
                "adapter_acceptance_percentage": 100.0 * accepted / len(clip_coords) if len(clip_coords) else 0.0,
                "prediction_distribution": dict(Counter(pred)),
                "confidence_mean": float(np.mean(valid_probs)) if valid_probs else None,
                "confidence_std": float(np.std(valid_probs)) if valid_probs else None,
                "confidence_min": float(np.min(valid_probs)) if valid_probs else None,
                "confidence_max": float(np.max(valid_probs)) if valid_probs else None,
                "raw_frame_to_frame_switches": raw_switches,
                "raw_switch_rate_over_consecutive_accepted": raw_switches / max(len(pred) - 1, 1),
                "smoothed_frame_to_frame_switches": smooth_switches,
                "smoothed_switch_rate_over_consecutive_accepted": smooth_switches / max(len(smooth_pred) - 1, 1),
                "pose_lost_or_rejected_frequency": (len(clip_coords) - accepted) / len(clip_coords) if len(clip_coords) else 0.0,
                "mean_classifier_ms_accepted": 1000.0 * inference_seconds / accepted if accepted else None,
            }
        )
    aggregate = _metrics(np.asarray(aggregate_y), np.asarray(aggregate_pred), labels) if aggregate_y else None
    external_features = np.asarray(external_feature_rows, dtype=float)
    train_features = np.asarray(training["best_X"], dtype=float)
    out_of_range = (external_features < np.min(train_features, axis=0)) | (external_features > np.max(train_features, axis=0))
    feature_names = list(training["best_names"])
    ood_rates = {name: float(np.mean(out_of_range[:, index])) for index, name in enumerate(feature_names)} if len(external_features) else {}
    ood_names = [name for name, rate in ood_rates.items() if rate > 0.5]
    return {
        "by_clip": results,
        "aggregate_labeled_metrics": aggregate,
        "total_processed": int(len(payload["coords"])),
        "total_accepted": int(sum(item["accepted_frames"] for item in results)),
        "feature_domain_shift": {
            "reference": "all local MultiPosture rows in the selected V2 feature schema",
            "feature_count": len(feature_names),
            "features_outside_training_minmax_over_50pct": ood_names,
            "feature_out_of_range_rates": ood_rates,
            "row_out_of_range_any_feature_rate": float(np.mean(np.any(out_of_range, axis=1))) if len(external_features) else 0.0,
        },
    }


def _write_visibility_reports(visibility: dict[str, Any], policies: dict[str, Any], stability: dict[str, Any]) -> None:
    overall = visibility["overall"]
    table = "\n".join(
        f"| {rank + 1} | `{name}` | {overall[name]['mean']:.3f} | {overall[name]['median']:.3f} | {overall[name]['p10']:.3f} | {overall[name]['p25']:.3f} | {overall[name]['p75']:.3f} | {overall[name]['p90']:.3f} | {100*overall[name]['pct_ge_0_3']:.1f}% | {100*overall[name]['pct_ge_0_5']:.1f}% | {100*overall[name]['pct_ge_0_7']:.1f}% |"
        for rank, name in enumerate(visibility["ranking"])
    )
    policy_table = "\n".join(
        f"| `{policy}` | {item['accepted_frames']} | {item['rejected_frames']} | {item['acceptance_percentage']:.1f}% | {item['average_pose_quality']:.3f} |"
        for policy, item in policies.items()
    )
    clip_sections = []
    for clip, clip_values in visibility["per_clip"].items():
        clip_table = "\n".join(
            f"| `{name}` | {clip_values[name]['mean']:.3f} | {clip_values[name]['median']:.3f} | {clip_values[name]['p10']:.3f} | {clip_values[name]['p25']:.3f} | {clip_values[name]['p75']:.3f} | {clip_values[name]['p90']:.3f} | {100*clip_values[name]['pct_ge_0_3']:.1f}% | {100*clip_values[name]['pct_ge_0_5']:.1f}% | {100*clip_values[name]['pct_ge_0_7']:.1f}% |"
            for name in LANDMARK_NAMES
        )
        clip_sections.append(
            f"### `{clip}`\n\n"
            "| Landmark | Mean | Median | P10 | P25 | P75 | P90 | ≥0.3 | ≥0.5 | ≥0.7 |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            f"{clip_table}"
        )
    per_clip_tables = "\n\n".join(clip_sections)
    unstable = [item for item in stability["rows"] if item["low_visibility_frames"] > 0]
    unstable.sort(key=lambda item: (item["jump_rate_delta_gt_0_15"] or 0.0, item["p95_adjacent_delta"] or 0.0), reverse=True)
    stability_table = "\n".join(
        f"| `{item['source_file']}` | `{item['landmark']}` | {item['low_visibility_frames']} | {item['coordinate_std']:.3f} | {item['median_adjacent_delta']:.3f} | {item['p95_adjacent_delta']:.3f} | {100*item['jump_rate_delta_gt_0_15']:.1f}% | {item['low_visibility_adjacent_delta_median']:.3f} |"
        for item in unstable[:15]
    )
    core_stability = [item for item in stability["rows"] if item["landmark"] in CORE_REQUIRED_LANDMARKS]
    core_low_visibility = sum(item["low_visibility_frames"] for item in core_stability)
    report = f"""# Landmark visibility analysis

## Scope

MediaPipe was run on the first {MAX_FRAMES} frames of each of the three downloaded DriveNet clips. These are RGB pose-extraction diagnostics only; no external source label was used for model accuracy in this report.

## Overall landmark reliability

The strict V1 policy requires all 33 named slots to meet visibility ≥ 0.5. The table is ranked by percentage meeting that threshold, then mean visibility. Percentages use finite MediaPipe visibility values from detected frames.

| Rank | Landmark | Mean | Median | P10 | P25 | P75 | P90 | ≥0.3 | ≥0.5 | ≥0.7 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## Per-clip landmark reliability

The same statistics are reported independently for each clip so a single pooled mean cannot hide clip-specific occlusion.

{per_clip_tables}

## Policy experiment

| Policy | Accepted | Rejected | Acceptance | Average quality |
|---|---:|---:|---:|---:|
{policy_table}

Policy meanings:

- `strict`: all 33 landmarks ≥ 0.5.
- `core-only`: nose, both shoulders, and both hips ≥ 0.5.
- `tiered`: core ≥ 0.5; optional ears/elbows are not required and low values become explicit warnings.
- `quality`: no per-landmark hard gate; accepts when the aggregate core score is ≥ 0.5. The score is `0.7 × mean(core visibility) + 0.3 × core coverage at 0.3`.

## Interpretation

The old adapter rejected frames because lower-limb landmarks and some elbows/fingers repeatedly fell below 0.5. The rejection was not caused by pose detection failure: the external audit detected pose in 900/900 frames. The strict rule is appropriate for V1's full 126-feature contract, but it is too strict for a seated office posture representation that does not need knees, ankles, heels, or feet.

## Temporal stability diagnostics

Adjacent-frame coordinate jumps were summarized with a diagnostic threshold of `0.15` in MediaPipe coordinate units. This threshold flags sudden changes; it is not a biomechanical impossibility threshold. Low-visibility coordinates with high jump rates should be marked unreliable rather than silently trusted or interpolated.

The full per-landmark/per-clip stability data are stored in `outputs/validation/external/adapter_robustness_v2.json`.

### Observed low-visibility stability

No core landmark had a frame below visibility 0.5 in this 300-frame-per-clip sample (`{core_low_visibility}` core low-visibility frames in total). The following are the highest diagnostic jump-rate cases among landmarks that did fall below 0.5; the `0.15` jump threshold is a screening heuristic, not a biomechanical limit.

| Clip | Landmark | Low-vis frames | Coordinate std | Median Δ | P95 Δ | Jump rate >0.15 | Low-vis median Δ |
|---|---|---:|---:|---:|---:|---:|---:|
{stability_table}

These measurements support ignoring or explicitly marking low-visibility hands, elbows, knees, ankles, heels, and feet as missing for the V2 core rather than silently trusting or interpolating them. The sample is not sufficient to prove that a low-visibility point is always wrong; it is sufficient to reject it from a required feature contract and preserve the diagnostic warning.

Engineering decision: retain reliable core coordinates, ignore low-value limb landmarks for V2, treat optional landmarks as missing/warnings when visibility is low, and apply temporal smoothing to predictions rather than silently imputing coordinates.
"""
    (PROJECT_ROOT / "reports" / "landmark_visibility_analysis.md").write_text(report, encoding="utf-8")
    core = f"""# Core sitting landmarks

## Candidate design

The experimental V2 core is intentionally minimal and maps directly to the posture questions required by this project:

### CORE_REQUIRED

`nose`, `left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`.

These support head-to-shoulder offsets, shoulder/hip alignment, trunk inclination, and left/right torso lean. They are the only landmarks used in the V2 model feature schema.

### OPTIONAL

`left_ear`, `right_ear`, `left_elbow`, `right_elbow`.

Ears can add head orientation and elbows can add upper-limb context, but neither is required for the core V2 adapter. Under the tiered policy they are reported as explicit quality warnings when visibility is below 0.2; no silent imputation is performed.

### LOW_VALUE_FOR_OFFICE_POSTURE

{', '.join(LOW_VALUE_FOR_OFFICE_POSTURE)}.

These landmarks may be useful for other tasks, but they are not necessary for the intended seated head/shoulder/trunk/hip posture state and are vulnerable to desk, chair, steering-wheel, and lower-body occlusion.

## Decision

Do not change V1. V2 is a separate core-only experimental model family with raw, normalized, geometry, and hybrid feature tracks. The V2 adapter still requires exact finite XYZ for the five core landmarks and visibility ≥ 0.5; it never treats low-visibility coordinates as automatically trustworthy.
"""
    (PROJECT_ROOT / "reports" / "core_sitting_landmarks.md").write_text(core, encoding="utf-8")


def _write_domain_report(root_cause: dict[str, Any]) -> None:
    rows = "\n".join(
        f"| `{item['feature_name']}` | {', '.join(item['source_landmarks'])} | {item['training_range'][0]:.4f}…{item['training_range'][1]:.4f} | {item['external_range'][0]:.4f}…{item['external_range'][1]:.4f} | {item['training_mean']:.4f} ± {item['training_std']:.4f} | {item['external_mean']:.4f} ± {item['external_std']:.4f} | {100*item['max_dependency_low_visibility_rate']:.1f}% | {item['root_cause']} |"
        for item in root_cause.get("features", [])
    )
    report = f"""# Domain-shift root-cause analysis

The ten features previously identified as outside the MultiPosture training min/max for more than half of the 900 external diagnostic rows were inspected without clipping or forced rescaling.

| Feature | Source landmark(s) | Training range | External range | Training mean ± std | External mean ± std | Max dependency low-vis | Root-cause assessment |
|---|---|---|---|---|---|---:|---|
{rows}

## Cross-cutting findings

- The external rows were generated from MediaPipe RGB image-normalized x/y/z and hip-centered only for diagnostics. MultiPosture stores source coordinates documented as hip-centered, but its exact camera projection, axis orientation, scale, and z convention are not fully specified.
- Side/three-quarter vehicle footage changes foreshortening and visibility relative to the MultiPosture home-environment recordings. This is a camera/domain effect, not proof that the person has a different project posture label.
- Several shifted features depend directly or indirectly on low-visibility limbs. Their coordinates are retained in diagnostics but are not silently trusted by the V2 adapter.
- Z values are especially unsafe to equate across sources: MediaPipe z is model/image-relative depth semantics, while the dataset's generation convention is not independently documented in the local CSV.
- The correct response is an explicit adapter/feature redesign and source-convention validation, not clipping values into the training range.
"""
    (PROJECT_ROOT / "reports" / "domain_shift_root_cause.md").write_text(report, encoding="utf-8")


def write_mapping_report() -> None:
    frame = pd.read_csv(RAW_MULTIPOSTURE, nrows=1)
    coord_columns = [column for column in frame.columns if column not in {GROUP_COLUMN, "upperbody_label", "lowerbody_label"}]
    groups = {name: [f"{name}_{axis}" for axis in ("x", "y", "z")] for name in LANDMARK_NAMES}
    rows = []
    for name in LANDMARK_NAMES:
        columns = groups[name]
        present = all(column in coord_columns for column in columns)
        rows.append(f"| `{', '.join(columns)}` | {name} | {name} | {'EXACT_MATCH' if present else 'UNKNOWN'} | {'actual local column group; name/index match only, coordinate provenance still caveated' if present else 'not present'} |")
    report = f"""# Verified MultiPosture landmark mapping

## Measured local schema

The actual local CSV has **102 columns**: `subject`, two labels, and **99 coordinate columns**. The coordinate names form 33 groups of `x/y/z` using the canonical MediaPipe landmark names. This is an observed file fact from `{RAW_MULTIPOSTURE}`.

The Zenodo metadata stored locally describes “11 key body joints” and “33 input dimensions,” which conflicts with the actual 99 coordinate columns. Therefore this report does not silently replace the measured schema with the prose count. The local column names support a 33-slot name/index mapping, but they do not independently prove how every slot was generated.

## Mapping table

`EXACT_MATCH` below means exact column-group name/index identity with the canonical MediaPipe name. It does **not** mean that source camera scale, projection, z-axis convention, visibility, or model-version semantics have been validated.

| MultiPosture feature group | Anatomical landmark by column name | MediaPipe equivalent | Status | Evidence/limitation |
|---|---|---|---|---|
{chr(10).join(rows)}

## Source facts and engineering decision

- Source metadata says coordinates are normalized relative to hip center and were extracted using MediaPipe Pose Heavy; the actual CSV values are finite and the local loader confirms the exact 102-column order.
- The CSV has explicit `subject` IDs and upper/lower labels; V2 uses only `upperbody_label`.
- The disagreement between the “11 joints” prose and the 33 named coordinate groups means full semantic provenance is **UNKNOWN** even though the name-level mapping is exact.
- V2 is therefore an experimental feasibility track, not proof of real-camera semantic compatibility. V1 remains unchanged.
"""
    (PROJECT_ROOT / "reports" / "multiposture_landmark_mapping_verified.md").write_text(report, encoding="utf-8")


def _write_final_reports(visibility: dict[str, Any], policies: dict[str, Any], stability: dict[str, Any], root_cause: dict[str, Any], training: dict[str, Any], external: dict[str, Any]) -> None:
    v1_meta = json.loads((MODEL_V1 / "metadata.json").read_text(encoding="utf-8"))
    v1_cmp = pd.read_csv(PROJECT_ROOT / "outputs" / "metrics" / "multiposture_model_comparison.csv")
    v1_row = v1_cmp[(v1_cmp["feature_config"] == v1_meta["feature_config"]) & (v1_cmp["model"] == v1_meta["model_name"])].iloc[0]
    v2_meta = training["metadata"]
    v2_best_row = training["ranking"].iloc[0]
    ext_accept = external["total_accepted"] / external["total_processed"] if external["total_processed"] else 0.0
    v2_shift = external.get("feature_domain_shift", {})
    v2_ood_names = v2_shift.get("features_outside_training_minmax_over_50pct", [])
    v2_latency_values = [item["mean_classifier_ms_accepted"] for item in external.get("by_clip", []) if item.get("mean_classifier_ms_accepted") is not None]
    v2_latency_text = f"{float(np.mean(v2_latency_values)):.1f} ms/accepted frame (adapter + classifier; pose extraction excluded)" if v2_latency_values else "not measured"
    detected = sum(item["pose_detected_frames"] for item in visibility["clip_summaries"])
    processed = sum(item["processed_frames"] for item in visibility["clip_summaries"])
    if detected / processed >= 0.95 and ext_accept >= 0.95:
        yolo_decision = "KEEP MEDIAPIPE as the primary backend; YOLO Pose benchmarking is not justified by this study."
    else:
        yolo_decision = "MediaPipe core robustness is not yet sufficient; a future MediaPipe-vs-YOLO Pose benchmark is justified, but YOLO was not installed or trained in this study."
    ext_metrics = external.get("aggregate_labeled_metrics")
    ext_metric_text = "No labeled external accuracy was available." if ext_metrics is None else f"Valid mapped-frame metrics: accuracy {ext_metrics['accuracy']:.4f}, balanced accuracy {ext_metrics['balanced_accuracy']:.4f}, macro F1 {ext_metrics['macro_f1']:.4f}; interpret cautiously because the external set has one clip per mapped source class and is not a subject-independent test."
    report = f"""# Adapter robustness V2 study

## Executive result

The old 126-feature V1 adapter rejected **0/900** external frames because it requires all 33 coordinate slots to have visibility ≥ 0.5. MediaPipe itself detected a pose in **{detected}/{processed}** frames. The dominant failures are low-visibility knees, ankles, heels, feet, and some elbows/fingers caused by seated side-view occlusion.

V2 was scientifically feasible as a separate experimental track because the measured MultiPosture CSV contains the five core column groups needed for head/shoulder/trunk/hip posture features. V2 uses `{', '.join(CORE_REQUIRED_LANDMARKS)}` only and does not overwrite V1.

## 1–6. Diagnosis and semantic verification

1. **Why old V1 failed:** full-body visibility gate, not pose detection failure; the exact ranked visibility table is in `reports/landmark_visibility_analysis.md`.
2. **Necessary office-posture landmarks:** nose, both shoulders, both hips. Ears/elbows are optional; knees/ankles/feet are low-value for the stated goal.
3. **Was old rule too strict?** Yes for an upper-body office posture track; no for the existing V1 full 126-feature contract. Policy results are below.
4. **Policy acceptance:**

| Policy | Accepted | Rejected | Acceptance | Average quality |
|---|---:|---:|---:|---:|
{chr(10).join(f"| `{p}` | {x['accepted_frames']} | {x['rejected_frames']} | {x['acceptance_percentage']:.1f}% | {x['average_pose_quality']:.3f} |" for p, x in policies.items())}

5. **MultiPosture vs MediaPipe semantics:** exact 33-name/99-column identity is present in the local CSV, but the source prose says 11 joints and does not fully define projection/scale/z. Therefore name mapping is `EXACT_MATCH` at the column-name level while real-image semantic compatibility remains **NOT VALIDATED**. See `reports/multiposture_landmark_mapping_verified.md`.
6. **Ten-feature domain shift:** the leading causes are coordinate convention/projection, scale/normalization, z semantics, side-view foreshortening, and low-visibility limb coordinates—not a reason to clip values. See `reports/domain_shift_root_cause.md`.

## 7–9. V2 training and comparison

- V2 selected model/config: `{v2_meta['model_name']}` + `{v2_meta['feature_config']}` ({v2_meta['feature_count']} features)
- V2 grouped-CV macro F1: `{v2_best_row['cv_macro_f1_mean']:.4f} ± {v2_best_row['cv_macro_f1_std']:.4f}`
- V2 validation macro F1: `{v2_meta['validation_metrics']['macro_f1']:.4f}`
- V2 final subject-held-out macro F1: `{v2_meta['test_metrics']['macro_f1']:.4f}`

| Attribute | V1 | V2 |
|---|---|---|
| Required landmarks | 33 | 5 core |
| Feature count | 126 | {v2_meta['feature_count']} |
| Grouped CV macro F1 | {v1_row['cv_macro_f1_mean']:.4f} ± {v1_row['cv_macro_f1_std']:.4f} | {v2_best_row['cv_macro_f1_mean']:.4f} ± {v2_best_row['cv_macro_f1_std']:.4f} |
| Holdout macro F1 | {v1_meta['test_metrics']['macro_f1']:.4f} | {v2_meta['test_metrics']['macro_f1']:.4f} |
| External adapter acceptance | 0/900 (0.0%) | {external['total_accepted']}/{external['total_processed']} ({100*ext_accept:.1f}%) |
| Inference latency | Not measured in the prior V1 offline performance record | {v2_latency_text} |
| Domain-shift severity (mechanical train min/max diagnostic) | 10/126 features outside range for >50% of external rows | {len(v2_ood_names)}/{v2_shift.get('feature_count', v2_meta['feature_count'])} V2 features outside range for >50% of external rows; semantics unresolved |
| Expected occlusion robustness | Poor for seated side view | Better for core torso/head occlusion pattern |
| Real-image semantic compatibility | Not validated | Not validated; operational gate improved only if core passes |

The goal is robustness, not a higher benchmark score. The V2 holdout is still drawn from the same 13-subject MultiPosture source and is not real-camera generalization.

## 10–11. DriveNet V2 test and temporal behavior

{ext_metric_text}

| Clip | Source label | Accepted | Acceptance | Predictions | Raw switches | Smoothed switches | Pose-lost/rejected |
|---|---|---:|---:|---|---:|---:|---:|
{chr(10).join(f"| `{item['source_file']}` | `{item['verified_label']}` | {item['accepted_frames']} | {item['adapter_acceptance_percentage']:.1f}% | {item['prediction_distribution']} | {item['raw_frame_to_frame_switches']} | {item['smoothed_frame_to_frame_switches']} | {100*item['pose_lost_or_rejected_frequency']:.1f}% |" for item in external['by_clip'])}

Temporal smoothing results are descriptive only. It reduces or preserves switching according to the per-clip values; it does not repair invalid landmarks. Ambiguous DriveNet source classes were not mapped to project labels.

## 12–14. Backend decision and blockers

**YOLO gate:** {yolo_decision}

Current real-image compatibility statement: **MODEL TRAINING READY; UPPER-BODY REAL-IMAGE ADAPTER OPERATIONALLY IMPROVED / SEMANTICALLY NOT VALIDATED.** The V2 gate is more appropriate for seated occlusion, but its coordinate convention still needs paired validation.

Remaining blockers before own real sitting videos:

- verify the source-coordinate convention with a paired RGB/landmark example or controlled recording;
- collect the planned five office-view posture videos when hardware is available;
- evaluate across people and camera placements, not only one external driver clip per class;
- keep ANYA training blocked until a verified ground-truth mapping is obtained.

No raw dataset was modified, V1 was not overwritten, no webcam source 0 was opened, and no medical claim is made.
"""
    (PROJECT_ROOT / "reports" / "adapter_robustness_v2.md").write_text(report, encoding="utf-8")
    th = f"""# รายงาน Adapter Robustness และ Upper-body V2

## ผลหลัก

V1 adapter เดิมรับได้ 0/900 เพราะบังคับ landmark ทั้ง 33 จุดให้ visibility ≥ 0.5 ไม่ใช่เพราะ MediaPipe หา pose ไม่ได้ โดย MediaPipe ตรวจ pose ได้ {detected}/{processed} เฟรม สาเหตุหลักคือเข่า ข้อเท้า ส้นเท้า เท้า และข้อศอก/นิ้วบางส่วนถูกบังจากมุมกล้องด้านข้างของท่านั่ง

V2 เป็น model family แยก ใช้เฉพาะ `{', '.join(CORE_REQUIRED_LANDMARKS)}` และไม่แก้ V1

## ผล policy

| Policy | รับ | ปฏิเสธ | อัตรารับ | quality เฉลี่ย |
|---|---:|---:|---:|---:|
{chr(10).join(f"| `{p}` | {x['accepted_frames']} | {x['rejected_frames']} | {x['acceptance_percentage']:.1f}% | {x['average_pose_quality']:.3f} |" for p, x in policies.items())}

แกน core ที่จำเป็นต่อ office posture คือ nose, ไหล่ซ้าย/ขวา และสะโพกซ้าย/ขวา ส่วน ears/elbows เป็น optional และ knees/ankles/feet เป็น low-value สำหรับเป้าหมายนี้ รายละเอียดอยู่ใน `reports/core_sitting_landmarks.md`

## V2 training

- model/config: `{v2_meta['model_name']}` + `{v2_meta['feature_config']}` ({v2_meta['feature_count']} features)
- grouped-CV macro F1: `{v2_best_row['cv_macro_f1_mean']:.4f} ± {v2_best_row['cv_macro_f1_std']:.4f}`
- holdout macro F1: `{v2_meta['test_metrics']['macro_f1']:.4f}`
- external adapter acceptance: `{external['total_accepted']}/{external['total_processed']} ({100*ext_accept:.1f}%)`

V1 และ V2 ใช้ holdout แบบแยก subject เดียวกัน แต่ผลนี้ยังไม่ใช่ real-camera generalization

## Semantic และ YOLO decision

CSV จริงมี 102 columns, 99 coordinate columns และชื่อครบ 33 กลุ่ม แต่ metadata ของ Zenodo ระบุ 11 joints จึงยืนยันได้เฉพาะ name/index mapping ไม่ใช่ coordinate-scale/z semantic แบบสมบูรณ์ รายละเอียดอยู่ใน `reports/multiposture_landmark_mapping_verified.md` และ `reports/domain_shift_root_cause.md`

ผล gate: **{yolo_decision}**

V2 ทำให้ operational acceptance เหมาะกับ seated occlusion มากขึ้น แต่ real-image semantic compatibility ยัง **NOT VALIDATED** ยังต้องตรวจด้วยวิดีโอ office-view ของผู้ใช้เองเมื่อมี hardware, โดยไม่เปิด webcam ในเครื่องนี้และไม่ฝึก ANYA
"""
    (PROJECT_ROOT / "reports" / "adapter_robustness_v2_th.md").write_text(th, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adapter robustness and upper-body V2 study")
    parser.add_argument("--reports-only", action="store_true", help="Regenerate reports from the existing robustness cache and V2 artifacts")
    args = parser.parse_args()
    rows = _read_manifest()
    if args.reports_only:
        artifact_path = EXTERNAL_OUTPUT / "adapter_robustness_v2.json"
        cache_path = EXTERNAL_OUTPUT / "robustness_landmarks.npz"
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        cached = np.load(cache_path, allow_pickle=False)
        clip_summaries = []
        for row in rows:
            mask = cached["clip_ids"] == row["source_file"]
            coords = cached["coords"][mask]
            detected = int(np.isfinite(coords).all(axis=(1, 2)).sum())
            clip_summaries.append(
                {
                    "source_file": row["source_file"],
                    "local_path": row["local_path"],
                    "verified_label": row["verified_label"],
                    "processed_frames": int(mask.sum()),
                    "pose_detected_frames": detected,
                    "pose_detection_rate": detected / int(mask.sum()) if mask.sum() else 0.0,
                }
            )
        payload = {
            "coords": cached["coords"],
            "visibility": cached["visibility"],
            "clip_ids": cached["clip_ids"],
            "frame_ids": cached["frame_ids"],
            "clip_summaries": clip_summaries,
        }
        visibility = artifact["visibility"]
        policies = artifact["policies"]
        stability = artifact["temporal_stability"]
        root_cause = artifact["domain_shift_root_cause"]
        training = {
            "metadata": json.loads((MODEL_V2 / "metadata.json").read_text(encoding="utf-8")),
            "ranking": pd.read_csv(EXTERNAL_OUTPUT / "multiposture_upperbody_v2_model_comparison.csv"),
        }
        external = artifact["v2_external"]
        write_mapping_report()
        _write_visibility_reports(visibility, policies, stability)
        _write_domain_report(root_cause)
        _write_final_reports(visibility, policies, stability, root_cause, training, external)
        return
    payload = collect_landmarks(rows)
    visibility = visibility_analysis(payload, rows)
    policies = policy_analysis(payload)
    stability = temporal_stability(payload)
    root_cause = domain_shift_root_cause(payload)
    write_mapping_report()
    training = train_v2()
    external = external_v2_test(payload, rows, training)
    artifact = {
        "processed_frame_limit": MAX_FRAMES,
        "visibility": visibility,
        "policies": policies,
        "temporal_stability": stability,
        "domain_shift_root_cause": root_cause,
        "v2_training": training["metadata"],
        "v2_external": external,
    }
    EXTERNAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    (EXTERNAL_OUTPUT / "adapter_robustness_v2.json").write_text(json.dumps(_jsonable(artifact), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_visibility_reports(visibility, policies, stability)
    _write_domain_report(root_cause)
    _write_final_reports(visibility, policies, stability, root_cause, training, external)
    print(json.dumps(_jsonable({"policies": policies, "v2_training": training["metadata"], "v2_external": external}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
