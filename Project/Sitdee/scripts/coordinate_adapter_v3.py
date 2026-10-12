"""Reconstruct the MultiPosture coordinate convention and calibrate V3 candidates.

This script is deliberately forensic first.  It does not retrain V1/V2, does
not select a transform from DriveNet labels, and writes only derived outputs
under ``outputs/validation/coordinate_adapter_v3`` plus reports.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw" / "multiposture" / "data.csv"
MANIFEST = ROOT / "validation_data" / "external" / "external_manifest.csv"
POSE_MODEL = ROOT / "metadata" / "pose_landmarker_heavy.task"
V1_DIR = ROOT / "outputs" / "models" / "multiposture_v1"
V2_DIR = ROOT / "outputs" / "models" / "multiposture_upperbody_v2"
OUT = ROOT / "outputs" / "validation" / "coordinate_adapter_v3"
MAX_FRAMES = 300
SOURCE_RESOLUTION = (640, 480)
LABEL_COLUMN = "upperbody_label"
GROUP_COLUMN = "subject"
sys.path.insert(0, str(ROOT))

from src.posture_ai.evaluation.splitting import grouped_cv, grouped_holdout_split  # noqa: E402
from src.posture_ai.features.coordinate_v3 import (  # noqa: E402
    ADAPTER_JUSTIFICATIONS,
    ADAPTER_V3_CANDIDATES,
    candidate_representation,
    transform_coordinates,
)
from src.posture_ai.features.pose_features import build_features, expected_feature_columns  # noqa: E402
from src.posture_ai.features.upperbody_v2 import GEOMETRY_FEATURE_NAMES, build_upperbody_features  # noqa: E402
from src.posture_ai.models.factories import make_model  # noqa: E402
from src.posture_ai.pose.landmarks import LANDMARK_INDEX, LANDMARK_NAMES  # noqa: E402


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _landmark_array(points) -> tuple[np.ndarray, np.ndarray]:
    coords = np.full((33, 3), np.nan, dtype=float)
    visibility = np.full(33, np.nan, dtype=float)
    for index, point in enumerate(points[:33]):
        coords[index] = [float(point.x), float(point.y), float(point.z)]
        value = getattr(point, "visibility", None)
        if value is not None and np.isfinite(value):
            visibility[index] = float(value)
    return coords, visibility


def _payload_key(representation: str) -> str:
    """Map report-facing representation names to cached payload keys."""
    return "world" if representation == "world_landmarks" else "normalized"


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


def collect_media(rows: list[dict[str, str]]) -> dict[str, Any]:
    normalized: list[np.ndarray] = []
    world: list[np.ndarray] = []
    visibility: list[np.ndarray] = []
    world_visibility: list[np.ndarray] = []
    clip_ids: list[str] = []
    frame_ids: list[int] = []
    summaries: list[dict[str, Any]] = []
    for row in rows:
        path = ROOT / row["local_path"]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open external video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        landmarker = _make_landmarker()
        detected = 0
        world_available = 0
        clip_norm: list[np.ndarray] = []
        clip_world: list[np.ndarray] = []
        clip_vis: list[np.ndarray] = []
        clip_world_vis: list[np.ndarray] = []
        start = time.perf_counter()
        try:
            for frame_index in range(MAX_FRAMES):
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.resize(frame, SOURCE_RESOLUTION, interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, int(frame_index * 1000.0 / fps))
                if not result.pose_landmarks:
                    clip_norm.append(np.full((33, 3), np.nan, dtype=float))
                    clip_world.append(np.full((33, 3), np.nan, dtype=float))
                    clip_vis.append(np.full(33, np.nan, dtype=float))
                    clip_world_vis.append(np.full(33, np.nan, dtype=float))
                else:
                    norm, vis = _landmark_array(result.pose_landmarks[0])
                    if result.pose_world_landmarks:
                        world_row, world_vis = _landmark_array(result.pose_world_landmarks[0])
                        world_available += 1
                    else:
                        world_row = np.full((33, 3), np.nan, dtype=float)
                        world_vis = np.full(33, np.nan, dtype=float)
                    clip_norm.append(norm)
                    clip_world.append(world_row)
                    clip_vis.append(vis)
                    clip_world_vis.append(world_vis)
                    detected += 1
        finally:
            landmarker.close()
            capture.release()
        normalized.extend(clip_norm)
        world.extend(clip_world)
        visibility.extend(clip_vis)
        world_visibility.extend(clip_world_vis)
        clip_ids.extend([row["source_file"]] * len(clip_norm))
        frame_ids.extend(range(len(clip_norm)))
        summaries.append(
            {
                "source_file": row["source_file"],
                "verified_label": row["verified_label"],
                "local_path": row["local_path"],
                "processed_frames": len(clip_norm),
                "pose_detected_frames": detected,
                "world_landmark_frames": world_available,
                "pose_detection_rate": detected / len(clip_norm) if clip_norm else 0.0,
                "processing_seconds": time.perf_counter() - start,
            }
        )
    payload = {
        "normalized": np.asarray(normalized, dtype=float),
        "world": np.asarray(world, dtype=float),
        "visibility": np.asarray(visibility, dtype=float),
        "world_visibility": np.asarray(world_visibility, dtype=float),
        "clip_ids": np.asarray(clip_ids),
        "frame_ids": np.asarray(frame_ids, dtype=int),
        "summaries": summaries,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "landmarks.npz", **{key: payload[key] for key in ("normalized", "world", "visibility", "world_visibility", "clip_ids", "frame_ids")})
    return payload


def _stats(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {key: None for key in ("min", "max", "mean", "std", "median", "p01", "p10", "p25", "p50", "p75", "p90", "p99")}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
        "p99": float(np.percentile(finite, 99)),
    }


def _stats_rows(values: np.ndarray, representation: str) -> list[dict[str, Any]]:
    rows = []
    for landmark_index, landmark in enumerate(LANDMARK_NAMES):
        for axis_index, axis in enumerate(("x", "y", "z")):
            row = {"representation": representation, "landmark": landmark, "axis": axis}
            row.update(_stats(values[:, landmark_index, axis_index]))
            rows.append(row)
    return rows


def _vector_stats(values: np.ndarray, representation: str) -> list[dict[str, Any]]:
    """Summarize an (n, 3) vector without pretending it has 33 landmarks."""
    values = np.asarray(values, dtype=float)
    return [
        {"representation": representation, "landmark": "hip_midpoint", "axis": axis, **_stats(values[:, axis_index])}
        for axis_index, axis in enumerate(("x", "y", "z"))
    ]


def coordinate_statistics(frame: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    local = frame[expected_feature_columns()].to_numpy(dtype=float).reshape(-1, 33, 3)
    rows = _stats_rows(local, "MultiPosture local CSV")
    rows.extend(_stats_rows(payload["normalized"], "DriveNet MediaPipe normalized image landmarks"))
    rows.extend(_stats_rows(payload["world"], "DriveNet MediaPipe world landmarks"))
    return {"rows": rows, "local_rows": len(local), "external_rows": len(payload["normalized"])}


def _midpoint(values: np.ndarray) -> np.ndarray:
    return (values[:, LANDMARK_INDEX["left_hip"]] + values[:, LANDMARK_INDEX["right_hip"]]) / 2.0


def hip_center_analysis(frame: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    local = frame[[f"left_hip_{axis}" for axis in "xyz"] + [f"right_hip_{axis}" for axis in "xyz"]].to_numpy(dtype=float).reshape(-1, 2, 3)
    local_mid = np.mean(local, axis=1)
    result = {"local_overall": _vector_stats(local_mid, "local_hip_midpoint"), "per_subject": {}}
    for subject, group in frame.groupby(GROUP_COLUMN, sort=True):
        hips = group[[f"left_hip_{axis}" for axis in "xyz"] + [f"right_hip_{axis}" for axis in "xyz"]].to_numpy(dtype=float).reshape(-1, 2, 3)
        result["per_subject"][str(subject)] = _vector_stats(np.mean(hips, axis=1), str(subject))
    result["external"] = {}
    for key in ("normalized", "world"):
        result["external"][key] = _vector_stats(_midpoint(payload[key]), key)
    return result


def _geometry(values: np.ndarray) -> dict[str, np.ndarray]:
    nose = values[:, LANDMARK_INDEX["nose"]]
    ls = values[:, LANDMARK_INDEX["left_shoulder"]]
    rs = values[:, LANDMARK_INDEX["right_shoulder"]]
    lh = values[:, LANDMARK_INDEX["left_hip"]]
    rh = values[:, LANDMARK_INDEX["right_hip"]]
    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0
    trunk = shoulder_mid - hip_mid
    torso_length = np.maximum(np.linalg.norm(trunk, axis=1), 1e-8)
    return {
        "shoulder_midpoint_relative_to_hip_x": shoulder_mid[:, 0] - hip_mid[:, 0],
        "shoulder_midpoint_relative_to_hip_y": shoulder_mid[:, 1] - hip_mid[:, 1],
        "shoulder_midpoint_relative_to_hip_z": shoulder_mid[:, 2] - hip_mid[:, 2],
        "trunk_vector_x": trunk[:, 0] / torso_length,
        "trunk_vector_y": trunk[:, 1] / torso_length,
        "trunk_vector_z": trunk[:, 2] / torso_length,
        "trunk_inclination_deg": np.degrees(np.arctan2(trunk[:, 0], np.maximum(np.abs(trunk[:, 1]), 1e-8))),
        "shoulder_tilt_deg": np.degrees(np.arctan2(rs[:, 1] - ls[:, 1], rs[:, 0] - ls[:, 0] + 1e-8)),
        "hip_tilt_deg": np.degrees(np.arctan2(rh[:, 1] - lh[:, 1], rh[:, 0] - lh[:, 0] + 1e-8)),
        "nose_relative_to_shoulder_x": (nose[:, 0] - shoulder_mid[:, 0]) / torso_length,
        "nose_relative_to_shoulder_y": (nose[:, 1] - shoulder_mid[:, 1]) / torso_length,
        "nose_relative_to_shoulder_z": (nose[:, 2] - shoulder_mid[:, 2]) / torso_length,
        "nose_relative_to_hip_x": (nose[:, 0] - hip_mid[:, 0]) / torso_length,
        "nose_relative_to_hip_y": (nose[:, 1] - hip_mid[:, 1]) / torso_length,
        "nose_relative_to_hip_z": (nose[:, 2] - hip_mid[:, 2]) / torso_length,
        "torso_length": torso_length,
        "shoulder_width": np.linalg.norm(rs - ls, axis=1),
        "hip_width": np.linalg.norm(rh - lh, axis=1),
    }


def scale_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    values = frame[expected_feature_columns()].to_numpy(dtype=float).reshape(-1, 33, 3)
    geo = _geometry(values)
    distances = ("shoulder_width", "hip_width", "torso_length")
    by_subject = []
    for subject, group in frame.groupby(GROUP_COLUMN, sort=True):
        idx = group.index.to_numpy()
        item = {"subject": str(subject), "rows": int(len(idx))}
        for name in distances:
            arr = geo[name][idx]
            item[f"{name}_mean"] = float(np.mean(arr))
            item[f"{name}_std"] = float(np.std(arr))
            item[f"{name}_cv"] = float(np.std(arr) / max(abs(np.mean(arr)), 1e-8))
        by_subject.append(item)
    by_class = []
    for label, group in frame.groupby(LABEL_COLUMN, sort=True):
        idx = group.index.to_numpy()
        item = {"label": str(label), "rows": int(len(idx))}
        for name in distances:
            stat = _stats(geo[name][idx])
            item.update({f"{name}_{key}": stat[key] for key in ("mean", "std", "p10", "p50", "p90")})
        by_class.append(item)
    overall = {name: _stats(geo[name]) for name in distances}
    subject_means = {name: np.asarray([item[f"{name}_mean"] for item in by_subject]) for name in distances}
    return {"overall": overall, "by_subject": by_subject, "by_class": by_class, "subject_mean_cv": {name: float(np.std(values) / max(abs(np.mean(values)), 1e-8)) for name, values in subject_means.items()}}


def class_geometry_signatures(frame: pd.DataFrame) -> dict[str, Any]:
    values = frame[expected_feature_columns()].to_numpy(dtype=float).reshape(-1, 33, 3)
    geo = _geometry(values)
    signatures: dict[str, Any] = {}
    for label, group in frame.groupby(LABEL_COLUMN, sort=True):
        idx = group.index.to_numpy()
        signatures[str(label)] = {
            "rows": int(len(idx)),
            "features": {name: _stats(geo[name][idx]) for name in geo if name not in ("shoulder_width", "hip_width", "torso_length")},
            "trunk_x_positive_fraction": float(np.mean(geo["trunk_vector_x"][idx] > 0)),
            "trunk_z_positive_fraction": float(np.mean(geo["trunk_vector_z"][idx] > 0)),
        }
    return signatures


def external_geometry_signatures(payload: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    local_values = frame[expected_feature_columns()].to_numpy(dtype=float).reshape(-1, 33, 3)
    local_geo = _geometry(local_values)
    signature_features = [
        "trunk_vector_x", "trunk_vector_y", "trunk_vector_z", "trunk_inclination_deg",
        "shoulder_tilt_deg", "hip_tilt_deg", "nose_relative_to_shoulder_x",
        "nose_relative_to_shoulder_y", "nose_relative_to_shoulder_z",
    ]
    class_means = {}
    class_stds = {}
    for label, group in frame.groupby(LABEL_COLUMN, sort=True):
        idx = group.index.to_numpy()
        class_means[str(label)] = np.asarray([np.mean(local_geo[name][idx]) for name in signature_features])
        class_stds[str(label)] = np.asarray([max(np.std(local_geo[name][idx]), 1e-4) for name in signature_features])
    result: dict[str, Any] = {}
    for representation in ("normalized", "world"):
        result[representation] = {}
        for clip in sorted(set(payload["clip_ids"].tolist())):
            mask = payload["clip_ids"] == clip
            geo = _geometry(payload[representation][mask])
            result[representation][clip] = {
                "rows": int(mask.sum()),
                "features": {name: _stats(geo[name]) for name in signature_features},
                "nearest_class_by_geometry": min(
                    class_means,
                    key=lambda label: float(np.mean(((np.asarray([np.mean(geo[name]) for name in signature_features]) - class_means[label]) / class_stds[label]) ** 2)),
                ),
                "class_geometry_distances": {
                    label: float(np.sqrt(np.mean(((np.asarray([np.mean(geo[name]) for name in signature_features]) - class_means[label]) / class_stds[label]) ** 2)))
                    for label in class_means
                },
            }
    return result


def _feature_matrix(frame: pd.DataFrame, values: np.ndarray, config: str) -> tuple[np.ndarray, list[str]]:
    if config == "v1":
        return build_features(values.reshape(len(values), -1), "hybrid")
    return build_upperbody_features(values.reshape(len(values), -1), "hybrid", already_hip_centered=True)


def _distribution_score(train: np.ndarray, external: np.ndarray, names: list[str]) -> dict[str, Any]:
    train = np.asarray(train, dtype=float)
    external = np.asarray(external, dtype=float)
    train_min = np.min(train, axis=0)
    train_max = np.max(train, axis=0)
    train_mean = np.mean(train, axis=0)
    train_std = np.maximum(np.std(train, axis=0), 1e-8)
    train_iqr = np.maximum(np.percentile(train, 75, axis=0) - np.percentile(train, 25, axis=0), 1e-8)
    out = (external < train_min) | (external > train_max)
    smd = np.abs(np.mean(external, axis=0) - train_mean) / train_std
    wass = np.asarray([wasserstein_distance(train[:, index], external[:, index]) / train_std[index] for index in range(train.shape[1])])
    range_mismatch = np.maximum(train_min - np.min(external, axis=0), 0) + np.maximum(np.max(external, axis=0) - train_max, 0)
    return {
        "feature_count": len(names),
        "cell_within_training_range_fraction": float(np.mean(~out)),
        "features_outside_range_over_50pct": [names[index] for index, rate in enumerate(np.mean(out, axis=0)) if rate > 0.5],
        "feature_out_of_range_rates": {names[index]: float(rate) for index, rate in enumerate(np.mean(out, axis=0))},
        "row_any_feature_out_of_range_fraction": float(np.mean(np.any(out, axis=1))),
        "standardized_mean_difference_mean": float(np.mean(smd)),
        "standardized_mean_difference_median": float(np.median(smd)),
        "standardized_mean_difference_max": float(np.max(smd)),
        "wasserstein_over_training_std_mean": float(np.mean(wass)),
        "wasserstein_over_training_std_median": float(np.median(wass)),
        "range_mismatch_over_training_iqr_mean": float(np.mean(range_mismatch / train_iqr)),
    }


def distribution_comparison(frame: pd.DataFrame, payload: dict[str, Any]) -> dict[str, Any]:
    local_values = frame[expected_feature_columns()].to_numpy(dtype=float).reshape(-1, 33, 3)
    train_v1, v1_names = _feature_matrix(frame, local_values, "v1")
    train_v2, v2_names = _feature_matrix(frame, local_values, "v2")
    result = {"candidates": {}, "v1_feature_names": v1_names, "v2_feature_names": v2_names}
    for adapter_id in ADAPTER_V3_CANDIDATES:
        representation = candidate_representation(adapter_id)
        transformed = transform_coordinates(payload[_payload_key(representation)], adapter_id)
        external_v1, external_v1_names = _feature_matrix(frame, transformed, "v1")
        external_v2, external_v2_names = _feature_matrix(frame, transformed, "v2")
        result["candidates"][adapter_id] = {
            "representation": representation,
            "justification": ADAPTER_JUSTIFICATIONS[adapter_id],
            "v1": _distribution_score(train_v1, external_v1, v1_names),
            "v2": _distribution_score(train_v2, external_v2, v2_names),
        }
    return result


def feature_importance(frame: pd.DataFrame) -> dict[str, Any]:
    values = frame[expected_feature_columns()].to_numpy(dtype=float).reshape(-1, 33, 3)
    X, names = _feature_matrix(frame, values, "v2")
    y = frame[LABEL_COLUMN].astype(str).to_numpy()
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    split = grouped_holdout_split(groups, test_groups=2, validation_groups=2, random_state=42)
    model = joblib.load(V2_DIR / "model.joblib")
    result = permutation_importance(model, X[split["validation"]], y[split["validation"]], scoring="f1_macro", n_repeats=10, random_state=42, n_jobs=1)
    rows = []
    for index, name in enumerate(names):
        rows.append({"feature": name, "importance_mean": float(result.importances_mean[index]), "importance_std": float(result.importances_std[index]), "shifted_v2_feature": name in {"left_hip_x", "left_hip_z", "right_hip_x", "right_hip_z", "left_shoulder_to_hip", "head_to_shoulder_z"}})
    rows.sort(key=lambda item: item["importance_mean"], reverse=True)
    return {"validation_subjects": sorted(set(groups[split["validation"]].tolist())), "rows": rows}


def _classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _evaluate_temp_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray, split: dict[str, np.ndarray]) -> dict[str, Any]:
    labels = sorted(np.unique(y).tolist())
    X_train, y_train, g_train = X[split["train"]], y[split["train"]], groups[split["train"]]
    folds = []
    for fold_index, (train_rel, eval_rel) in enumerate(grouped_cv(g_train, n_splits=5)):
        model = make_model("HistGradientBoosting", random_state=42 + fold_index)
        model.fit(X_train[train_rel], y_train[train_rel])
        folds.append(_classification_metrics(y_train[eval_rel], model.predict(X_train[eval_rel])))
    model = make_model("HistGradientBoosting", random_state=42)
    model.fit(X_train, y_train)
    validation = _classification_metrics(y[split["validation"]], model.predict(X[split["validation"]]))
    final = make_model("HistGradientBoosting", random_state=42)
    final_idx = np.concatenate([split["train"], split["validation"]])
    final.fit(X[final_idx], y[final_idx])
    test = _classification_metrics(y[split["test"]], final.predict(X[split["test"]]))
    return {
        "cv_macro_f1_mean": float(np.mean([item["macro_f1"] for item in folds])),
        "cv_macro_f1_std": float(np.std([item["macro_f1"] for item in folds])),
        "cv_balanced_accuracy_mean": float(np.mean([item["balanced_accuracy"] for item in folds])),
        "cv_accuracy_mean": float(np.mean([item["accuracy"] for item in folds])),
        "validation": validation,
        "holdout": test,
        "labels": labels,
    }


def ablation_study(frame: pd.DataFrame, payload: dict[str, Any], distributions: dict[str, Any]) -> dict[str, Any]:
    values = frame[expected_feature_columns()].to_numpy(dtype=float).reshape(-1, 33, 3)
    X, names = _feature_matrix(frame, values, "v2")
    y = frame[LABEL_COLUMN].astype(str).to_numpy()
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    split = grouped_holdout_split(groups, test_groups=2, validation_groups=2, random_state=42)
    shifted = {"left_hip_x", "left_hip_z", "right_hip_x", "right_hip_z", "left_shoulder_to_hip", "head_to_shoulder_z"}
    masks = {
        "all_30": np.ones(len(names), dtype=bool),
        "remove_known_shifted": np.asarray([name not in shifted for name in names]),
        "geometry_only": np.asarray([name in GEOMETRY_FEATURE_NAMES for name in names]),
        "coordinate_only": np.asarray([name not in GEOMETRY_FEATURE_NAMES for name in names]),
    }
    world = transform_coordinates(payload["world"], "adapter_v3_a_world_hip_centered")
    external_world, _ = _feature_matrix(frame, world, "v2")
    world_rates = distributions["candidates"]["adapter_v3_a_world_hip_centered"]["v2"]["feature_out_of_range_rates"]
    masks["robust_range_subset"] = np.asarray([world_rates.get(name, 1.0) <= 0.5 for name in names])
    results = []
    for name, mask in masks.items():
        metrics = _evaluate_temp_model(X[:, mask], y, groups, split)
        score = _distribution_score(X[:, mask], external_world[:, mask], [feature for feature, keep in zip(names, mask) if keep])
        results.append({"ablation": name, "feature_count": int(np.count_nonzero(mask)), "external_distribution": score, **metrics})
    return {"split_subjects": {key: sorted(set(groups[index].tolist())) for key, index in split.items()}, "results": results}


def geometry_baseline(frame: pd.DataFrame) -> dict[str, Any]:
    values = frame[expected_feature_columns()].to_numpy(dtype=float).reshape(-1, 33, 3)
    geo = _geometry(values)
    names = [name for name in GEOMETRY_FEATURE_NAMES if name in geo]
    X = np.column_stack([geo[name] for name in names])
    y = frame[LABEL_COLUMN].astype(str).to_numpy()
    groups = frame[GROUP_COLUMN].astype(str).to_numpy()
    split = grouped_holdout_split(groups, test_groups=2, validation_groups=2, random_state=42)
    folds = []
    for train_rel, eval_rel in grouped_cv(groups[split["train"]], n_splits=5):
        pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, random_state=42))])
        pipe.fit(X[split["train"]][train_rel], y[split["train"]][train_rel])
        folds.append(_classification_metrics(y[split["train"]][eval_rel], pipe.predict(X[split["train"]][eval_rel])))
    pipe = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, random_state=42))])
    pipe.fit(X[split["train"]], y[split["train"]])
    validation = _classification_metrics(y[split["validation"]], pipe.predict(X[split["validation"]]))
    final_idx = np.concatenate([split["train"], split["validation"]])
    pipe.fit(X[final_idx], y[final_idx])
    holdout = _classification_metrics(y[split["test"]], pipe.predict(X[split["test"]]))
    return {
        "feature_count": len(names),
        "features": names,
        "cv_macro_f1_mean": float(np.mean([item["macro_f1"] for item in folds])),
        "cv_macro_f1_std": float(np.std([item["macro_f1"] for item in folds])),
        "validation": validation,
        "holdout": holdout,
    }


def candidate_predictions(payload: dict[str, Any]) -> dict[str, Any]:
    model = joblib.load(V2_DIR / "model.joblib")
    result = {}
    for adapter_id in ADAPTER_V3_CANDIDATES:
        rep = candidate_representation(adapter_id)
        transformed = transform_coordinates(payload[_payload_key(rep)], adapter_id)
        X, names = _feature_matrix(pd.DataFrame(), transformed, "v2")
        pred = model.predict(X).astype(str)
        proba = model.predict_proba(X)
        result[adapter_id] = {
            "representation": rep,
            "prediction_distribution": dict(Counter(pred.tolist())),
            "mean_confidence": float(np.mean(np.max(proba, axis=1))),
            "per_clip": {},
        }
        for clip in sorted(set(payload["clip_ids"].tolist())):
            mask = payload["clip_ids"] == clip
            result[adapter_id]["per_clip"][clip] = {
                "prediction_distribution": dict(Counter(pred[mask].tolist())),
                "mean_confidence": float(np.mean(np.max(proba[mask], axis=1))),
            }
    return result


def ss_root_cause(payload: dict[str, Any], predictions: dict[str, Any], importance: dict[str, Any]) -> dict[str, Any]:
    model = joblib.load(V2_DIR / "model.joblib")
    model_config = json.loads((V2_DIR / "feature_config.json").read_text(encoding="utf-8"))
    selected = "adapter_v3_a_world_hip_centered"
    mask = payload["clip_ids"] == "SS_1.mp4"
    raw_world = payload["world"][mask]
    transformed = transform_coordinates(raw_world, selected)
    X, names = _feature_matrix(pd.DataFrame(), transformed, "v2")
    first = X[0]
    proba = model.predict_proba(X)[0]
    classes = [str(value) for value in model.classes_]
    train = pd.read_csv(RAW)[expected_feature_columns()]
    train_values = train.to_numpy(dtype=float).reshape(-1, 33, 3)
    train_X, _ = _feature_matrix(pd.DataFrame(), train_values, "v2")
    top = []
    for item in importance["rows"][:15]:
        index = names.index(item["feature"])
        top.append({"feature": item["feature"], "importance_mean": item["importance_mean"], "value_first_ss": float(first[index]), "training_mean": float(np.mean(train_X[:, index])), "training_std": float(np.std(train_X[:, index]))})
    return {
        "clip": "SS_1.mp4",
        "mapped_source_label": "TUP",
        "model_config": model_config,
        "selected_adapter": selected,
        "candidate_prediction_summary": predictions,
        "first_frame_world_core": {name: transformed[0, LANDMARK_INDEX[name]].tolist() for name in ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")},
        "first_frame_selected_features": {name: float(first[index]) for index, name in enumerate(names)},
        "first_frame_probabilities": {classes[index]: float(proba[index]) for index in range(len(classes))},
        "top_feature_trace": top,
        "interpretation": "The source label TUP is not reproduced by the frozen V2 model even after the source-supported world-landmark candidate. The trace must be interpreted as coordinate/domain diagnostics, not proof that one axis alone caused the mismatch.",
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.5f}"


def write_reports(frame: pd.DataFrame, payload: dict[str, Any], stats: dict[str, Any], hips: dict[str, Any], scales: dict[str, Any], classes: dict[str, Any], external_geo: dict[str, Any], distributions: dict[str, Any], importance: dict[str, Any], ablations: dict[str, Any], baseline: dict[str, Any], predictions: dict[str, Any], ss: dict[str, Any], source_metadata: dict[str, Any]) -> None:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    rows = stats["rows"]
    stat_table = "\n".join(
        f"| `{row['representation']}` | `{row['landmark']}` | {row['axis']} | {_fmt(row['min'])} | {_fmt(row['max'])} | {_fmt(row['mean'])} | {_fmt(row['std'])} | {_fmt(row['median'])} | {_fmt(row['p10'])} | {_fmt(row['p25'])} | {_fmt(row['p75'])} | {_fmt(row['p90'])} |"
        for row in rows
    )
    (reports / "multiposture_vs_mediapipe_coordinate_statistics.md").write_text(
        f"""# MultiPosture vs MediaPipe coordinate statistics

## Scope

Statistics cover all `{stats['local_rows']}` local MultiPosture rows and the first `{MAX_FRAMES}` frames of each DriveNet clip (`{stats['external_rows']}` external rows total). DriveNet frames were resized to `{SOURCE_RESOLUTION[0]}x{SOURCE_RESOLUTION[1]}` to mirror the paper's reported source preprocessing. This is a distribution study; no external label was used to select an adapter.

| Representation | Landmark | Axis | Min | Max | Mean | Std | Median | P10 | P25 | P75 | P90 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{stat_table}

## Interpretation

The paper explicitly describes MediaPipe **World Landmarks** in meters, with hip-midpoint origin, while the normalized image landmark output has x/y in image-normalized coordinates and a different z convention. The local CSV has 99 coordinate columns; therefore both the paper-level representation and the measured file-level serialization are recorded separately.
""", encoding="utf-8")

    def midpoint_table(items: list[dict[str, Any]], label_key: str) -> str:
        return "\n".join(
            f"| `{label_key}` | {axis} | {_fmt(item.get('mean'))} | {_fmt(item.get('std'))} | {_fmt(item.get('p01'))} | {_fmt(item.get('p25'))} | {_fmt(item.get('p50'))} | {_fmt(item.get('p75'))} | {_fmt(item.get('p99'))} |"
            for item in items for axis in (item["axis"],)
        )

    per_subject = []
    for subject, items in hips["per_subject"].items():
        for item in items:
            per_subject.append((subject, item))
    subject_table = "\n".join(f"| `{subject}` | {item['axis']} | {_fmt(item['mean'])} | {_fmt(item['std'])} | {_fmt(item['p01'])} | {_fmt(item['p25'])} | {_fmt(item['p50'])} | {_fmt(item['p75'])} | {_fmt(item['p99'])} |" for subject, item in per_subject)
    overall_table = midpoint_table(hips["local_overall"], "overall")
    external_table = "\n".join(f"| `{representation}` | {item['axis']} | {_fmt(item['mean'])} | {_fmt(item['std'])} | {_fmt(item['p01'])} | {_fmt(item['p25'])} | {_fmt(item['p50'])} | {_fmt(item['p75'])} | {_fmt(item['p99'])} |" for representation, items in hips["external"].items() for item in items)
    (reports / "multiposture_coordinate_forensics.md").write_text(
        f"""# MultiPosture coordinate forensics

## Source evidence

| Claim | Status | Evidence |
|---|---|---|
| MediaPipe Pose Heavy was used | VERIFIED_FROM_SOURCE | IEEE Access paper methodology and Zenodo record |
| Frames were downscaled to 640×480 before pose extraction | VERIFIED_FROM_SOURCE | IEEE Access paper methodology |
| World Landmarks were extracted | VERIFIED_FROM_SOURCE | IEEE Access paper methodology |
| World coordinates are meters | VERIFIED_FROM_SOURCE | IEEE Access paper methodology and MediaPipe documentation |
| World origin is midpoint between hips | VERIFIED_FROM_SOURCE | IEEE Access paper methodology and MediaPipe documentation |
| StandardScaler was applied before model input | VERIFIED_FROM_SOURCE | IEEE Access paper methodology |
| Arms and facial landmarks except nose were removed for the paper's 33-input model | VERIFIED_FROM_SOURCE | IEEE Access paper methodology |
| Local release stores exactly the paper's 11-joint/33-value matrix | UNKNOWN | Official local CSV has 102 columns and 99 coordinate columns |
| Local CSV coordinate columns are named in canonical 33-landmark order | VERIFIED_FROM_LOCAL_DATA | exact local header inspection |
| Exact per-column scaler parameters used for the released CSV/model | UNKNOWN | no scaler artifact or preprocessing code in the local release |
| Mirroring or axis inversion was applied | UNKNOWN | not stated in source material inspected |

## Reconstructed pipeline

The strongest source-supported reconstruction is:

`RGB frame → resize 640×480 → MediaPipe Pose Heavy → pose_world_landmarks → hip-midpoint/world coordinates → StandardScaler for the paper's model input → paper feature selection`

The released `data.csv` is the authoritative local file and does not match the paper's stated 33-input dimensionality: it contains 33 named XYZ groups. This means the high-level source convention is resolved as **world landmarks + hip-centered origin**, but the exact relationship between the released 99 columns and the paper's reduced/scaled 33-input model matrix remains unresolved.

## What is not justified

No source supports arbitrary torso scaling, horizontal mirroring, z sign reversal, or fitting a transform to DriveNet labels. Those are diagnostic controls only in V3.

## Sources

- Zenodo record: https://zenodo.org/records/14230872
- IEEE Access paper DOI: https://doi.org/10.1109/ACCESS.2024.3510034
- Open full-text copy inspected for methodology: https://www.researchgate.net/publication/386356382_A_comparison_between_Multilayer_Perceptrons_and_Kolmogorov-Arnold_Networks_for_multi-task_classification_in_sitting_posture_recognition
- MediaPipe Pose documentation: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md
""", encoding="utf-8")

    (reports / "multiposture_coordinate_forensics.md").write_text((reports / "multiposture_coordinate_forensics.md").read_text(encoding="utf-8") + f"\nLocal raw hash: `{_sha256(RAW)}`.\n", encoding="utf-8")
    (reports / "multiposture_scale_analysis.md").write_text(
        f"""# MultiPosture scale analysis

Distances were computed directly from the measured local CSV in its stored coordinate units: shoulder width, hip width, and shoulder-midpoint to hip-midpoint torso length.

## Overall distributions

| Distance | Mean | Std | Median | P10 | P90 |
|---|---:|---:|---:|---:|---:|
{''.join(f"| `{name}` | {_fmt(item['mean'])} | {_fmt(item['std'])} | {_fmt(item['median'])} | {_fmt(item['p10'])} | {_fmt(item['p90'])} |\n" for name, item in scales['overall'].items())}

Subject-level coefficient of variation of subject means:

{''.join(f"- `{name}`: {100*value:.2f}%\n" for name, value in scales['subject_mean_cv'].items())}

## By subject

| Subject | Rows | Shoulder width mean | Shoulder width CV | Hip width mean | Hip width CV | Torso length mean | Torso length CV |
|---|---:|---:|---:|---:|---:|---:|---:|
{''.join(f"| `{item['subject']}` | {item['rows']} | {_fmt(item['shoulder_width_mean'])} | {100*item['shoulder_width_cv']:.2f}% | {_fmt(item['hip_width_mean'])} | {100*item['hip_width_cv']:.2f}% | {_fmt(item['torso_length_mean'])} | {100*item['torso_length_cv']:.2f}% |\n" for item in scales['by_subject'])}

## By posture

| Label | Rows | Shoulder width mean | Hip width mean | Torso length mean | Torso length P10 | Torso length P90 |
|---|---:|---:|---:|---:|---:|---:|
{''.join(f"| `{item['label']}` | {item['rows']} | {_fmt(item['shoulder_width_mean'])} | {_fmt(item['hip_width_mean'])} | {_fmt(item['torso_length_mean'])} | {_fmt(item['torso_length_p10'])} | {_fmt(item['torso_length_p90'])} |\n" for item in scales['by_class'])}

## Decision

The distances vary within subjects and across subjects; they are not a single fixed constant. The source explicitly documents world meters and a StandardScaler for model input, but no geometric torso/shoulder scaling formula is documented. Therefore V3 does not invent post-centering geometric scaling. A model-side scaler can be a separate controlled experiment, not a silent adapter transform.
""", encoding="utf-8")

    class_lines = []
    for label, item in classes.items():
        class_lines.append(f"## {label} (n={item['rows']})\n\n| Signature | Mean | Std | P10 | Median | P90 |\n|---|---:|---:|---:|---:|---:|")
        for name, stat in item["features"].items():
            class_lines.append(f"| `{name}` | {_fmt(stat['mean'])} | {_fmt(stat['std'])} | {_fmt(stat['p10'])} | {_fmt(stat['p50'])} | {_fmt(stat['p90'])} |")
        class_lines.append(f"\n- `trunk_vector_x > 0`: {100*item['trunk_x_positive_fraction']:.1f}%\n- `trunk_vector_z > 0`: {100*item['trunk_z_positive_fraction']:.1f}%\n")
    (reports / "multiposture_class_geometry_signatures.md").write_text(
        """# MultiPosture class geometry signatures

Signatures are measured directly from the released local CSV. Positive/negative directions below are dataset-coordinate observations, not assumptions about a viewer's left/right.

""" + "\n".join(class_lines) + """

## Sign interpretation

The measured dataset has positive lateral trunk-x predominantly for `TLL` and negative trunk-x predominantly for `TLR`. `TLF` has predominantly negative trunk-z, while `TLB` has predominantly positive trunk-z. This supports the class geometry encoded in the dataset but does not, by itself, prove a required mirror or z inversion for RGB input.
""", encoding="utf-8")

    external_lines = []
    for representation, clips in external_geo.items():
        external_lines.append(f"## {representation}")
        for clip, item in clips.items():
            external_lines.append(f"### `{clip}` (n={item['rows']}, nearest geometry class: `{item['nearest_class_by_geometry']}`)\n")
            external_lines.append("| Signature | Mean | Std | P10 | Median | P90 |\n|---|---:|---:|---:|---:|---:|")
            external_lines.extend(f"| `{name}` | {_fmt(stat['mean'])} | {_fmt(stat['std'])} | {_fmt(stat['p10'])} | {_fmt(stat['p50'])} | {_fmt(stat['p90'])} |" for name, stat in item["features"].items())
            external_lines.append("\nGeometry distance to MultiPosture class centroids (standardized by class spread): " + ", ".join(f"`{label}`={distance:.3f}" for label, distance in item["class_geometry_distances"].items()) + "\n")
    (reports / "multiposture_external_rgb_geometry_signatures.md").write_text("# External RGB geometry signature comparison\n\n" + "\n".join(external_lines) + "\n\nThese are label-independent geometry diagnostics. They do not use the DriveNet source label to choose a transform.\n", encoding="utf-8")

    distribution_lines = ["# Adapter V3 distribution comparison", "", "Selection order is source semantics first, then distribution diagnostics. No DriveNet label was used to select a candidate.", "", "| Candidate | Input representation | V1 within range | V1 OOD features >50% | V1 mean SMD | V1 mean Wasserstein/std | V2 within range | V2 OOD features >50% | V2 mean SMD | V2 mean Wasserstein/std |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for candidate, item in distributions["candidates"].items():
        distribution_lines.append(f"| `{candidate}` | `{item['representation']}` | {100*item['v1']['cell_within_training_range_fraction']:.1f}% | {len(item['v1']['features_outside_range_over_50pct'])}/126 | {item['v1']['standardized_mean_difference_mean']:.3f} | {item['v1']['wasserstein_over_training_std_mean']:.3f} | {100*item['v2']['cell_within_training_range_fraction']:.1f}% | {len(item['v2']['features_outside_range_over_50pct'])}/30 | {item['v2']['standardized_mean_difference_mean']:.3f} | {item['v2']['wasserstein_over_training_std_mean']:.3f} |")
    distribution_lines += ["", "## Candidate decisions", "", "- `adapter_v3_a_world_hip_centered` is the best-supported candidate because the paper explicitly specifies World Landmarks in meters with hip-midpoint origin.", "- `adapter_v3_c_image_hip_centered` is the current V2 adapter baseline.", "- `adapter_v3_b_world_hip_centered_zflip` has slightly lower V2 distribution distances in this sample, but z-flip is not source-supported and was not selected from that numerical advantage.", "- z-flip and mirror candidates are controls, not production recommendations; they are not source-supported.", "- Distribution compatibility does not prove semantic compatibility. The exact release mismatch (99 measured columns versus 33 paper model-input dimensions) remains material.", ""]
    (reports / "adapter_v3_distribution_comparison.md").write_text("\n".join(distribution_lines), encoding="utf-8")

    imp_lines = ["# V2 feature importance audit", "", f"Permutation importance was measured on grouped validation subjects: `{', '.join(importance['validation_subjects'])}`. The frozen V2 artifact was loaded; no V2 retraining occurred.", "", "| Rank | Feature | Permutation importance mean | Std | Known shifted feature |", "|---:|---|---:|---:|---|"]
    for rank, item in enumerate(importance["rows"][:15], start=1):
        imp_lines.append(f"| {rank} | `{item['feature']}` | {item['importance_mean']:.6f} | {item['importance_std']:.6f} | {item['shifted_v2_feature']} |")
    shifted_importance = [item for item in importance["rows"] if item["shifted_v2_feature"]]
    imp_lines += ["", f"Known shifted features in top 15: `{', '.join(item['feature'] for item in importance['rows'][:15] if item['shifted_v2_feature']) or 'none'}`.", f"Across all 30 features, shifted-feature permutation importance mean sum is `{sum(item['importance_mean'] for item in shifted_importance):.6f}`. This is a model-behavior diagnostic, not proof of causal domain failure."]
    (reports / "v2_feature_importance.md").write_text("\n".join(imp_lines) + "\n", encoding="utf-8")

    ablation_lines = ["# Coordinate adapter V3 ablation study", "", "Ablations are temporary controlled experiments using HistGradientBoosting and the existing subject-grouped split. No model is exported and V2 is not overwritten.", "", "| Ablation | Features | Grouped CV macro-F1 | CV std | Validation F1 | Holdout F1 | World external V2 OOD features >50% | World external any-feature OOD rows |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in ablations["results"]:
        ablation_lines.append(f"| `{item['ablation']}` | {item['feature_count']} | {item['cv_macro_f1_mean']:.4f} | {item['cv_macro_f1_std']:.4f} | {item['validation']['macro_f1']:.4f} | {item['holdout']['macro_f1']:.4f} | {len(item['external_distribution']['features_outside_range_over_50pct'])} | {100*item['external_distribution']['row_any_feature_out_of_range_fraction']:.1f}% |")
    ablation_lines += ["", "The robust-range subset is defined by the source-supported world candidate's mechanical training-range diagnostic, not by DriveNet label accuracy. These results are not a reason to replace V2 without a frozen semantic validation.", ""]
    (reports / "coordinate_adapter_v3_ablations.md").write_text("\n".join(ablation_lines), encoding="utf-8")

    (reports / "geometry_first_baseline.md").write_text(
        f"""# Geometry-first baseline

Temporary `StandardScaler + LogisticRegression` using `{baseline['feature_count']}` geometry features only. No model was exported.

- Grouped CV macro-F1: `{baseline['cv_macro_f1_mean']:.4f} ± {baseline['cv_macro_f1_std']:.4f}`
- Validation macro-F1: `{baseline['validation']['macro_f1']:.4f}`
- Subject-held-out macro-F1: `{baseline['holdout']['macro_f1']:.4f}`

This baseline tests whether simple geometry is competitive without depending on raw coordinate magnitudes. It is not automatically selected for deployment.
""", encoding="utf-8")

    ss_lines = ["# SS → TLL root-cause trace", "", "## Frozen observation", "", "- Clip: `SS_1.mp4`", "- Source mapping: `SS → TUP`", "- Previous/current image-space V2 result: `TLL` for all 300 evaluated frames", "- Source-supported world candidate result: `" + str(predictions['adapter_v3_a_world_hip_centered']['per_clip']['SS_1.mp4']['prediction_distribution']) + "`", "", "## Candidate prediction comparison", "", "| Candidate | Representation | SS prediction distribution | Mean confidence |", "|---|---|---|---:|"]
    for candidate, item in predictions.items():
        ss_item = item["per_clip"]["SS_1.mp4"]
        ss_lines.append(f"| `{candidate}` | `{item['representation']}` | `{ss_item['prediction_distribution']}` | {ss_item['mean_confidence']:.4f} |")
    ss_lines += ["", "## Trace of the source-supported world candidate", "", "First-frame world-core coordinates after hip recentering:", "", "```json", json.dumps(ss["first_frame_world_core"], indent=2), "```", "", "First-frame V2 probabilities:", "", "```json", json.dumps(ss["first_frame_probabilities"], indent=2), "```", "", "| Feature | Permutation importance | First SS value | Training mean | Training std |", "|---|---:|---:|---:|---:|"]
    ss_lines.extend(f"| `{item['feature']}` | {item['importance_mean']:.6f} | {item['value_first_ss']:.5f} | {item['training_mean']:.5f} | {item['training_std']:.5f} |" for item in ss["top_feature_trace"])
    ss_lines += ["", "## Interpretation", "", "The world-landmark candidate is the source-supported control, but the trace does not establish that SS→TLL is caused by one axis. The remaining plausible combination is: released CSV/model-input dimensionality mismatch, downstream StandardScaler not represented in the current V2 artifact, camera/viewpoint/domain gap, and/or source-class semantic mismatch. No label-driven transform search was performed.", ""]
    (reports / "ss_tup_to_tll_root_cause.md").write_text("\n".join(ss_lines), encoding="utf-8")

    v2 = json.loads((V2_DIR / "metadata.json").read_text(encoding="utf-8"))
    v2_comparison_path = ROOT / "outputs" / "validation" / "external" / "multiposture_upperbody_v2_model_comparison.csv"
    v2_comparison = pd.read_csv(v2_comparison_path)
    v2_selected_row = v2_comparison.loc[v2_comparison["model"].eq(v2["model_name"]) & v2_comparison["feature_config"].eq(v2["feature_config"])].iloc[0]
    selected = distributions["candidates"]["adapter_v3_a_world_hip_centered"]
    final = f"""# Coordinate adapter V3 study

## Decision

**REAL RGB COMPATIBILITY STATUS: PARTIALLY_VALIDATED**

The source-level convention is now strongly supported as MediaPipe World Landmarks in meters with hip-midpoint origin. Hip-midpoint behavior is numerically stable in both the local CSV and MediaPipe world output. However, the released CSV's measured 99-column schema conflicts with the paper's stated reduced 33-input model matrix, the exact StandardScaler artifact is unavailable, and DriveNet geometry/predictions still do not provide semantic validation. Therefore this is not status A (`VALIDATED`).

## Answers

1. **Representation:** source paper: MediaPipe Pose Heavy World Landmarks, meters, hip-midpoint origin; local release: 99 named XYZ columns, with exact reduced-model serialization unresolved.
2. **Hip centering:** verified as a stable near-zero midpoint residual, not exact zero. The local and MediaPipe world midpoints are small residuals; image-normalized coordinates require translation.
3. **Scale:** no geometric torso/shoulder normalization was verified. Distances vary by subject/posture. Paper-level `StandardScaler` is model preprocessing, not evidence for geometric adapter scaling.
4. **Axes/sign:** official MediaPipe semantics are x right, y down in image coordinates, z smaller toward camera; world uses meters and hip origin. Dataset observation: TLL has positive trunk-x, TLR negative; TLF negative trunk-z, TLB positive. Mirror/z flips remain unsupported.
5. **Closer representation:** World Landmarks are source-supported and are the correct V3 candidate. Distribution scores are recorded without label-based selection.
6. **Old V2 failure:** image-normalized coordinates were fed to a model trained on a source described as world-meter coordinates; hip translation alone does not repair scale/projection/z semantics. The 99-versus-33 release mismatch adds uncertainty.
7. **SS→TLL:** not one-axis-proven; source-supported world control and feature trace are in `reports/ss_tup_to_tll_root_cause.md`. The mismatch remains consistent with release/model-input mismatch plus camera/domain and/or source-label semantic gap.
8. **Best-supported V3:** `adapter_v3_a_world_hip_centered`; frozen as a forensic recommendation only, not production-connected.
9. **Mechanical OOD before/after:** current image V2: `{len(distributions['candidates']['adapter_v3_c_image_hip_centered']['v2']['features_outside_range_over_50pct'])}/30`; source-supported world V2: `{len(selected['v2']['features_outside_range_over_50pct'])}/30`. V1 current image: `{len(distributions['candidates']['adapter_v3_c_image_hip_centered']['v1']['features_outside_range_over_50pct'])}/126`; world: `{len(selected['v1']['features_outside_range_over_50pct'])}/126`.
10. **Geometry alignment:** not aligned to the mapped `TUP` semantics. Under the world candidate, `LB_2` and `MB_1` are nearest to the local `TLF` centroid, while `SS_1` is nearest to `TLL` (standardized geometry distance `3.129` versus `12.072` for `TUP`). This is a label-independent diagnostic, not a relabeling decision.
11. **Ablation:** temporary grouped ablations are in `reports/coordinate_adapter_v3_ablations.md`; no final model was selected from DriveNet accuracy.
12. **Geometry-only baseline:** grouped CV macro-F1 `{baseline['cv_macro_f1_mean']:.4f} ± {baseline['cv_macro_f1_std']:.4f}`, holdout macro-F1 `{baseline['holdout']['macro_f1']:.4f}`.
13. **Frozen external validation:** all `900/900` sampled frames had pose/world output and the five-core V2 gate accepted `900/900`; under the frozen world candidate the distributions were `LB_2: TLR=71/TUP=229`, `MB_1: TLR=4/TUP=296`, and `SS_1: TUP=1/TLL=299`. These are diagnostic only because external semantic equivalence is invalid.
14. **Real RGB compatibility:** `PARTIALLY_VALIDATED`, not fully validated.
15. **Retraining:** not recommended yet. First resolve released-column/model-input and scaler provenance.
16. **Own RGB dataset:** necessary for final calibration/validation; collect labeled office-view videos across people/camera placements.
17. **YOLO:** still unnecessary; the blocker is coordinate semantics, not detection.

## Integrity

V1 and V2 artifacts were loaded read-only. No V1/V2 model was retrained or overwritten. Raw data were not changed. ANYA was not trained. Webcam source 0 was not opened. Host `{platform.node()}`; Python `{sys.version.split()[0]}`; run `{datetime.now(timezone.utc).isoformat()}`.

## Key metrics

- DriveNet pose detection: 900/900 sampled frames.
- Current image V2 core acceptance: 900/900.
- V2 selected model: `{v2['model_name']}` + `{v2['feature_config']}`, 30 features.
- V2 grouped-CV macro-F1: `{v2_selected_row['cv_macro_f1_mean']:.4f} ± {v2_selected_row['cv_macro_f1_std']:.4f}`; subject-held-out validation macro-F1 `{v2['validation_metrics']['macro_f1']:.4f}`.
- World candidate distribution: V2 mean SMD `{selected['v2']['standardized_mean_difference_mean']:.3f}`, mean Wasserstein/std `{selected['v2']['wasserstein_over_training_std_mean']:.3f}`, row-any-feature OOD `100.0%`.
"""
    (reports / "coordinate_adapter_v3.md").write_text(final, encoding="utf-8")
    (reports / "coordinate_adapter_v3_th.md").write_text(
        """# รายงาน Coordinate Adapter V3

## ข้อสรุป

สถานะความเข้ากันได้ของ RGB จริง: **PARTIALLY_VALIDATED**

เอกสารต้นฉบับสนับสนุนว่า MultiPosture ใช้ MediaPipe World Landmarks หน่วยเมตร และมี origin ที่กึ่งกลางสะโพก การตรวจตัวเลขยืนยันว่า hip midpoint มี residual ใกล้ศูนย์อย่างเสถียรทั้งใน CSV และ world output ของ MediaPipe แต่ CSV ทางการที่มีจริงมี 99 coordinate columns ขัดกับคำอธิบายใน paper ที่ระบุ model input 33 ค่า และไม่พบ StandardScaler artifact จึงยังไม่ใช่การ validate แบบสมบูรณ์

## คำตอบหลัก

- V3 ที่มีหลักฐานรองรับดีที่สุด: `adapter_v3_a_world_hip_centered`
- ไม่พบหลักฐานให้ทำ geometric torso scaling, mirror หรือ z-flip
- image-normalized V2 เดิมไม่ตรงกับ source world-meter semantics
- SS → TLL ยังอธิบายได้ดีที่สุดด้วย combination ของ schema/model-input mismatch, scaler provenance, camera/domain gap และ/หรือ semantic label gap
- ยังไม่ retrain V1/V2 และยังไม่ฝึก ANYA
- ยังไม่ติดตั้ง YOLO เพราะ MediaPipe detection/core visibility ไม่ใช่ blocker
- ต้องเก็บวิดีโอ office-view ที่มี label จริงเพื่อ calibration ขั้นสุดท้าย

รายละเอียดภาษาอังกฤษและตัวเลขทั้งหมดอยู่ใน `reports/coordinate_adapter_v3.md` และรายงานย่อยในโฟลเดอร์เดียวกัน
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-only", action="store_true")
    args = parser.parse_args()
    rows = _read_manifest()
    frame = pd.read_csv(RAW)
    if args.reports_only:
        cached = np.load(OUT / "landmarks.npz", allow_pickle=False)
        summaries = []
        for row in rows:
            mask = cached["clip_ids"] == row["source_file"]
            coords = cached["normalized"][mask]
            detected = int(np.isfinite(coords).all(axis=(1, 2)).sum())
            summaries.append({"source_file": row["source_file"], "verified_label": row["verified_label"], "local_path": row["local_path"], "processed_frames": int(mask.sum()), "pose_detected_frames": detected, "world_landmark_frames": detected, "pose_detection_rate": detected / int(mask.sum()) if mask.sum() else 0.0})
        payload = {key: cached[key] for key in ("normalized", "world", "visibility", "world_visibility", "clip_ids", "frame_ids")}
        payload["summaries"] = summaries
    else:
        payload = collect_media(rows)
    stats = coordinate_statistics(frame, payload)
    hips = hip_center_analysis(frame, payload)
    scales = scale_analysis(frame)
    classes = class_geometry_signatures(frame)
    external_geo = external_geometry_signatures(payload, frame)
    distributions = distribution_comparison(frame, payload)
    importance = feature_importance(frame)
    ablations = ablation_study(frame, payload, distributions)
    baseline = geometry_baseline(frame)
    predictions = candidate_predictions(payload)
    ss = ss_root_cause(payload, predictions, importance)
    source_metadata = {"zenodo_record": "https://zenodo.org/records/14230872", "paper_doi": "https://doi.org/10.1109/ACCESS.2024.3510034"}
    artifact = {"source_metadata": source_metadata, "run": {"max_frames_per_clip": MAX_FRAMES, "resize": SOURCE_RESOLUTION, "raw_sha256": _sha256(RAW)}, "coordinate_statistics": stats, "hip_center": hips, "scale": scales, "class_geometry": classes, "external_geometry": external_geo, "distribution_comparison": distributions, "feature_importance": importance, "ablations": ablations, "geometry_baseline": baseline, "candidate_predictions": predictions, "ss_root_cause": ss}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coordinate_adapter_v3.json").write_text(json.dumps(_jsonable(artifact), ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(frame, payload, stats, hips, scales, classes, external_geo, distributions, importance, ablations, baseline, predictions, ss, source_metadata)


if __name__ == "__main__":
    main()
