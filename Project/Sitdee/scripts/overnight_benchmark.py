"""Frozen overnight benchmark and audit for the RGB-native V4 track.

This module never trains or overwrites a model.  It reads the six verified
DriveNet videos, the frozen V4 artifact, and the existing V4 training cache;
all outputs are derived validation records under outputs/validation/overnight.
The dense pass uses 10 FPS and reports deterministic 1/2/5 FPS subsequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
import psutil
from scipy.stats import wasserstein_distance
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.posture_ai.features.rgb_native_v4 import (  # noqa: E402
    V4_CORE_LANDMARKS,
    V4_FEATURE_FAMILIES,
    build_rgb_native_v4_features,
)
from src.posture_ai.inference.pipeline import TemporalSmoother  # noqa: E402
from src.posture_ai.pose.landmarks import LANDMARK_INDEX  # noqa: E402

VIDEO_DIR = ROOT / "datasets" / "raw" / "rgb_native" / "drivenet_v6" / "source_videos"
V4_MODEL_DIR = ROOT / "outputs" / "models" / "rgb_native_v4"
V4_PROCESSED_DIR = ROOT / "datasets" / "processed" / "rgb_native"
OUT_DIR = ROOT / "outputs" / "validation" / "overnight"
REPORT_DIR = ROOT / "reports"
POSE_MODEL = ROOT / "metadata" / "pose_landmarker_heavy.task"
KNOWN_DRIVENET = ROOT / "outputs" / "metrics" / "rgb_native_v5_source_hashes.json"
ANYA_HASH_FILE = ROOT / "metadata" / "anya_hashes.sha256"
RAW_ANYA_DIR = ROOT / "datasets" / "raw" / "anya" / "official-download"
RAW_MULTIPOSTURE = ROOT / "datasets" / "raw" / "multiposture" / "data.csv"
BENCHMARK_FPS = 10.0
REPORT_RATES = (1.0, 2.0, 5.0, 10.0)


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values: Any) -> dict[str, float | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {name: None for name in ("min", "max", "mean", "std", "median", "p10", "p25", "p75", "p90")}
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def sampling_step(source_fps: float, target_fps: float) -> int:
    """Return the deterministic source-frame stride used by the benchmark."""
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    return max(1, int(round(source_fps / target_fps)))


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


def _point_array(points: Any) -> np.ndarray:
    return np.asarray([[float(point.x), float(point.y), float(point.z)] for point in points], dtype=float)


def _visibility(points: Any) -> np.ndarray:
    return np.asarray([float(getattr(point, "visibility", np.nan)) for point in points], dtype=float)


def _source_inventory() -> list[dict[str, Any]]:
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(VIDEO_DIR)
    inventory = []
    for path in sorted(VIDEO_DIR.glob("*.mp4")):
        if path.stem[:2] not in {"LB", "SS"}:
            continue
        inventory.append({
            "filename": path.name,
            "path": str(path),
            "source_label": path.stem.split("_")[0],
            "participant_id": path.stem.rsplit("_", 1)[-1],
            "video_id": path.stem,
            "mapped_project_label": {"LB": "TLB", "SS": "TUP"}.get(path.stem.split("_")[0], ""),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    if len(inventory) != 6:
        raise RuntimeError(f"Expected six verified LB/SS videos, found {len(inventory)}")
    return inventory


def _core_status(world: np.ndarray, image_vis: np.ndarray) -> tuple[bool, str]:
    required = [LANDMARK_INDEX[name] for name in V4_CORE_LANDMARKS]
    if not np.isfinite(world).all():
        return False, "missing_world_landmarks"
    if not np.isfinite(image_vis[required]).all():
        return False, "missing_core_visibility"
    if np.any(image_vis[required] < 0.5):
        return False, "core_visibility_below_0.5"
    return True, "accepted"


def _prediction_switches(labels: list[str]) -> int:
    return int(sum(left != right for left, right in zip(labels, labels[1:])))


def _smooth_predictions(rows: list[dict[str, Any]], labels: list[str]) -> list[str]:
    smoother = TemporalSmoother(labels, window_size=5, persistence_frames=3)
    result = []
    for row in rows:
        if row.get("prediction") is None:
            continue
        probabilities = np.asarray([row["probabilities"].get(label, 0.0) for label in labels], dtype=float)
        result.append(str(smoother.update(probabilities)["label"]))
    return result


def _rate_rows(rows: list[dict[str, Any]], source_fps: float, rate: float) -> list[dict[str, Any]]:
    step = sampling_step(source_fps, rate)
    return [row for row in rows if int(row["frame_index"]) % step == 0]


def _rate_summary(rows: list[dict[str, Any]], source_fps: float, rate: float, labels: list[str], train_min: np.ndarray, train_max: np.ndarray, train_mean: np.ndarray, train_std: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    selected = _rate_rows(rows, source_fps, rate)
    pose = [row for row in selected if row["pose_detected"]]
    accepted = [row for row in selected if row["accepted"]]
    predictions = [str(row["prediction"]) for row in accepted]
    smooth = _smooth_predictions(accepted, labels)
    matrix = np.asarray([row["features"] for row in accepted], dtype=float) if accepted else np.empty((0, len(train_min)))
    if len(matrix):
        outside = (matrix < train_min[None, :]) | (matrix > train_max[None, :])
        smd = np.abs((np.mean(matrix, axis=0) - train_mean) / np.maximum(train_std, 1e-8))
        ood = {
            "cell_fraction": float(outside.mean()),
            "feature_fraction_over_50pct": int(np.sum(outside.mean(axis=0) > 0.5)),
            "feature_names_over_50pct": [feature_names[index] for index, value in enumerate(outside.mean(axis=0)) if value > 0.5],
            "mean_abs_smd": float(np.mean(smd)),
        }
    else:
        outside = np.empty((0, len(train_min)), dtype=bool)
        ood = {"cell_fraction": None, "feature_fraction_over_50pct": 0, "feature_names_over_50pct": [], "mean_abs_smd": None}
    return {
        "rate_fps": rate,
        "stride_frames": sampling_step(source_fps, rate),
        "selected_frames": len(selected),
        "pose_detected_frames": len(pose),
        "core_accepted_frames": len(accepted),
        "pose_detection_rate": len(pose) / len(selected) if selected else 0.0,
        "core_acceptance_rate": len(accepted) / len(selected) if selected else 0.0,
        "prediction_distribution": dict(Counter(predictions)),
        "mean_confidence": float(np.mean([row["confidence"] for row in accepted])) if accepted else None,
        "raw_prediction_switches": _prediction_switches(predictions),
        "smoothed_prediction_switches": _prediction_switches(smooth),
        "smoothed_prediction_distribution": dict(Counter(smooth)),
        "pose_lost_frames": int(len(selected) - len(accepted)),
        "failure_reasons": dict(Counter(row["failure_reason"] for row in selected if not row["accepted"])),
        "ood": ood,
        "feature_matrix": matrix,
        "accepted_rows": accepted,
        "selected_rows": selected,
    }


def benchmark_drivenet() -> dict[str, Any]:
    inventory = _source_inventory()
    model = joblib.load(V4_MODEL_DIR / "model.joblib")
    config = json.loads((V4_MODEL_DIR / "feature_config.json").read_text(encoding="utf-8"))
    labels = [str(label) for label in getattr(model, "classes_", config.get("labels", ["TLB", "TUP"]))]
    feature_names = list(config["columns"])
    train_manifest = pd.read_csv(V4_PROCESSED_DIR / "manifest.csv")
    train_arrays = np.load(V4_PROCESSED_DIR / "features.npz", allow_pickle=False)
    train_mask = train_manifest["participant_id"].astype(str).isin(["1", "2"]).to_numpy()
    train_rows = train_manifest.loc[train_mask, "feature_row"].astype(int).to_numpy()
    train_matrix = np.asarray(train_arrays[config["name"]][train_rows], dtype=float)
    train_min = np.min(train_matrix, axis=0)
    train_max = np.max(train_matrix, axis=0)
    train_mean = np.mean(train_matrix, axis=0)
    train_std = np.std(train_matrix, axis=0)

    process = psutil.Process(os.getpid())
    rss_start = process.memory_info().rss
    global_start = time.perf_counter()
    per_video: dict[str, Any] = {}
    for item in inventory:
        capture = cv2.VideoCapture(item["path"])
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {item['path']}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dense_step = sampling_step(source_fps, BENCHMARK_FPS)
        rows: list[dict[str, Any]] = []
        frame_index = 0
        decoded = 0
        decode_start = time.perf_counter()
        pose_seconds = 0.0
        adapter_seconds = 0.0
        landmarker = _make_landmarker()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                decoded += 1
                if frame_index % dense_step != 0:
                    frame_index += 1
                    continue
                timestamp_ms = int(frame_index * 1000.0 / source_fps)
                pose_start = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                pose_seconds += time.perf_counter() - pose_start
                record: dict[str, Any] = {
                    "frame_index": frame_index,
                    "pose_detected": bool(result.pose_landmarks and result.pose_world_landmarks),
                    "accepted": False,
                    "failure_reason": "no_pose_detected",
                    "prediction": None,
                    "confidence": None,
                    "probabilities": {},
                    "features": None,
                    "image_visibility": None,
                    "geometry": None,
                }
                if result.pose_landmarks and result.pose_world_landmarks:
                    image_points = result.pose_landmarks[0]
                    world_points = result.pose_world_landmarks[0]
                    world = _point_array(world_points)
                    image_vis = _visibility(image_points)
                    accepted, reason = _core_status(world, image_vis)
                    record["failure_reason"] = reason
                    record["image_visibility"] = image_vis.tolist()
                    if accepted:
                        adapter_start = time.perf_counter()
                        features, names = build_rgb_native_v4_features(world, config["name"])
                        if names != feature_names:
                            raise AssertionError("Frozen V4 feature order differs from feature_config.json")
                        probabilities = np.asarray(model.predict_proba(features)[0], dtype=float) if hasattr(model, "predict_proba") else np.asarray([1.0], dtype=float)
                        prediction = str(model.predict(features)[0])
                        adapter_seconds += time.perf_counter() - adapter_start
                        record.update({
                            "accepted": True,
                            "failure_reason": "accepted",
                            "prediction": prediction,
                            "confidence": float(np.max(probabilities)),
                            "probabilities": {str(label): float(probabilities[i]) for i, label in enumerate(model.classes_)},
                            "features": features[0].tolist(),
                            "geometry": build_rgb_native_v4_features(world, "geometry")[0][0, :].tolist(),
                        })
                rows.append(record)
                frame_index += 1
        finally:
            landmarker.close()
            capture.release()
        decode_seconds = time.perf_counter() - decode_start
        rates = {}
        for rate in REPORT_RATES:
            summary = _rate_summary(rows, source_fps, rate, labels, train_min, train_max, train_mean, train_std, feature_names)
            summary.pop("feature_matrix", None)
            summary.pop("accepted_rows", None)
            summary.pop("selected_rows", None)
            rates[str(rate)] = summary
        per_video[item["video_id"]] = {
            **item,
            "source_fps": source_fps,
            "declared_frame_count": declared_count,
            "decoded_frame_count": decoded,
            "dense_inference_fps": BENCHMARK_FPS,
            "dense_step_frames": dense_step,
            "pose_seconds": pose_seconds,
            "adapter_classifier_seconds": adapter_seconds,
            "wall_seconds": decode_seconds,
            "effective_decoded_fps": decoded / decode_seconds if decode_seconds else 0.0,
            "effective_pose_fps": sum(1 for row in rows if row["pose_detected"]) / pose_seconds if pose_seconds else 0.0,
            "rates": rates,
        }

    elapsed = time.perf_counter() - global_start
    rss_end = process.memory_info().rss
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_contract": {
            "frozen_model": str(V4_MODEL_DIR),
            "feature_family": config["name"],
            "feature_count": len(feature_names),
            "dense_rate_fps": BENCHMARK_FPS,
            "reported_subsequence_rates_fps": list(REPORT_RATES),
            "selection_rule": "source frame index modulo round(source_fps / target_fps); all source frames decoded, only dense selected frames pose-processed; dense 10 FPS makes 1/2/5 FPS subsequences nested",
            "accuracy_status": "not_reported_as_independent_external_accuracy; DriveNet is the V4 training source",
        },
        "inventory": inventory,
        "per_video": per_video,
        "summary": {
            "videos": len(per_video),
            "participants": sorted({item["participant_id"] for item in inventory}),
            "decoded_frames": int(sum(item["decoded_frame_count"] for item in per_video.values())),
            "dense_selected_frames": int(sum(item["rates"][str(BENCHMARK_FPS)]["selected_frames"] for item in per_video.values())),
            "pose_detected_frames": int(sum(item["rates"][str(BENCHMARK_FPS)]["pose_detected_frames"] for item in per_video.values())),
            "core_accepted_frames": int(sum(item["rates"][str(BENCHMARK_FPS)]["core_accepted_frames"] for item in per_video.values())),
            "wall_seconds": elapsed,
            "rss_start_mb": rss_start / (1024**2),
            "rss_end_mb": rss_end / (1024**2),
            "rss_delta_mb": (rss_end - rss_start) / (1024**2),
        },
    }
    # Add aggregate rate summaries and per-feature OOD names after the pass.
    for rate in REPORT_RATES:
        all_rates = [item["rates"][str(rate)] for item in per_video.values()]
        prediction_distribution = Counter()
        smoothed_distribution = Counter()
        for item in all_rates:
            prediction_distribution.update(item["prediction_distribution"])
            smoothed_distribution.update(item["smoothed_prediction_distribution"])
        result.setdefault("aggregate_rates", {})[str(rate)] = {
            "selected_frames": int(sum(item["selected_frames"] for item in all_rates)),
            "pose_detected_frames": int(sum(item["pose_detected_frames"] for item in all_rates)),
            "core_accepted_frames": int(sum(item["core_accepted_frames"] for item in all_rates)),
            "pose_detection_rate": float(sum(item["pose_detected_frames"] for item in all_rates) / max(sum(item["selected_frames"] for item in all_rates), 1)),
            "core_acceptance_rate": float(sum(item["core_accepted_frames"] for item in all_rates) / max(sum(item["selected_frames"] for item in all_rates), 1)),
            "raw_prediction_switches": int(sum(item["raw_prediction_switches"] for item in all_rates)),
            "smoothed_prediction_switches": int(sum(item["smoothed_prediction_switches"] for item in all_rates)),
            "pose_lost_frames": int(sum(item["pose_lost_frames"] for item in all_rates)),
            "prediction_distribution": dict(prediction_distribution),
            "smoothed_prediction_distribution": dict(smoothed_distribution),
        }
    write_json(OUT_DIR / "drivenet_v4_benchmark.json", result)
    return result


def feature_audit() -> dict[str, Any]:
    manifest = pd.read_csv(V4_PROCESSED_DIR / "manifest.csv")
    config = json.loads((V4_MODEL_DIR / "feature_config.json").read_text(encoding="utf-8"))
    arrays = np.load(V4_PROCESSED_DIR / "features.npz", allow_pickle=False)
    matrix = np.asarray(arrays[config["name"]], dtype=float)
    names = list(config["columns"])
    groups = manifest["participant_id"].astype(str)
    train_rows = manifest.index[groups.isin(["1", "2"])].to_numpy()
    holdout_rows = manifest.index[groups == "3"].to_numpy()
    train = matrix[manifest.loc[train_rows, "feature_row"].astype(int).to_numpy()]
    holdout = matrix[manifest.loc[holdout_rows, "feature_row"].astype(int).to_numpy()]
    feature_rows = []
    for index, name in enumerate(names):
        train_std = float(np.std(train[:, index]))
        hold_std = float(np.std(holdout[:, index]))
        feature_rows.append({
            "feature": name,
            "train_p1_p2": stats(train[:, index]),
            "holdout_p3": stats(holdout[:, index]),
            "smd_holdout_vs_train": float((np.mean(holdout[:, index]) - np.mean(train[:, index])) / max(train_std, 1e-8)),
            "abs_smd": float(abs((np.mean(holdout[:, index]) - np.mean(train[:, index])) / max(train_std, 1e-8))),
            "wasserstein_over_train_std": float(wasserstein_distance(train[:, index], holdout[:, index]) / max(train_std, 1e-8)),
            "holdout_outside_train_range_fraction": float(np.mean((holdout[:, index] < np.min(train[:, index])) | (holdout[:, index] > np.max(train[:, index])))),
        })
    model = joblib.load(V4_MODEL_DIR / "model.joblib")
    coefficient_model = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
    coefficients = np.asarray(getattr(coefficient_model, "coef_", np.empty((0, len(names)))), dtype=float)
    if coefficients.ndim == 2 and coefficients.shape[1] == len(names):
        importance = np.max(np.abs(coefficients), axis=0)
        coefficient_rows = [{"feature": names[i], "max_abs_coefficient": float(importance[i]), "coefficients": coefficients[:, i].tolist()} for i in np.argsort(-importance)]
    else:
        coefficient_rows = []
    participant_counts = manifest.groupby([manifest["participant_id"].astype(str), "mapped_project_label"]).size().unstack(fill_value=0).to_dict("index")
    video_counts = manifest.groupby([manifest["video_id"], "mapped_project_label"]).size().unstack(fill_value=0).to_dict("index")
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"rows": int(len(manifest)), "feature_family": config["name"], "features": len(names), "train_participants": ["1", "2"], "holdout_participant": ["3"]},
        "features": feature_rows,
        "top_coefficient_features": coefficient_rows[:15],
        "participant_label_counts": {str(k): {str(kk): int(vv) for kk, vv in value.items()} for k, value in participant_counts.items()},
        "video_label_counts": {str(k): {str(kk): int(vv) for kk, vv in value.items()} for k, value in video_counts.items()},
        "interpretation": [
            "This is a frozen-domain audit; it does not retrain V4 and does not select a new model.",
            "Participant 3 is a single held-out participant, so feature-shift estimates are descriptive, not population estimates.",
            "The per-participant and per-video label counts show whether source label and identity are separable confounds; they do not prove causality.",
        ],
    }
    write_json(OUT_DIR / "v4_feature_audit.json", result)
    return result


def integrity_audit() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in [RAW_MULTIPOSTURE, *sorted(RAW_ANYA_DIR.glob("*.csv")), *sorted(VIDEO_DIR.glob("*.mp4"))]:
        if path.exists():
            files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    expected = {}
    if ANYA_HASH_FILE.exists():
        for line in ANYA_HASH_FILE.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                expected[parts[1]] = parts[0]
    anya_checks = []
    for item in files:
        path = Path(item["path"])
        if path.parent == RAW_ANYA_DIR and path.name in expected:
            anya_checks.append({"filename": path.name, "matches_prior_hash": item["sha256"] == expected[path.name], "current_sha256": item["sha256"], "prior_sha256": expected[path.name]})
    disk = __import__("shutil").disk_usage(ROOT)
    memory = psutil.virtual_memory()
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.version,
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "mediapipe": mp.__version__, "sklearn": __import__("sklearn").__version__},
        "disk_c": {"free_bytes": disk.free, "free_gb_decimal": disk.free / 1e9, "total_bytes": disk.total},
        "ram": {"available_bytes": memory.available, "available_gb_decimal": memory.available / 1e9, "total_bytes": memory.total},
        "raw_files": files,
        "anya_prior_hash_checks": anya_checks,
        "raw_data_mutation_policy": "hash-only audit; no raw file is written, renamed, or deleted",
    }
    write_json(OUT_DIR / "reproducibility_audit.json", result)
    return result


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_reports(benchmark: dict[str, Any], audit: dict[str, Any], integrity: dict[str, Any]) -> None:
    rates = benchmark["aggregate_rates"]
    lines = [
        "# Overnight performance benchmark",
        "",
        "## Contract",
        "",
        "This report evaluates the frozen RGB-native V4 artifact on all six verified DriveNet videos. It is a same-source robustness/performance benchmark, not an independent external accuracy claim. All source frames were decoded; MediaPipe and V4 inference were run on a deterministic dense 10 FPS subsequence. 1/2/5 FPS rows are nested deterministic subsequences of that same pass.",
        "",
        "| Rate | Selected frames | Pose detected | Core accepted | Pose rate | Core acceptance | Raw switches | Smoothed switches | POSE_LOST |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rate in REPORT_RATES:
        item = rates[str(rate)]
        lines.append(f"| {rate:g} | {item['selected_frames']} | {item['pose_detected_frames']} | {item['core_accepted_frames']} | {100*item['pose_detection_rate']:.2f}% | {100*item['core_acceptance_rate']:.2f}% | {item['raw_prediction_switches']} | {item['smoothed_prediction_switches']} | {item['pose_lost_frames']} |")
    lines += ["", "## Per-video 5 FPS diagnostics", "", "| Video | Participant | Source/mapped label | Decoded | Selected | Pose | Accepted | Prediction distribution | Mean confidence | OOD features >50% |", "|---|---:|---|---:|---:|---:|---:|---|---:|---:|"]
    for video_id, item in benchmark["per_video"].items():
        five = item["rates"]["5.0"]
        lines.append(f"| `{video_id}` | {item['participant_id']} | {item['source_label']} / {item['mapped_project_label']} | {item['decoded_frame_count']} | {five['selected_frames']} | {five['pose_detected_frames']} | {five['core_accepted_frames']} | `{five['prediction_distribution']}` | {_fmt(five['mean_confidence'])} | {five['ood']['feature_fraction_over_50pct']} |")
    lines += ["", "## Performance and limits", "", f"- Dense pass wall time: `{benchmark['summary']['wall_seconds']:.2f}` seconds; decoded throughput: `{benchmark['summary']['decoded_frames']/max(benchmark['summary']['wall_seconds'], 1e-8):.2f}` frames/s.", f"- RSS: `{benchmark['summary']['rss_start_mb']:.1f}` MB → `{benchmark['summary']['rss_end_mb']:.1f}` MB (delta `{benchmark['summary']['rss_delta_mb']:.1f}` MB).", "- Video labels are from the same DriveNet source used to train V4; therefore prediction counts are diagnostics and are not reported as independent accuracy.", "- No webcam source 0 was opened. No YOLO package was installed. V1/V2/V3/V4 model artifacts were not overwritten.", "- Feature OOD detail and train-vs-held-out feature statistics are in `reports/v4_feature_audit.md` and the JSON artifact."]
    (REPORT_DIR / "overnight_performance_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    feature_lines = ["# V4 feature audit", "", "The frozen V4 training cache contains the six DriveNet videos sampled at 300 frames/video. This audit compares participants 1/2 (the declared training participants) with participant 3 (the declared holdout) without retraining.", "", "## Participant/video confound inventory", "", f"- Participant-label counts: `{audit['participant_label_counts']}`", f"- Video-label counts: `{audit['video_label_counts']}`", "- Every participant has both TUP and TLB source labels in the pilot, so participant identity is not perfectly identical to the class label. Nevertheless, frame-level observations are temporally correlated and the small number of videos limits the strength of this conclusion.", "", "## Shift table", "", "| Feature | Holdout abs SMD | Wasserstein / train std | Holdout outside train range | Train mean ± std | Holdout mean ± std |", "|---|---:|---:|---:|---:|---:|"]
    for row in sorted(audit["features"], key=lambda item: item["abs_smd"], reverse=True):
        tr, ho = row["train_p1_p2"], row["holdout_p3"]
        feature_lines.append(f"| `{row['feature']}` | {row['abs_smd']:.3f} | {row['wasserstein_over_train_std']:.3f} | {100*row['holdout_outside_train_range_fraction']:.1f}% | {_fmt(tr['mean'])} ± {_fmt(tr['std'])} | {_fmt(ho['mean'])} ± {_fmt(ho['std'])} |")
    feature_lines += ["", "## Top 15 frozen LogisticRegression coefficient magnitudes", "", "| Rank | Feature | Max absolute coefficient | Coefficients by class |", "|---:|---|---:|---|"]
    for rank, row in enumerate(audit["top_coefficient_features"], start=1):
        feature_lines.append(f"| {rank} | `{row['feature']}` | {row['max_abs_coefficient']:.5f} | `{[round(v, 5) for v in row['coefficients']]}` |")
    feature_lines += ["", "## Interpretation", "", "- This is descriptive domain-shift evidence, not a reason to clip or rescale the model input.", "- Coefficient magnitude indicates what the frozen linear V4 model uses in its own feature scale; it is not a causal importance measure.", "- The pilot has only three participants and six videos. A new RGB dataset with held-out people and camera views remains the strongest next experiment."]
    (REPORT_DIR / "v4_feature_audit.md").write_text("\n".join(feature_lines) + "\n", encoding="utf-8")

    integrity_lines = ["# Reproducibility and raw-data integrity audit", "", f"- Audit timestamp UTC: `{integrity['created_utc']}`", f"- Host: `{integrity['host']}`", f"- Python: `{integrity['python'].splitlines()[0]}`", f"- C: free space at audit: `{integrity['disk_c']['free_gb_decimal']:.2f} GB`", f"- Available RAM at audit: `{integrity['ram']['available_gb_decimal']:.2f} GB`", "", "## Raw-file hashes", "", "| File | Bytes | SHA-256 |", "|---|---:|---|"]
    for row in integrity["raw_files"]:
        integrity_lines.append(f"| `{Path(row['path']).name}` | {row['bytes']} | `{row['sha256']}` |")
    integrity_lines += ["", "## ANYA prior-hash checks", "", "| File | Matches prior hash |", "|---|---|"]
    for row in integrity["anya_prior_hash_checks"]:
        integrity_lines.append(f"| `{row['filename']}` | `{row['matches_prior_hash']}` |")
    integrity_lines += ["", "The audit only reads and hashes files. It does not modify, rename, delete, or replace anything under `datasets/raw/`. No new large external archive was downloaded because free space is below a comfortable margin for multi-gigabyte experiments."]
    (REPORT_DIR / "reproducibility_audit.md").write_text("\n".join(integrity_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen DriveNet V4 overnight benchmark and audit")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    integrity = integrity_audit()
    audit = feature_audit()
    if args.skip_benchmark and (OUT_DIR / "drivenet_v4_benchmark.json").exists():
        benchmark = json.loads((OUT_DIR / "drivenet_v4_benchmark.json").read_text(encoding="utf-8"))
    else:
        benchmark = benchmark_drivenet()
    write_reports(benchmark, audit, integrity)
    print(json.dumps({"benchmark": str(OUT_DIR / "drivenet_v4_benchmark.json"), "feature_audit": str(REPORT_DIR / "v4_feature_audit.md"), "integrity": str(REPORT_DIR / "reproducibility_audit.md"), "free_gb": integrity["disk_c"]["free_gb_decimal"]}, indent=2))


if __name__ == "__main__":
    main()
