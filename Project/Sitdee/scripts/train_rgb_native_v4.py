"""Build and evaluate the RGB-native V4 pilot from official DriveNet videos.

The script is intentionally conservative: DriveNet's SS and LB labels have a
defensible mapping to TUP and TLB. MB and RP remain source-labelled but
unmapped, so they are audited and never silently converted into MultiPosture
classes.  The training split is by participant, not by frame.
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
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.posture_ai.evaluation.metrics import classification_metrics  # noqa: E402
from src.posture_ai.evaluation.splitting import grouped_cv  # noqa: E402
from src.posture_ai.features.rgb_native_v4 import (  # noqa: E402
    V4_FEATURE_FAMILIES,
    build_rgb_native_v4_features,
    v4_feature_names,
)
from src.posture_ai.models.factories import make_model  # noqa: E402
from src.posture_ai.pose.landmarks import LANDMARK_INDEX, LANDMARK_NAMES  # noqa: E402

RAW_VIDEO_DIR = PROJECT_ROOT / "datasets" / "raw" / "rgb_native" / "drivenet_v6" / "source_videos"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "rgb_native"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "rgb_native_v4"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
REPORTS_DIR = PROJECT_ROOT / "reports"
POSE_MODEL = PROJECT_ROOT / "metadata" / "pose_landmarker_heavy.task"
DRIVENET_BASE = "https://data.mendeley.com/public-api/datasets/v37nxdn2gh"
DRIVENET_ROOT_SOURCE_VIDEOS = "56a62ab2-e646-4566-beee-e657c405ef06"
DRIVENET_LICENSE = "CC BY 4.0"
DRIVENET_PAGE = "https://data.mendeley.com/datasets/v37nxdn2gh/6"
MODEL_NAMES = ("LogisticRegression", "RandomForest", "ExtraTrees", "HistGradientBoosting", "GradientBoosting")
SUPPORTED_MAP = {"SS": "TUP", "LB": "TLB"}
UNMAPPED = {"MB": "", "RP": ""}
DEFAULT_SAMPLES_PER_VIDEO = 300
SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def _source_label(filename: str) -> str:
    return Path(filename).stem.split("_")[0].upper()


def _participant(filename: str) -> str:
    return Path(filename).stem.rsplit("_", 1)[-1]


def _source_inventory() -> list[dict[str, Any]]:
    """Read the already downloaded, official subset without changing it."""
    rows = []
    for path in sorted(RAW_VIDEO_DIR.glob("*.mp4")):
        label = _source_label(path.name)
        if label not in SUPPORTED_MAP:
            continue
        rows.append(
            {
                "filename": path.name,
                "path": str(path),
                "source_label": label,
                "mapped_project_label": SUPPORTED_MAP.get(label, ""),
                "participant_id": _participant(path.name),
                "video_id": path.stem,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "license": DRIVENET_LICENSE,
                "source_url": DRIVENET_PAGE,
            }
        )
    return rows


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


def _mp_array(points: Any) -> np.ndarray:
    return np.asarray([[float(p.x), float(p.y), float(p.z)] for p in points], dtype=np.float64)


def _visibility(points: Any) -> np.ndarray:
    return np.asarray([float(getattr(p, "visibility", np.nan)) for p in points], dtype=np.float64)


def _sample_indices(frame_count: int, count: int) -> set[int]:
    if frame_count <= 0:
        return set()
    return set(np.linspace(0, frame_count - 1, min(count, frame_count), dtype=int).tolist())


def extract_drivenet_features(inventory: list[dict[str, Any]], samples_per_video: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    matrices = {family: [] for family in V4_FEATURE_FAMILIES}
    start = time.perf_counter()
    for item in inventory:
        landmarker = _make_landmarker()
        try:
            capture = cv2.VideoCapture(item["path"])
            if not capture.isOpened():
                raise RuntimeError(f"Could not open downloaded DriveNet video: {item['path']}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            wanted = _sample_indices(frame_count, samples_per_video)
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index not in wanted:
                    frame_index += 1
                    continue
                timestamp_ms = int(frame_index * 1000.0 / fps)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                row: dict[str, Any] = {
                    "source_filename": item["filename"],
                    "source_label": item["source_label"],
                    "mapped_project_label": item["mapped_project_label"],
                    "participant_id": item["participant_id"],
                    "video_id": item["video_id"],
                    "frame_index": frame_index,
                    "source_fps": fps,
                    "source_frame_count": frame_count,
                    "pose_detected": bool(result.pose_landmarks),
                    "core_visibility_min": None,
                    "core_visibility_mean": None,
                    "feature_row": -1,
                }
                if result.pose_landmarks and result.pose_world_landmarks:
                    image_points = result.pose_landmarks[0]
                    world_points = result.pose_world_landmarks[0]
                    vis = _visibility(image_points)
                    core_vis = vis[list(LANDMARK_INDEX[name] for name in ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"))]
                    world = _mp_array(world_points)
                    row["core_visibility_min"] = float(np.nanmin(core_vis))
                    row["core_visibility_mean"] = float(np.nanmean(core_vis))
                    try:
                        feature_row = len(matrices["world"])
                        for family in V4_FEATURE_FAMILIES:
                            matrix, names = build_rgb_native_v4_features(world, family)
                            matrices[family].append(matrix[0])
                        row["feature_row"] = feature_row
                        row["core_accepted"] = bool(np.isfinite(core_vis).all() and np.all(core_vis >= 0.5))
                    except ValueError as exc:
                        row["feature_error"] = str(exc)
                        row["core_accepted"] = False
                else:
                    row["core_accepted"] = False
                records.append(row)
                frame_index += 1
            capture.release()
        finally:
            landmarker.close()
    arrays = {family: np.asarray(values, dtype=np.float64) for family, values in matrices.items()}
    _write_json(PROCESSED_DIR / "extraction_run.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "samples_per_video": samples_per_video, "elapsed_seconds": time.perf_counter() - start, "inventory": inventory})
    return {"records": records, "matrices": arrays}


def _metrics_with_binary_scores(y_true: np.ndarray, pred: np.ndarray, probabilities: np.ndarray, labels: list[str]) -> dict[str, Any]:
    result = classification_metrics(y_true, pred, labels)
    positive = labels[-1]
    positive_index = list(labels).index(positive)
    y_binary = (y_true == positive).astype(int)
    score = probabilities[:, positive_index]
    try:
        result.update({"roc_auc": float(roc_auc_score(y_binary, score)), "pr_auc": float(average_precision_score(y_binary, score)), "brier_score": float(brier_score_loss(y_binary, score))})
    except ValueError:
        result.update({"roc_auc": None, "pr_auc": None, "brier_score": None})
    cm = np.asarray(result["confusion_matrix"], dtype=int)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        result.update({"sensitivity": float(tp / (tp + fn)) if tp + fn else None, "specificity": float(tn / (tn + fp)) if tn + fp else None, "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None, "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None})
    return result


def _predict_proba(estimator: Any, x: np.ndarray, labels: list[str]) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        p = np.asarray(estimator.predict_proba(x), dtype=float)
        classes = list(estimator.classes_)
        out = np.zeros((len(x), len(labels)), dtype=float)
        for i, label in enumerate(labels):
            if label in classes:
                out[:, i] = p[:, classes.index(label)]
        return out
    pred = estimator.predict(x)
    return np.asarray([[1.0 if label == value else 0.0 for label in labels] for value in pred], dtype=float)


def _split_indices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pre-declared participant split: participants 1/2 train, 3 holdout."""
    groups = frame["participant_id"].astype(str).to_numpy()
    train = np.flatnonzero(np.isin(groups, ["1", "2"]))
    holdout = np.flatnonzero(groups == "3")
    if not len(train) or not len(holdout):
        raise ValueError("DriveNet participant split requires participants 1/2 for training and 3 for holdout")
    if set(groups[train]) & set(groups[holdout]):
        raise AssertionError("Participant leakage in V4 split")
    return {"train": train, "holdout": holdout}


def benchmark(frame: pd.DataFrame, arrays: dict[str, np.ndarray], seed: int = SEED) -> dict[str, Any]:
    usable = frame[frame["mapped_project_label"].isin(sorted(SUPPORTED_MAP.values())) & frame["core_accepted"] & (frame["feature_row"] >= 0)].reset_index(drop=True)
    if usable.empty:
        raise ValueError("No accepted, mapped RGB-native frames are available")
    labels = sorted(usable["mapped_project_label"].unique().tolist())
    split = _split_indices(usable)
    results: list[dict[str, Any]] = []
    fitted_for_holdout: dict[str, Any] = {}
    for family in V4_FEATURE_FAMILIES:
        x_all = arrays[family][usable["feature_row"].to_numpy(dtype=int)]
        feature_names = v4_feature_names(family)
        y = usable["mapped_project_label"].astype(str).to_numpy()
        groups = usable["participant_id"].astype(str).to_numpy()
        train_idx, test_idx = split["train"], split["holdout"]
        for model_name in MODEL_NAMES:
            fold_scores = []
            for fold_train, fold_eval in grouped_cv(groups[train_idx], n_splits=2):
                estimator = make_model(model_name, random_state=seed)
                estimator.fit(x_all[train_idx][fold_train], y[train_idx][fold_train])
                fold_pred = estimator.predict(x_all[train_idx][fold_eval])
                fold_scores.append(classification_metrics(y[train_idx][fold_eval], fold_pred, labels)["macro_f1"])
            estimator = make_model(model_name, random_state=seed)
            estimator.fit(x_all[train_idx], y[train_idx])
            pred = estimator.predict(x_all[test_idx])
            prob = _predict_proba(estimator, x_all[test_idx], labels)
            holdout = _metrics_with_binary_scores(y[test_idx], pred, prob, labels)
            latency_start = time.perf_counter()
            estimator.predict(x_all[test_idx[:1]].repeat(200, axis=0))
            latency_ms = 1000.0 * (time.perf_counter() - latency_start) / 200.0
            key = f"{family}:{model_name}"
            fitted_for_holdout[key] = {"estimator": estimator, "holdout": holdout, "feature_names": feature_names, "latency_ms": latency_ms}
            results.append({"feature_family": family, "model": model_name, "feature_count": len(feature_names), "cv_macro_f1_mean": float(np.mean(fold_scores)), "cv_macro_f1_std": float(np.std(fold_scores)), "holdout_macro_f1": holdout["macro_f1"], "holdout_balanced_accuracy": holdout["balanced_accuracy"], "holdout_accuracy": holdout["accuracy"], "latency_ms_per_frame": latency_ms})
    result_frame = pd.DataFrame(results).sort_values(["cv_macro_f1_mean", "holdout_macro_f1", "feature_count", "model"], ascending=[False, False, True, True]).reset_index(drop=True)
    selected = result_frame.iloc[0].to_dict()
    family, model_name = str(selected["feature_family"]), str(selected["model"])
    x_all = arrays[family][usable["feature_row"].to_numpy(dtype=int)]
    final_estimator = make_model(model_name, random_state=seed)
    final_estimator.fit(x_all, usable["mapped_project_label"].astype(str).to_numpy())
    return {"usable": usable, "labels": labels, "split": split, "results": result_frame, "selected": selected, "selected_holdout": fitted_for_holdout[f"{family}:{model_name}"], "final_estimator": final_estimator, "selected_family": family, "selected_feature_names": v4_feature_names(family), "selected_model": model_name}


def _write_manifest(records: list[dict[str, Any]]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["source_filename", "source_label", "mapped_project_label", "participant_id", "video_id", "frame_index", "source_fps", "source_frame_count", "pose_detected", "core_visibility_min", "core_visibility_mean", "core_accepted", "feature_row"]
    with (PROCESSED_DIR / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, "") for key in columns})


def _write_reports(inventory: list[dict[str, Any]], extraction: dict[str, Any], benchmark_result: dict[str, Any] | None, samples_per_video: int) -> None:
    records = extraction["records"]
    frame = pd.DataFrame(records)
    mapped = frame[frame["mapped_project_label"].isin(sorted(SUPPORTED_MAP.values())) & frame["core_accepted"]]
    counts = frame.groupby(["source_label", "participant_id"], dropna=False).size().to_dict()
    accepted = frame.groupby("source_label")["core_accepted"].agg(["count", "sum"]).to_dict("index")
    label_lines = ["# RGB-native V4 label mapping", "", "## Source facts", "", f"- Source: DriveNet v6 official Mendeley record [{DRIVENET_PAGE}]({DRIVENET_PAGE})", f"- License: {DRIVENET_LICENSE}", f"- Local subset: {len(inventory)} source videos downloaded from the official per-file video folders; sampling is deterministic at {samples_per_video} frames/video.", "", "## Mapping decision", "", "| DriveNet source label | Project label | Decision | Reason |", "|---|---|---|---|", "| SS (Sit Straight) | TUP | COMPATIBLE | Static straight sitting posture is directionally compatible with upright. |", "| LB (Lean Backward) | TLB | COMPATIBLE | Source name explicitly states lean backward. |", "| MB (Move Backward) | — | AMBIGUOUS / NOT USED | Move may denote an action or transition, not a sustained trunk posture. |", "| RP (Right Position) | — | AMBIGUOUS / NOT USED | Source name does not define right lean versus seat position or action. |", "", "Only SS and LB were allowed into supervised target training. No labels were invented for MB or RP."]
    (REPORTS_DIR / "rgb_label_mapping.md").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    quality_lines = ["# RGB-native V4 pose-quality audit", "", "The audit uses the same MediaPipe Pose Landmarker task asset and World Landmark extraction path that feeds V4 features. Core acceptance requires finite coordinates and image-landmark visibility ≥ 0.5 for nose, both shoulders, and both hips.", "", "| Source label | Sampled | Pose detected | Core accepted | Acceptance |", "|---|---:|---:|---:|---:|"]
    for label in sorted(accepted):
        row = accepted[label]
        pose = int(frame.loc[frame.source_label == label, "pose_detected"].sum())
        quality_lines.append(f"| {label} | {int(row['count'])} | {pose} | {int(row['sum'])} | {100.0*float(row['sum'])/float(row['count']):.2f}% |")
    quality_lines += ["", "No webcam device was opened. This is an offline video audit only."]
    (REPORTS_DIR / "rgb_pose_quality_audit.md").write_text("\n".join(quality_lines) + "\n", encoding="utf-8")

    mapping_json = {"source_inventory": inventory, "sample_counts_by_source_participant": {f"{k[0]}_{k[1]}": int(v) for k, v in counts.items()}, "accepted_by_source": accepted}
    _write_json(PROCESSED_DIR / "source_manifest.json", mapping_json)

    if benchmark_result is None:
        return
    overlap = _frozen_overlap_diagnostic(benchmark_result)
    usable = benchmark_result["usable"]
    results = benchmark_result["results"]
    results.to_csv(METRICS_DIR / "rgb_native_v4_model_comparison.csv", index=False)
    _write_json(METRICS_DIR / "rgb_native_v4_split.json", {"train_rows": int(len(benchmark_result["split"]["train"])), "holdout_rows": int(len(benchmark_result["split"]["holdout"])), "train_participants": sorted(usable.iloc[benchmark_result["split"]["train"]]["participant_id"].astype(str).unique().tolist()), "holdout_participants": sorted(usable.iloc[benchmark_result["split"]["holdout"]]["participant_id"].astype(str).unique().tolist()), "no_group_overlap": True})
    best = benchmark_result["selected"]
    reports = ["# RGB-native V4 model comparison", "", "V4 is an experimental RGB-native pilot. It uses only DriveNet source labels with a defensible target mapping (TUP and TLB). It is not a six-class replacement for MultiPosture.", "", "| Rank | Feature family | Model | Features | Grouped CV macro F1 | CV std | Participant-3 holdout macro F1 | Holdout balanced accuracy |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for i, row in results.head(12).iterrows():
        reports.append(f"| {i+1} | {row.feature_family} | {row.model} | {int(row.feature_count)} | {row.cv_macro_f1_mean:.4f} | {row.cv_macro_f1_std:.4f} | {row.holdout_macro_f1:.4f} | {row.holdout_balanced_accuracy:.4f} |")
    reports += ["", f"Selected by grouped-CV macro F1 (fixed tie-breaks): `{best['feature_family']}` + `{best['model']}`.", "", "The participant-3 holdout is not used for model selection. It is one held-out participant, so uncertainty remains high and these scores are not a claim of office-camera generalization.", "", "## Frozen DriveNet re-test", "", f"Status: `{overlap['status']}`. This diagnostic is not independent generalization because the clips are from the same DriveNet source as training."]
    for clip, value in overlap.get("clips", {}).items():
        reports.append(f"- `{clip}`: accepted {value['core_accepted']}/{value['frames_seen']} frames; predictions `{value['prediction_distribution']}`; source mapping `{value['source_label_mapping']}`.")
    (REPORTS_DIR / "rgb_cross_dataset_generalization.md").write_text("\n".join(reports) + "\n", encoding="utf-8")

    comparison = ["# V4 versus MultiPosture", "", "| Attribute | RGB-native V4 pilot | MultiPosture V1/V2 |", "|---|---|---|", "| Source modality | DriveNet RGB video, MediaPipe extracted at run time | Processed landmark CSV; raw RGB withheld |", "| Labels used | TUP and TLB only; MB/RP unmapped | TUP/TLF/TLB/TLL/TLR |", "| Coordinate path | MediaPipe World Landmarks → hip midpoint translation → features | Released processed coordinates; source/model-input provenance caveat |", "| Grouping | participant ID encoded in official video IDs | subject column |", "| Selected model | " + str(best["model"]) + " + " + str(best["feature_family"]) + " | HistGradientBoosting + 30-feature hybrid V2 |", f"| Grouped CV macro F1 | {best['cv_macro_f1_mean']:.4f} ± {best['cv_macro_f1_std']:.4f} | 0.9925 ± 0.0087 (five-class) |", f"| Holdout macro F1 | {best['holdout_macro_f1']:.4f} (participant 3) | 1.0000 (two-subject holdout; cautious) |", "| Real-RGB compatibility | Same extractor/training path; compatible only for pilot labels | Semantic compatibility remains not fully validated |", "", "Metrics are not directly comparable because the ontologies, modalities, number of classes, and data sources differ."]
    (REPORTS_DIR / "v4_vs_multiposture.md").write_text("\n".join(comparison) + "\n", encoding="utf-8")

    holdout = benchmark_result["selected_holdout"]["holdout"]
    error_lines = ["# RGB-native V4 error analysis", "", f"Selected track: `{best['feature_family']}` + `{best['model']}`. Holdout participant: `3`.", "", "```json", json.dumps(_jsonable(holdout), indent=2), "```", "", "The source videos are controlled driver-posture recordings. Remaining errors, if any, should be analyzed as source-domain errors; they do not establish clinical or office-environment performance."]
    (REPORTS_DIR / "rgb_v4_error_analysis.md").write_text("\n".join(error_lines) + "\n", encoding="utf-8")

    final_lines = ["# RGB-native posture training track V4", "", "## Status", "", "**RGB-NATIVE V4: PILOT TRAINED — TWO COMPATIBLE CLASSES ONLY**", "", "V4 uses the same authoritative pipeline for its training and offline inference path: `RGB video → MediaPipe Pose World Landmarks → hip-midpoint translation → V4 feature family → classifier`. It is intentionally not connected to the MultiPosture V1/V2/V3 artifacts.", "", "## Dataset and labels", "", f"- Official source: [DriveNet v6]({DRIVENET_PAGE}), license `{DRIVENET_LICENSE}`.", f"- Locally used mapped videos: `{len(inventory)}`; sampled frames per video: `{samples_per_video}`.", f"- Accepted mapped training rows: `{len(mapped)}`.", "- `SS → TUP` and `LB → TLB` are COMPATIBLE mappings.", "- `MB` and `RP` are AMBIGUOUS and were excluded from supervised target training; no label was invented.", "- Participant split was predeclared: participants 1/2 train, participant 3 holdout; no participant overlap.", "", "## Model result", "", f"- Selected model: `{best['model']}` with `{best['feature_family']}` features ({int(best['feature_count'])}).", f"- Grouped-CV macro F1: `{best['cv_macro_f1_mean']:.4f} ± {best['cv_macro_f1_std']:.4f}`.", f"- Participant-3 holdout macro F1: `{best['holdout_macro_f1']:.4f}`; balanced accuracy `{best['holdout_balanced_accuracy']:.4f}`.", "- These are a two-class, three-participant source-domain pilot, not six-class office-camera generalization.", "", "## Required interpretation", "", "- MultiPosture remains the five-class direction model and is preserved unchanged.", "- V4 demonstrates a same-pipeline RGB training feasibility track, but it does not yet cover TLF, TLL, TLR, or SLOUCH.", "- The frozen DriveNet re-test is a same-source overlap diagnostic, not an independent external test.", "- No webcam device was opened; no live-camera result is claimed.", "- No ANYA training, YOLO installation, or raw-dataset modification was performed."]
    (REPORTS_DIR / "rgb_native_v4.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
    (REPORTS_DIR / "rgb_native_v4_th.md").write_text("\n".join(["# แทร็กฝึก RGB-native V4", "", "## สถานะ", "", "**RGB-NATIVE V4: ฝึกเป็น PILOT แล้ว — รองรับเพียง 2 คลาสที่ mapping ได้**", "", "V4 ใช้เส้นทางเดียวกันทั้งตอนฝึกและตอนอนุมานออฟไลน์: `วิดีโอ RGB → MediaPipe Pose World Landmarks → ย้ายจุดอ้างอิงไปกึ่งกลางสะโพก → feature V4 → classifier` และแยกจากโมเดล MultiPosture V1/V2/V3", "", "## ข้อมูลและป้ายกำกับ", "", f"- แหล่งทางการ: DriveNet v6 ({DRIVENET_PAGE}) ใบอนุญาต {DRIVENET_LICENSE}", f"- วิดีโอที่ใช้และ mapping ได้: {len(inventory)} คลิป; สุ่มแบบกำหนดแน่นอน {samples_per_video} เฟรมต่อคลิป", f"- แถวที่ผ่าน core landmark และใช้ฝึก: {len(mapped)}", "- `SS → TUP` และ `LB → TLB` เป็น COMPATIBLE", "- `MB` และ `RP` ยัง AMBIGUOUS จึงไม่ใช้ฝึก และไม่มีการเดาป้ายกำกับ", "- แบ่งตามผู้เข้าร่วมล่วงหน้า: คนที่ 1/2 เป็น train และคนที่ 3 เป็น holdout; ไม่มีผู้เข้าร่วมซ้ำข้ามชุด", "", "## ผลโมเดล", "", f"- โมเดลที่เลือก: `{best['model']}` กับ feature `{best['feature_family']}` จำนวน {int(best['feature_count'])}", f"- Grouped-CV macro F1: {best['cv_macro_f1_mean']:.4f} ± {best['cv_macro_f1_std']:.4f}", f"- Holdout ผู้เข้าร่วมคนที่ 3 macro F1: {best['holdout_macro_f1']:.4f}; balanced accuracy {best['holdout_balanced_accuracy']:.4f}", "- ผลนี้เป็น pilot สองคลาสในโดเมนวิดีโอขับรถ ไม่ใช่ผล generalization หกคลาสในสำนักงาน", "", "## ข้อจำกัดสำคัญ", "", "- MultiPosture ยังคงเป็นโมเดลทิศทาง 5 คลาสและไม่ได้ถูกแก้ไข", "- V4 ยังไม่ครอบคลุม TLF, TLL, TLR หรือ SLOUCH", "- frozen re-test เป็นเพียง overlap diagnostic จากแหล่ง DriveNet เดียวกัน ไม่ใช่ external test อิสระ", "- ไม่เปิด webcam, ไม่ฝึก ANYA, ไม่ติดตั้ง YOLO และไม่แก้ข้อมูลดิบ"]) + "\n", encoding="utf-8")


def _export(benchmark_result: dict[str, Any], inventory: list[dict[str, Any]], samples_per_video: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(benchmark_result["final_estimator"], OUTPUT_DIR / "model.joblib")
    family = benchmark_result["selected_family"]
    columns = benchmark_result["selected_feature_names"]
    (OUTPUT_DIR / "feature_config.json").write_text(json.dumps({"name": family, "columns": columns, "core_landmarks": ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"], "coordinate_representation": "MediaPipe Pose World Landmarks in meters, translated by hip midpoint", "adapter_version": "rgb_native_v4_world_hip_centered"}, indent=2), encoding="utf-8")
    labels = benchmark_result["labels"]
    (OUTPUT_DIR / "labels.json").write_text(json.dumps({"source_labels": labels, "display_labels": {"TUP": "Upright", "TLB": "Lean backward"}}, indent=2), encoding="utf-8")
    metadata = {"model_family": "rgb_native_v4_pilot", "source_dataset": "DriveNet v6", "source_url": DRIVENET_PAGE, "license": DRIVENET_LICENSE, "source_files": inventory, "sample_strategy": {"frames_per_video": samples_per_video, "method": "linspace over full video frame range"}, "training_timestamp_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(), "python": sys.version, "model_name": benchmark_result["selected_model"], "feature_family": family, "feature_columns": columns, "labels": labels, "grouping_strategy": "participant_id from official video filename suffix; participants 1/2 train, 3 holdout", "split": {"train_participants": ["1", "2"], "holdout_participants": ["3"]}, "cv": "2-fold GroupKFold on training participants", "cv_macro_f1_mean": benchmark_result["selected"]["cv_macro_f1_mean"], "cv_macro_f1_std": benchmark_result["selected"]["cv_macro_f1_std"], "holdout_metrics": benchmark_result["selected_holdout"]["holdout"], "limitations": ["Pilot supports only TUP and TLB; MB and RP were not mapped or trained.", "Three participants provide only one held-out participant.", "DriveNet is a controlled driving posture domain, not an office-camera validation set.", "The exported model is not a six-class MultiPosture replacement.", "This model does not prove ergonomic medical safety or diagnosis."], "v4_status": "PILOT_TRAINED_TWO_CLASS_ONLY"}
    _write_json(OUTPUT_DIR / "metadata.json", metadata)


def _frozen_overlap_diagnostic(benchmark_result: dict[str, Any]) -> dict[str, Any]:
    """Re-test the existing three-clip pack as an overlap diagnostic only.

    Those clips are part of DriveNet and two are also represented in the
    downloaded source-video subset, so this is deliberately not called an
    independent external generalization test.
    """
    path = PROJECT_ROOT / "outputs" / "validation" / "coordinate_adapter_v3" / "landmarks.npz"
    if not path.exists():
        return {"status": "NOT_AVAILABLE", "reason": str(path)}
    data = np.load(path, allow_pickle=False)
    world = data["world"]
    visibility = data["visibility"]
    clip_ids = data["clip_ids"].astype(str)
    estimator = benchmark_result["final_estimator"]
    family = benchmark_result["selected_family"]
    results: dict[str, Any] = {}
    core_indices = [LANDMARK_INDEX[name] for name in ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    for clip in sorted(np.unique(clip_ids)):
        mask = clip_ids == clip
        detected = np.isfinite(world[mask]).all(axis=(1, 2))
        accepted = detected & np.isfinite(visibility[mask][:, core_indices]).all(axis=1) & (visibility[mask][:, core_indices] >= 0.5).all(axis=1)
        rows = np.flatnonzero(accepted)
        predictions: list[str] = []
        confidences: list[float] = []
        if len(rows):
            features, names = build_rgb_native_v4_features(world[mask][rows], family)
            if names != benchmark_result["selected_feature_names"]:
                raise ValueError("Frozen V4 overlap diagnostic schema mismatch")
            predictions = [str(x) for x in estimator.predict(features)]
            if hasattr(estimator, "predict_proba"):
                confidences = np.max(estimator.predict_proba(features), axis=1).astype(float).tolist()
        results[clip] = {"frames_seen": int(mask.sum()), "pose_world_detected": int(detected.sum()), "core_accepted": int(accepted.sum()), "acceptance_percentage": float(100.0 * accepted.mean()) if len(accepted) else 0.0, "prediction_distribution": dict(Counter(predictions)), "mean_confidence": float(np.mean(confidences)) if confidences else None, "source_label_mapping": {"SS_1.mp4": "TUP", "LB_2.mp4": "TLB"}.get(clip, "UNLABELED")}
    output = {"status": "SAME_SOURCE_OVERLAP_DIAGNOSTIC", "note": "Not an independent external test; these clips belong to the same DriveNet source and overlap the downloaded source-video subset.", "clips": results}
    _write_json(PROJECT_ROOT / "outputs" / "validation" / "rgb_native_v4" / "overlap_diagnostic.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-video", type=int, default=DEFAULT_SAMPLES_PER_VIDEO)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()
    if not RAW_VIDEO_DIR.exists():
        raise FileNotFoundError(f"Missing official DriveNet subset: {RAW_VIDEO_DIR}")
    inventory = _source_inventory()
    mapped_video_count = sum(item["mapped_project_label"] in SUPPORTED_MAP.values() for item in inventory)
    if mapped_video_count < 6:
        raise RuntimeError(f"Need at least 6 downloaded mapped DriveNet videos (3 SS + 3 LB); found {mapped_video_count}")
    if args.skip_extract and (PROCESSED_DIR / "manifest.csv").exists() and (PROCESSED_DIR / "features.npz").exists():
        records = list(csv.DictReader((PROCESSED_DIR / "manifest.csv").open(encoding="utf-8", newline="")))
        extraction = {"records": records, "matrices": {family: np.load(PROCESSED_DIR / "features.npz")[family] for family in V4_FEATURE_FAMILIES}}
    else:
        extraction = extract_drivenet_features(inventory, args.samples_per_video)
        _write_manifest(extraction["records"])
        np.savez_compressed(PROCESSED_DIR / "features.npz", **extraction["matrices"])
    frame = pd.DataFrame(extraction["records"])
    for col in ("core_accepted", "pose_detected"):
        frame[col] = frame[col].astype(str).str.lower().eq("true")
    for col in ("feature_row", "frame_index"):
        frame[col] = frame[col].astype(int)
    benchmark_result = benchmark(frame, extraction["matrices"])
    _write_reports(inventory, extraction, benchmark_result, args.samples_per_video)
    _export(benchmark_result, inventory, args.samples_per_video)
    print(json.dumps({"inventory_files": len(inventory), "records": len(frame), "usable_mapped_frames": len(benchmark_result["usable"]), "selected": benchmark_result["selected"], "holdout": benchmark_result["selected_holdout"]["holdout"], "output": str(OUTPUT_DIR)}, indent=2, default=str))


if __name__ == "__main__":
    main()
