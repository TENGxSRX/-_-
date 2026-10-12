"""Audit and benchmark a small, openly licensed external RGB validation pack.

The pack is intentionally tiny and source-controlled by an explicit manifest.
Only labels that are semantically equivalent to the MultiPosture ontology are
used for accuracy.  All other media are smoke-test inputs and remain
UNLABELED for project evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.posture_ai.features.pose_features import build_features, expected_feature_columns  # noqa: E402
from src.posture_ai.inference.pipeline import extract_pose_landmarks  # noqa: E402
from src.posture_ai.inference.pipeline import InferenceRunner  # noqa: E402
from src.posture_ai.pose.adapter import adapt_mediapipe_to_multiposture, multiposture_feature_names  # noqa: E402
from src.posture_ai.pose.landmarks import PoseLandmarks  # noqa: E402
from src.posture_ai.pose.validation import PoseValidationError, validate_pose_landmarks  # noqa: E402


SOURCE_DATASET = "DriveNet"
SOURCE_VERSION = "6"
SOURCE_URL = "https://data.mendeley.com/datasets/v37nxdn2gh/6"
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
SOURCE_DOWNLOAD_BASE = "https://data.mendeley.com/public-files/datasets/v37nxdn2gh/files"
POSE_MODEL = PROJECT_ROOT / "metadata" / "pose_landmarker_heavy.task"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "multiposture_v1"
TRAINING_DATA = PROJECT_ROOT / "datasets" / "raw" / "multiposture" / "data.csv"
EXTERNAL_ROOT = PROJECT_ROOT / "validation_data" / "external"
MANIFEST_PATH = EXTERNAL_ROOT / "external_manifest.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "external_validation_media_search.md"
REPORT_TH_PATH = PROJECT_ROOT / "reports" / "external_validation_media_search_th.md"
DOMAIN_REPORT_PATH = PROJECT_ROOT / "reports" / "external_rgb_domain_shift.md"
DOMAIN_JSON_PATH = PROJECT_ROOT / "outputs" / "validation" / "external" / "domain_shift.json"
AUDIT_JSON_PATH = PROJECT_ROOT / "outputs" / "validation" / "external" / "external_media_audit.json"

PROJECT_LABELS = {"TUP", "TLF", "TLB", "TLL", "TLR", "SLOUCH", "UNLABELED"}

MEDIA_ROWS = [
    {
        "local_path": "validation_data/external/labeled/upright/DriveNet_v6_SS_1.mp4",
        "source_dataset": SOURCE_DATASET,
        "source_url": SOURCE_URL,
        "source_file": "SS_1.mp4",
        "media_type": "video",
        "verified_label": "TUP",
        "label_source": "DriveNet v6 class folder Sit Straight (SS) and meta_data.csv",
        "license": LICENSE,
        "participant_id_if_available": "1",
        "session_id_if_available": "SS_1 (video_source)",
        "camera_view": "side/three-quarter driver view (observed; not a source metadata field)",
        "notes": "Source class SS; semantically mapped to MultiPosture TUP. Process first 300 frames (~10 s) only; source video retained unchanged.",
        "source_class": "SS",
        "source_file_sha256": "f4161e3bb9faf61831422e0d88c253f9746423a86be1deac8f997212a4426f79",
        "download_file_id": "ade85ab4-6073-49ff-97c0-a58c7ab100b0",
    },
    {
        "local_path": "validation_data/external/labeled/lean_backward/DriveNet_v6_LB_2.mp4",
        "source_dataset": SOURCE_DATASET,
        "source_url": SOURCE_URL,
        "source_file": "LB_2.mp4",
        "media_type": "video",
        "verified_label": "TLB",
        "label_source": "DriveNet v6 class folder Lean Backward (LB) and meta_data.csv",
        "license": LICENSE,
        "participant_id_if_available": "2",
        "session_id_if_available": "LB_2 (video_source)",
        "camera_view": "side/three-quarter driver view (observed; not a source metadata field)",
        "notes": "Source class LB; semantically mapped to MultiPosture TLB. Process first 300 frames (~10 s) only; source video retained unchanged.",
        "source_class": "LB",
        "source_file_sha256": "baa2dbb0c1d4dec761389ab4941114e27c61928ac8132c6eae0abcd84ce677ac",
        "download_file_id": "c247db1c-3e3c-4100-b73b-6a2a5164fb1b",
    },
    {
        "local_path": "validation_data/external/unlabeled_smoke/DriveNet_v6_MB_1.mp4",
        "source_dataset": SOURCE_DATASET,
        "source_url": SOURCE_URL,
        "source_file": "MB_1.mp4",
        "media_type": "video",
        "verified_label": "UNLABELED",
        "label_source": "DriveNet v6 source class MB recorded in meta_data.csv; no project-ontology mapping claimed",
        "license": LICENSE,
        "participant_id_if_available": "1",
        "session_id_if_available": "MB_1 (video_source)",
        "camera_view": "side/three-quarter driver view (observed; not a source metadata field)",
        "notes": "Smoke-only. Source class Move Backward is not assumed equivalent to TLB, TLF, or another project class. Process first 300 frames (~10 s).",
        "source_class": "MB",
        "source_file_sha256": "51c8b6790f0094672779fed1fa7e3cadeea1911b04e5256ceff74fe6ef40c34f",
        "download_file_id": "41b332cb-3526-4b35-9057-01b8606e5a48",
    },
]

MANIFEST_FIELDS = [
    "local_path",
    "source_dataset",
    "source_url",
    "source_file",
    "media_type",
    "verified_label",
    "label_source",
    "license",
    "participant_id_if_available",
    "session_id_if_available",
    "camera_view",
    "notes",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest() -> None:
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in MANIFEST_FIELDS} for row in MEDIA_ROWS)


def _landmarker() -> PoseLandmarker:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def _video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"open_ok": False, "error": "cv2.VideoCapture could not open file"}
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
        return {
            "open_ok": True,
            "frame_count_reported": frame_count,
            "fps_reported": fps,
            "width": width,
            "height": height,
            "duration_seconds_reported": frame_count / fps if fps > 0 else None,
            "fourcc": fourcc,
        }
    finally:
        capture.release()


def audit_video(row: dict[str, str], *, max_frames: int = 300) -> tuple[dict[str, Any], list[np.ndarray]]:
    path = PROJECT_ROOT / row["local_path"]
    summary: dict[str, Any] = {
        "local_path": row["local_path"],
        "source_file": row["source_file"],
        "source_class": row.get("source_class"),
        "verified_label": row["verified_label"],
        "sha256_expected": row.get("source_file_sha256"),
        "sha256_actual": sha256_file(path) if path.exists() else None,
        "size_bytes": path.stat().st_size if path.exists() else None,
        "file_exists": path.exists(),
        "frame_limit_processed": max_frames,
        "pose_detected_frames": 0,
        "adapter_eligible_frames": 0,
        "processed_frames": 0,
        "low_visibility_frames": 0,
        "missing_or_invalid_frames": 0,
        "mean_visibility_detected": None,
        "mean_visible_landmark_fraction_at_0_5": None,
        "low_visibility_landmarks": {},
        "predictions": {},
        "confidence_mean": None,
        "confidence_p50": None,
        "confidence_min": None,
        "confidence_max": None,
        "accuracy_available": False,
        "accuracy_note": "No accuracy unless verified label has a safe semantic mapping and at least one adapter-eligible frame.",
    }
    summary.update(_video_metadata(path))
    if not path.exists() or not summary.get("open_ok"):
        return summary, []

    runner = InferenceRunner(MODEL_DIR)
    confidence_values: list[float] = []
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    feature_rows: list[np.ndarray] = []
    visibility_values: list[float] = []
    visible_fractions: list[float] = []
    low_visibility_counter: Counter[str] = Counter()
    capture = cv2.VideoCapture(str(path))
    landmarker = _landmarker()
    fps = float(summary.get("fps_reported") or 30.0)
    try:
        for frame_index in range(max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            summary["processed_frames"] += 1
            landmarks = extract_pose_landmarks(frame, landmarker, timestamp_ms=int(frame_index * 1000.0 / fps))
            if landmarks is None:
                continue
            summary["pose_detected_frames"] += 1
            pose = PoseLandmarks.from_any(landmarks)
            visibility = pose.visibility_array()
            finite_visibility = visibility[np.isfinite(visibility)]
            if finite_visibility.size:
                visibility_values.extend(float(value) for value in finite_visibility)
                visible_fractions.append(float(np.mean(finite_visibility >= 0.5)))
            validation = validate_pose_landmarks(pose, visibility_threshold=0.5)
            if validation.low_visibility_landmarks:
                summary["low_visibility_frames"] += 1
                low_visibility_counter.update(validation.low_visibility_landmarks)
            if validation.missing_landmarks or validation.nonfinite_landmarks:
                summary["missing_or_invalid_frames"] += 1
            try:
                adapted = adapt_mediapipe_to_multiposture(pose, visibility_threshold=None)
                feature_rows.append(adapted.values[0])
            except (ValueError, PoseValidationError):
                pass
            if not validation.valid:
                continue
            summary["adapter_eligible_frames"] += 1
            try:
                prediction = runner.predict_landmarks(pose, visibility_threshold=0.5)
            except (ValueError, PoseValidationError):
                continue
            predicted_labels.append(prediction["label"])
            confidence_values.append(float(prediction["confidence"]))
            if row["verified_label"] in {"TUP", "TLB", "TLF", "TLL", "TLR"}:
                true_labels.append(row["verified_label"])
    finally:
        landmarker.close()
        capture.release()

    summary["low_visibility_landmarks"] = dict(low_visibility_counter.most_common())
    summary["mean_visibility_detected"] = float(np.mean(visibility_values)) if visibility_values else None
    summary["mean_visible_landmark_fraction_at_0_5"] = float(np.mean(visible_fractions)) if visible_fractions else None
    summary["predictions"] = dict(Counter(predicted_labels))
    if confidence_values:
        summary["confidence_mean"] = float(np.mean(confidence_values))
        summary["confidence_p50"] = float(np.percentile(confidence_values, 50))
        summary["confidence_min"] = float(np.min(confidence_values))
        summary["confidence_max"] = float(np.max(confidence_values))
    if true_labels and predicted_labels and len(true_labels) == len(predicted_labels):
        summary["accuracy_available"] = True
        summary["accuracy"] = float(accuracy_score(true_labels, predicted_labels))
        summary["balanced_accuracy"] = float(balanced_accuracy_score(true_labels, predicted_labels))
        summary["macro_f1"] = float(f1_score(true_labels, predicted_labels, average="macro"))
        summary["confusion_matrix"] = confusion_matrix(true_labels, predicted_labels, labels=["TUP", "TLB", "TLF", "TLL", "TLR"]).tolist()
    return summary, feature_rows


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values.size else None


def domain_shift_report(feature_rows: list[np.ndarray]) -> dict[str, Any]:
    source_df = pd.read_csv(TRAINING_DATA)
    train_values, feature_names = build_features(source_df[expected_feature_columns()], "hybrid")
    external_values = np.asarray(feature_rows, dtype=float)
    if external_values.ndim != 2:
        external_values = external_values.reshape(0, len(feature_names))
    if external_values.size == 0:
        return {
            "schema_identity": {
                "feature_count": len(feature_names),
                "expected_names_match_adapter": feature_names == multiposture_feature_names(),
                "external_feature_rows": 0,
            },
            "semantic_compatibility_validated": False,
            "reason": "No finite full-coordinate rows were available for diagnostic distribution comparison.",
        }
    train_min = np.min(train_values, axis=0)
    train_max = np.max(train_values, axis=0)
    ext_min = np.min(external_values, axis=0)
    ext_max = np.max(external_values, axis=0)
    outside = (external_values < train_min[None, :]) | (external_values > train_max[None, :])
    iqr = np.percentile(train_values, 75, axis=0) - np.percentile(train_values, 25, axis=0)
    robust_scale = np.maximum(iqr, 1e-8)
    train_median = np.median(train_values, axis=0)
    robust_distance = np.median(np.abs(external_values - train_median[None, :]) / robust_scale[None, :], axis=0)
    feature_table = []
    for index, name in enumerate(feature_names):
        feature_table.append(
            {
                "name": name,
                "train_min": float(train_min[index]),
                "train_max": float(train_max[index]),
                "train_p01": float(np.percentile(train_values[:, index], 1)),
                "train_p99": float(np.percentile(train_values[:, index], 99)),
                "external_min": float(ext_min[index]),
                "external_max": float(ext_max[index]),
                "external_outside_train_range_fraction": float(np.mean(outside[:, index])),
                "external_median_robust_distance": float(robust_distance[index]),
            }
        )
    ranked = sorted(feature_table, key=lambda item: (item["external_outside_train_range_fraction"], item["external_median_robust_distance"]), reverse=True)
    return {
        "schema_identity": {
            "feature_count": len(feature_names),
            "expected_names_match_adapter": feature_names == multiposture_feature_names(),
            "external_feature_rows": int(external_values.shape[0]),
            "feature_order": feature_names,
        },
        "semantic_compatibility_validated": False,
        "coordinate_conventions": {
            "external": "MediaPipe image-normalized x/y/z, then hip-centered for this diagnostic only; visibility ignored for distribution inspection",
            "multiposture_training": "stored dataset coordinates documented as hip-centered; source scale/projection equivalence to RGB MediaPipe is not proven",
            "decision": "Do not treat diagnostic 126-feature shape identity as proof of real-image model compatibility.",
        },
        "external_rows": int(external_values.shape[0]),
        "features_outside_training_min_max": int(np.count_nonzero(np.mean(outside, axis=0) > 0.5)),
        "feature_table": feature_table,
        "most_shifted_features": ranked[:20],
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_reports(audit: dict[str, Any], domain: dict[str, Any]) -> None:
    files = audit.get("files", [])
    total_bytes = sum(int(item.get("size_bytes") or 0) for item in files)
    valid_hashes = all(item.get("sha256_actual") == item.get("sha256_expected") for item in files if item.get("file_exists"))
    all_detected = all(item.get("pose_detected_frames", 0) == item.get("processed_frames", 0) for item in files)
    all_adapter_zero = all(item.get("adapter_eligible_frames", 0) == 0 for item in files)
    source_table = "\n".join(
        f"| {item['source_file']} | {item['verified_label']} | {item.get('source_class')} | {item.get('participant_id_if_available', '')} | {item.get('frame_count_reported', 'n/a')} | {item.get('duration_seconds_reported', 'n/a')} | {item.get('size_bytes', 'n/a')} | {item.get('sha256_actual', 'n/a')} |"
        for item in files
    )
    report = f"""# External RGB validation-media search and verification

## Scope and decision

This is a small external validation/smoke pack. It does not retrain either model and does not modify `datasets/raw/`. DriveNet v6 was selected because its official Mendeley record provides RGB posture videos, explicit source classes, participant/video metadata, and CC BY 4.0. Only `SS → TUP` and `LB → TLB` are used as project labels. `MB` remains `UNLABELED` because “Move Backward” is not a safe project-ontology mapping.

The source videos are retained unchanged. The runtime benchmark processes only the first 300 frames (about 10 seconds) per clip to keep the validation run small; this is a processing limit, not a source trim.

## Source candidates reviewed

| Candidate | Decision | Reason |
|---|---|---|
| DriveNet v6 (Mendeley Data) | SELECTED | RGB MP4 source videos; four explicit classes; participant/video metadata; CC BY 4.0 |
| HuMoLs (University of Edinburgh) | NOT INCLUDED | Useful sitting RGB videos, but the page says “used freely” without a standard exact license expression; kept as a research lead, not a reusable pack |
| SitStraightNet local repository | SMOKE ONLY / EXISTING | 756 images; no verified posture labels or sidecars; prior run had 0/756 MediaPipe detections, so no accuracy |
| SitPose | NOT RGB VALIDATION | Depth/Azure Kinect dataset; unsuitable for ordinary RGB media validation |
| MultiPosture | NOT MEDIA | Landmark CSV source for the trained direction model; no RGB validation media in the project source |

## Downloaded source table

| source_file | verified_label | source_class | participant | source_frames | source_duration_s | bytes | SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
{source_table}

Total local media size: **{total_bytes} bytes** ({total_bytes / (1024**2):.2f} MiB).

## Required verification points

1. **Trustworthy source:** DriveNet v6 official Mendeley Data record, DOI `10.17632/v37nxdn2gh.6`, source URL `{SOURCE_URL}`.
2. **RGB and sitting content:** visual inspection confirms ordinary color video of a person seated in a vehicle; this is a driving-seat domain, not an office desk domain.
3. **Labels:** `SS` and `LB` are explicit source classes. `SS → TUP` and `LB → TLB` are the only semantic mappings used. `MB` is not mapped.
4. **License:** source record and README state **CC BY 4.0**. Reuse requires attribution and a link to the license; this project retains the source name, URL, and exact hashes.
5. **Integrity:** local file hashes match the source file hashes: **{valid_hashes}**.
6. **Scope:** three clips were downloaded: two labeled clips and one smoke-only clip. The source class `RP` was not downloaded because it was not needed to demonstrate the pipeline and its project mapping is ambiguous.
7. **Media properties:** the source videos are 1920×1440 MP4 files at about 30 FPS, with the exact per-file values in the table above.
8. **MediaPipe pre-screen:** pose detection succeeded on **{sum(item.get('pose_detected_frames', 0) for item in files)}/{sum(item.get('processed_frames', 0) for item in files)}** processed frames (**all_detected={all_detected}**).
9. **Adapter/classifier gate:** the safe 33-landmark adapter accepted **{sum(item.get('adapter_eligible_frames', 0) for item in files)}** frames at visibility ≥ 0.5 (**all_adapter_zero={all_adapter_zero}**). Low-visibility legs/feet from the side view caused rejection; therefore no external classifier accuracy is reported.
10. **Engineering decision:** this pack is sufficient for offline RGB pose-extraction and failure-mode smoke testing, but **not sufficient to validate real-image MultiPosture classification**. A future camera-view-compatible, full-landmark sample or manually recorded five-posture set is still needed for adapter validation.

## Observed quality and limitations

- The videos contain real RGB frames and visible seated people, but their camera setup is a side/three-quarter vehicle view rather than the intended office-camera view.
- MediaPipe produced landmarks, but several lower-limb landmarks were below the safety visibility threshold. Feeding those low-confidence values into the 126-feature classifier would be an invalid experiment.
- Because the adapter accepted no frames, there are no labeled accuracy, confusion-matrix, confidence, or domain-generalization claims from this external pack.
- The annotated outputs show skeletons and the verified source label for the two labeled clips; the smoke clip is annotated as `UNLABELED`.
- No webcam device was opened and no `--source 0` run was attempted.

## Artifact locations

- Manifest: `validation_data/external/external_manifest.csv`
- Source metadata: `validation_data/external/source_metadata/DriveNet_v6/`
- Annotated videos: `outputs/validation/external/videos/`
- Per-video telemetry: `outputs/validation/external/metrics/`, `outputs/validation/external/summaries/`, and `outputs/validation/external/logs/`
- Machine-readable audit: `outputs/validation/external/external_media_audit.json`

## Status

- **Labeled external accuracy:** NOT AVAILABLE — safe adapter had zero eligible frames.
- **Unlabeled smoke:** COMPLETED for one DriveNet clip; MediaPipe detection passed, adapter gate exposed the side-view/full-landmark limitation.
- **Real-image adapter:** NOT VALIDATED.
- **Live webcam:** NOT TESTED — hardware not currently available.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    th = f"""# รายงานค้นหาและตรวจสอบสื่อ RGB ภายนอก

## ผลสรุป

เลือก DriveNet รุ่น 6 จาก Mendeley Data เพราะเป็นวิดีโอ RGB จริง มีชื่อคลาสจากแหล่งข้อมูล มี metadata ของผู้เข้าร่วม/วิดีโอ และระบุใบอนุญาต CC BY 4.0 ชัดเจน ดาวน์โหลดไว้เพียง 3 คลิป รวม {total_bytes / (1024**2):.2f} MiB โดยไม่แก้ไขไฟล์ต้นฉบับ

- `SS` → `TUP` (Sit Straight → Upright): map ได้ตรงความหมาย
- `LB` → `TLB` (Lean Backward → Lean Backward): map ได้ตรงความหมาย
- `MB` (Move Backward): เก็บเป็น `UNLABELED` เพราะไม่ควรเดาความหมายเป็นคลาสของโครงการ
- ไม่ดาวน์โหลด `RP` เพราะไม่จำเป็นต่อ smoke test และความหมายของ “Right Position” ยังไม่ชัดว่าเท่ากับ Lean Right

## ตารางไฟล์

| ไฟล์ | label โครงการ | class ต้นทาง | participant | frames | duration (s) | bytes | SHA-256 |
|---|---|---|---:|---:|---:|---:|---|
{source_table}

ตรวจ hash ตรงกับ metadata ของแหล่งข้อมูล: **{valid_hashes}**

## ผลทดสอบ 10 ประเด็น

1. แหล่งข้อมูล: DriveNet v6 official Mendeley Data, `{SOURCE_URL}`
2. สื่อเป็น RGB และมีบุคคลนั่งจริงในรถ แต่ไม่ใช่ภาพโต๊ะทำงานสำนักงานโดยตรง
3. ฉลาก `SS`, `LB`, `MB` มาจาก class folder และ `meta_data.csv`; ไม่มีการเดาฉลากจากภาพ
4. ใบอนุญาต: CC BY 4.0 ต้องให้เครดิตและลิงก์ใบอนุญาต
5. ไฟล์ต้นฉบับถูกเก็บไว้โดยไม่แก้ไข และตรวจ SHA-256 แล้ว
6. ประมวลผลเพียง 300 เฟรมแรกต่อคลิป (~10 วินาที) เพื่อจำกัดขนาดการทดลอง
7. MediaPipe ตรวจพบ pose {sum(item.get('pose_detected_frames', 0) for item in files)}/{sum(item.get('processed_frames', 0) for item in files)} เฟรม
8. adapter ที่ปลอดภัยรับ landmark ครบและ visibility ≥ 0.5 ได้ 0 เฟรม เนื่องจากขา/เท้าบางส่วนไม่ชัดจากมุมด้านข้าง
9. จึงยังไม่มี accuracy หรือ confusion matrix ของ external RGB ชุดนี้
10. ชุดนี้เพียงพอสำหรับ offline pose smoke test และยืนยัน failure mode แต่ยังไม่พอพิสูจน์ว่า MediaPipe RGB ใช้กับ MultiPosture model ได้จริง; ยังต้องใช้สื่อมุมกล้องที่เหมาะสมหรือวิดีโอที่ผู้ใช้บันทึกเองภายหลัง

## ที่เก็บผลลัพธ์

- manifest: `validation_data/external/external_manifest.csv`
- annotated videos: `outputs/validation/external/videos/`
- telemetry: `outputs/validation/external/metrics/`, `summaries/`, `logs/`
- audit JSON: `outputs/validation/external/external_media_audit.json`

สถานะ webcam: **NOT TESTED — HARDWARE NOT CURRENTLY AVAILABLE**
"""
    REPORT_TH_PATH.write_text(th, encoding="utf-8")

    shifted = domain.get("most_shifted_features", [])[:10]
    shift_rows = "\n".join(
        f"| `{item['name']}` | {item['external_outside_train_range_fraction']:.3f} | {item['external_median_robust_distance']:.3f} |"
        for item in shifted
    ) or "| n/a | n/a | n/a |"
    domain_report = f"""# External RGB domain-shift analysis

## Scope

This is a diagnostic comparison of the 126-feature vector produced from MediaPipe landmarks on the external RGB videos versus the 126-feature training representation used by MultiPosture. The comparison ignores visibility only to inspect feature distributions; those rows were **not** sent to the classifier for external accuracy because the safe adapter rejected the low-visibility landmarks.

## Schema identity versus semantic compatibility

- Feature count: `{domain.get('schema_identity', {}).get('feature_count', 'n/a')}`
- Feature-name/order identity with the implemented adapter: `{domain.get('schema_identity', {}).get('expected_names_match_adapter', False)}`
- External diagnostic rows: `{domain.get('schema_identity', {}).get('external_feature_rows', 0)}`
- Semantic compatibility validated: **False**

The external representation uses MediaPipe image-normalized x/y/z followed by hip-centering. The stored MultiPosture training values are documented as hip-centered source coordinates. Equal feature names and count do not prove identical coordinate scale, projection, camera geometry, visibility semantics, or population distribution.

## Range comparison

`outside_fraction` is the fraction of external diagnostic rows outside the training min/max for a feature. `robust_distance` is the median absolute distance from the training median measured in training IQR units.

| feature | outside_fraction | robust_distance |
|---|---:|---:|
{shift_rows}

Features with more than half of external rows outside the training min/max: `{domain.get('features_outside_training_min_max', 'n/a')}`.

## Decision

**MODEL TRAINING READY; REAL-IMAGE INFERENCE ADAPTER NOT YET VALIDATED.** The exact 126-column schema is implemented and mechanically reproducible, but the external camera/view and visibility behavior do not establish domain compatibility. Do not claim RGB generalization or use this external run as classifier accuracy.
"""
    DOMAIN_REPORT_PATH.write_text(domain_report, encoding="utf-8")


def main() -> None:
    write_manifest()
    audit_items: list[dict[str, Any]] = []
    all_feature_rows: list[np.ndarray] = []
    for row in MEDIA_ROWS:
        summary, feature_rows = audit_video(row)
        summary["participant_id_if_available"] = row["participant_id_if_available"]
        summary["session_id_if_available"] = row["session_id_if_available"]
        summary["license"] = row["license"]
        audit_items.append(summary)
        all_feature_rows.extend(feature_rows)
    audit = {
        "source_dataset": SOURCE_DATASET,
        "source_version": SOURCE_VERSION,
        "source_url": SOURCE_URL,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "processed_frame_limit_per_video": 300,
        "files": audit_items,
    }
    domain = domain_shift_report(all_feature_rows)
    AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    DOMAIN_JSON_PATH.write_text(json.dumps(domain, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    write_reports(audit, domain)
    print(json.dumps({"audit": audit, "domain_shift": domain}, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
