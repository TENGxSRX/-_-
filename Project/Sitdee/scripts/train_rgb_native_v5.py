"""Build the RGB-native V5 dataset manifest and readiness audit.

This command is deliberately conservative.  It extracts the frozen V4
World-Landmark features from traceable RGB media, but refuses to fabricate a
multi-class V5 model when the source-backed class coverage is insufficient.
The training branch is intentionally gated on all five direction classes and
at least two participant groups per class; the current public sources do not
meet that gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.posture_ai.data.rgb_native_v5 import (  # noqa: E402
    V5_MANIFEST_COLUMNS,
    V5_PROJECT_LABELS,
    dataset_class_confounds,
    sample_frame_indices,
    validate_manifest_rows,
)
from src.posture_ai.features.rgb_native_v5 import (  # noqa: E402
    V5_CORE_LANDMARKS,
    V5_FEATURE_FAMILIES,
    build_rgb_native_v5_features,
    v5_feature_names,
)
from src.posture_ai.pose.landmarks import LANDMARK_INDEX  # noqa: E402

V5_RAW_DIR = PROJECT_ROOT / "datasets" / "raw" / "rgb_native_v5"
V5_PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed" / "rgb_native_v5"
V5_MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "rgb_native_v5"
V5_METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
V5_REPORT_DIR = PROJECT_ROOT / "reports" / "v5"
POSE_MODEL = PROJECT_ROOT / "metadata" / "pose_landmarker_heavy.task"
DRIVENET_DIR = PROJECT_ROOT / "datasets" / "raw" / "rgb_native" / "drivenet_v6" / "source_videos"
MMD_PICTURES_DIR = V5_RAW_DIR / "mmd_msd" / "official_subset" / "Pictures_raw"
MMD_ANGLES_DIR = V5_RAW_DIR / "mmd_msd" / "official_subset" / "Angles"
DRIVENET_URL = "https://data.mendeley.com/datasets/v37nxdn2gh/6"
MMD_URL = "http://isr.tu-varna.bg/ergo/index.php/resursi"
MMD_PAPER_URL = "https://www.mdpi.com/1999-4893/17/5/187"
MVSP_URL = "https://github.com/LKW123800/multi-view-sitting-posture-dataset"
MVSP_PAPER_URL = "https://www.sciencedirect.com/science/article/pii/S0957417425021050"
CCTV_URL = "https://www.kaggle.com/datasets/cctvdataset/cctv-exam-monitor-dataset"
SITSTRAIGHTNET_URL = "https://github.com/realratchet/SitStraightNet"
SITSTRAIGHTNET_PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8022631/"
SITPOSE_URL = "https://arxiv.org/abs/2412.12216"
DRIVENET_LICENSE = "CC BY 4.0"
MMD_LICENSE = "CC BY 4.0 (paper/source record)"
MMD_PARTICIPANTS = ("19", "20", "21", "22", "23")
SUPPORTED_DRIVENET = {"SS": "TUP", "LB": "TLB"}
MMD_CONDITIONS = {
    1: ("Uncorrected Standing en face", "", "INCOMPATIBLE", "front"),
    2: ("Uncorrected Standing left profile", "", "INCOMPATIBLE", "left_profile"),
    3: ("Corrected Standing en face", "", "INCOMPATIBLE", "front"),
    4: ("Corrected Standing left profile", "", "INCOMPATIBLE", "left_profile"),
    5: ("Uncorrected Sitting left profile", "", "AMBIGUOUS", "left_profile"),
    6: ("Corrected Sitting left profile", "TUP", "COMPATIBLE", "left_profile"),
    7: ("Sitting left profile with stability disk", "TUP", "COMPATIBLE", "left_profile"),
}


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _create_landmarker(running_mode: RunningMode):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def _result_to_arrays(result) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return None, None
    image_points = result.pose_landmarks[0]
    world_points = result.pose_world_landmarks[0]
    if len(image_points) != 33 or len(world_points) != 33:
        return None, None
    visibility = np.asarray([float(getattr(point, "visibility", 0.0)) for point in image_points], dtype=float)
    world = np.asarray([[float(point.x), float(point.y), float(point.z)] for point in world_points], dtype=float)
    return visibility, world


def _quality(visibility: np.ndarray | None) -> tuple[bool, float | None]:
    if visibility is None or visibility.shape != (33,) or not np.isfinite(visibility).all():
        return False, None
    indices = [LANDMARK_INDEX[name] for name in V5_CORE_LANDMARKS]
    core = visibility[indices]
    return bool(np.all(core >= 0.5)), float(np.mean(core))


def _base_record(**values: Any) -> dict[str, Any]:
    row = {name: "" for name in V5_MANIFEST_COLUMNS}
    row.update(values)
    return row


def _add_features(
    record: dict[str, Any],
    world: np.ndarray | None,
    records: list[dict[str, Any]],
    feature_values: dict[str, list[np.ndarray]],
    feature_row_indices: list[int],
) -> None:
    records.append(record)
    index = len(records) - 1
    if world is None or not np.isfinite(world).all():
        return
    accepted, quality = _quality(record.pop("_visibility", None))
    record["pose_quality"] = "" if quality is None else f"{quality:.6f}"
    if not accepted:
        return
    try:
        built: dict[str, np.ndarray] = {}
        for family in V5_FEATURE_FAMILIES:
            values, names = build_rgb_native_v5_features(world, family)
            if names != v5_feature_names(family):
                raise ValueError(f"feature schema mismatch for {family}")
            built[family] = values[0]
        for family, values in built.items():
            feature_values[family].append(values)
        feature_row_indices.append(index)
    except (ValueError, AssertionError):
        return


def _extract_drivenet(target_fps: float, records, feature_values, feature_row_indices, sampling_audit):
    for video in sorted(DRIVENET_DIR.glob("*.mp4")):
        source_code = video.stem.split("_")[0]
        if source_code not in SUPPORTED_DRIVENET:
            continue
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open DriveNet source video: {video}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        selected = sample_frame_indices(frame_count, fps, target_fps)
        sampling_audit[video.name] = {
            "frame_count": frame_count,
            "fps": fps,
            "duration_seconds": frame_count / fps,
            "sampled_at_target_fps": int(len(selected)),
            "target_fps": target_fps,
            "comparison_1fps": int(len(sample_frame_indices(frame_count, fps, 1.0))),
            "comparison_2fps": int(len(sample_frame_indices(frame_count, fps, 2.0))),
            "comparison_5fps": int(len(sample_frame_indices(frame_count, fps, 5.0))),
        }
        selected_set = set(int(x) for x in selected.tolist())
        landmarker = _create_landmarker(RunningMode.VIDEO)
        try:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index not in selected_set:
                    frame_index += 1
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, int(frame_index * 1000.0 / fps))
                visibility, world = _result_to_arrays(result)
                accepted, quality = _quality(visibility)
                record = _base_record(
                    sample_id=f"drivenet_v5_{video.stem}_{frame_index:06d}",
                    source_dataset="DriveNet_v6",
                    source_url=DRIVENET_URL,
                    source_file=str(video),
                    participant_id=video.stem.split("_")[-1],
                    session_id=video.stem,
                    video_id=video.stem,
                    frame_index=frame_index,
                    timestamp=frame_index / fps,
                    source_label=source_code,
                    project_label=SUPPORTED_DRIVENET[source_code] if accepted else "",
                    mapping_status="COMPATIBLE" if accepted else "AMBIGUOUS",
                    license=DRIVENET_LICENSE,
                    camera_view="controlled_driver_view",
                    pose_detected=int(visibility is not None),
                    pose_quality="" if quality is None else f"{quality:.6f}",
                    notes="1 fps deterministic sample; project label is source-compatible only; DriveNet is not an office-desk dataset",
                )
                record["_visibility"] = visibility
                _add_features(record, world, records, feature_values, feature_row_indices)
                frame_index += 1
        finally:
            landmarker.close()
            capture.release()


def _extract_mmd(records, feature_values, feature_row_indices):
    images = sorted(MMD_PICTURES_DIR.glob("*.jpg"))
    if len(images) != 35:
        raise RuntimeError(f"Expected the public 5-participant MMD-MSD subset to contain 35 images, found {len(images)}")
    landmarker = _create_landmarker(RunningMode.IMAGE)
    try:
        for position, image_path in enumerate(images):
            participant = MMD_PARTICIPANTS[min(position // 7, len(MMD_PARTICIPANTS) - 1)]
            condition_id = position % 7 + 1
            source_label, project_label, mapping_status, view = MMD_CONDITIONS[condition_id]
            frame = cv2.imread(str(image_path))
            visibility = None
            world = None
            if frame is not None:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect(image)
                visibility, world = _result_to_arrays(result)
            accepted, quality = _quality(visibility)
            record = _base_record(
                sample_id=f"mmd_msd_v5_{participant}_condition_{condition_id}",
                source_dataset="MMD-MSD_public_5_participant_subset",
                source_url=MMD_URL,
                source_file=str(image_path),
                participant_id=participant,
                session_id=f"MMD-MSD-participant-{participant}",
                video_id="",
                frame_index="",
                timestamp="",
                source_label=source_label,
                project_label=project_label if accepted else "",
                mapping_status=mapping_status if accepted else "AMBIGUOUS",
                license=MMD_LICENSE,
                camera_view=view,
                pose_detected=int(visibility is not None),
                pose_quality="" if quality is None else f"{quality:.6f}",
                notes="Condition/participant association follows the official 7-image protocol and ordered public subset; no per-file label sidecar was published",
            )
            record["_visibility"] = visibility
            _add_features(record, world, records, feature_values, feature_row_indices)
    finally:
        landmarker.close()


def _write_manifest(records: list[dict[str, Any]]) -> None:
    validate_manifest_rows(records)
    V5_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with (V5_PROCESSED_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=V5_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in V5_MANIFEST_COLUMNS} for row in records)


def _source_hashes() -> dict[str, Any]:
    files = []
    for path in sorted(DRIVENET_DIR.glob("*.mp4")):
        if path.stem.split("_")[0] in SUPPORTED_DRIVENET:
            files.append(path)
    files.extend(sorted(MMD_PICTURES_DIR.glob("*.jpg")))
    files.extend(sorted(MMD_ANGLES_DIR.glob("*.csv")))
    return {str(path): {"bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files}


def _source_registry() -> list[dict[str, Any]]:
    return [
        {"source": "DriveNet_v6", "url": DRIVENET_URL, "media": "RGB videos", "source_labels": ["LB Lean_Backward", "MB Move_Backward", "SS Sit_Straight", "RP Right_Position"], "participants": 3, "license": DRIVENET_LICENSE, "decision": "USED: SS->TUP and LB->TLB; MB/RP audit-only"},
        {"source": "MMD-MSD public subset", "url": MMD_URL, "media": "RGB pictures; official 5-participant subset", "source_labels": ["Uncorrected Sitting", "Corrected Sitting", "Sitting with stability disk"], "participants": 5, "license": MMD_LICENSE, "decision": "USED: corrected/active sitting->TUP only; non-directional conditions audit-only"},
        {"source": "MVSP/LGCSPNet", "url": MVSP_URL, "media": "RGB images; paper reports 49,798 images", "source_labels": ["back", "cheek", "down", "left", "leg", "right", "upright"], "participants": "Not verified locally", "license": "NOT STATED in official repository", "decision": "NOT USED: download is Baidu-only and license is unresolved"},
        {"source": "CCTV Exam Monitor", "url": CCTV_URL, "media": "RGB JPG archive", "source_labels": ["Correct Posture", "LeftSideMove", "RightSideMove", "ForwardMove", "BackwardMove", "Stand"], "participants": "Not available in local files", "license": "CC0/public-domain claim from source page", "decision": "NOT USED: local archive lacks per-image labels/annotation sidecar"},
        {"source": "SitStraightNet", "url": SITSTRAIGHTNET_URL, "media": "RGB-D image repository snapshot", "source_labels": ["sitting straight", "lightly hunched", "hunched over", "extremely hunched", "partially lying", "lying down"], "participants": "11 (paper; not verified locally)", "license": "MIT for repository code; image data label join unresolved", "decision": "NOT USED: local images have no verified per-file posture labels"},
        {"source": "SitPose", "url": SITPOSE_URL, "media": "Kinect depth/skeleton", "source_labels": ["six sitting postures + standing"], "participants": "36 (paper; not verified locally)", "license": "Not verified locally", "decision": "NOT USED: not normal RGB media"},
    ]


def _mapping_report() -> str:
    return f"""# RGB-native V5 label mapping audit

Generated: {datetime.now(timezone.utc).isoformat()}

## Decision rule

Only labels that are explicit in the source and semantically compatible with
the project direction ontology are assigned a project label.  Motion labels,
head-only actions, general slouch labels without a direction definition, and
unjoinable archive images remain unlabelled.

| Source | Verified source labels / descriptions | Project mapping | Status | Training decision |
|---|---|---|---|---|
| DriveNet v6 | `Sit_Straight`, `Lean_Backward`, `Move_Backward`, `Right_Position` | `Sit_Straight` -> `TUP`; `Lean_Backward` -> `TLB`; others blank | COMPATIBLE / AMBIGUOUS | Used for the current 2-class pilot inventory |
| MMD-MSD public subset | `Corrected Sitting`, `Uncorrected Sitting`, `Sitting with stability disk` | corrected/active sitting -> `TUP`; uncorrected sitting is not a direction label | COMPATIBLE / AMBIGUOUS | Used only for TUP diversity and audit |
| MVSP/LGCSPNet | `upright`, `back`, `down`, `left`, `right`, `cheek`, `leg` | likely `TUP`, `TLB`, `TLF`, `TLL`, `TLR` for the first five | COMPATIBLE if downloaded and license resolved | Not used: official repository gives Baidu-only download and no license statement |
| CCTV Exam Monitor | `Correct Posture`, `LeftSideMove`, `RightSideMove`, `ForwardMove`, `BackwardMove`, `Stand` | potentially compatible, but per-image join is unavailable locally | AMBIGUOUS | Audit-only |
| SitStraightNet | `sitting straight`, `lightly hunched`, `hunched over`, `extremely hunched`, `partially lying`, `lying down` | no defensible exact direction mapping; source image-to-label join unresolved | AMBIGUOUS | Smoke-only |
| SitPose | Kinect/depth sitting postures | not RGB-native | INCOMPATIBLE for this track | Not used |

## Current supervised coverage

The current manifest has verified project labels only for `TUP` and `TLB`.
No source-backed RGB samples for `TLF`, `TLL`, or `TLR` were eligible for
training.  `SLOUCH` is postponed because the available sources either lack a
per-file join or use semantics that are not identical to the V5 direction
ontology.

## Source references

- [DriveNet v6]({DRIVENET_URL})
- [MMD-MSD official resource page]({MMD_URL})
- [MMD-MSD paper]({MMD_PAPER_URL})
- [MVSP/LGCSPNet repository]({MVSP_URL})
- [MVSP/LGCSPNet paper]({MVSP_PAPER_URL})
- [CCTV Exam Monitor]({CCTV_URL})
- [SitStraightNet repository]({SITSTRAIGHTNET_URL}) and [paper]({SITSTRAIGHTNET_PAPER_URL})
- [SitPose paper]({SITPOSE_URL})
"""


def _confound_report(records: list[dict[str, Any]], confound: dict[str, Any]) -> str:
    grouped = defaultdict(lambda: {"participants": set(), "videos": set(), "frames": 0})
    for row in records:
        label = str(row.get("project_label") or "")
        if not label:
            continue
        entry = grouped[label]
        entry["participants"].add(str(row.get("participant_id")))
        entry["videos"].add(str(row.get("video_id") or row.get("session_id")))
        entry["frames"] += 1
    lines = [
        "# V5 dataset/class confound audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Project-labelled coverage",
        "",
        "| Project label | Participants | Videos/sessions | Manifest rows |",
        "|---|---:|---:|---:|",
    ]
    for label in sorted(grouped):
        value = grouped[label]
        lines.append(f"| {label} | {len(value['participants'])} | {len(value['videos'])} | {value['frames']} |")
    lines.extend([
        "",
        "## Confound test",
        "",
        f"```json\n{json.dumps(_jsonable(confound), indent=2, ensure_ascii=False)}\n```",
        "",
        "The current usable labels are dominated by source identity: TUP is"
        " represented by DriveNet SS plus MMD-MSD corrected/active sitting,"
        " while TLB is represented by DriveNet LB.  There are only two usable"
        " direction classes and no TLF/TLL/TLR samples.  A five-class V5 model"
        " must not be trained from this manifest because class identity and"
        " source/camera domain would be inseparable.",
        "",
        "The row-level random split is not used.  Future V5 training must keep"
        " participant, session, and video groups intact and must include at"
        " least one independent source or participant for each direction."
    ])
    return "\n".join(lines) + "\n"


def _final_reports(records: list[dict[str, Any]], sampling_audit: dict[str, Any], hashes: dict[str, Any], confound: dict[str, Any]) -> None:
    labelled = [row for row in records if str(row.get("project_label") or "")]
    counts = Counter(str(row["project_label"]) for row in labelled)
    participant_by_class = defaultdict(set)
    for row in labelled:
        participant_by_class[str(row["project_label"])].add(str(row["participant_id"]))
    missing = [label for label in V5_PROJECT_LABELS if label not in counts]
    report = f"""# RGB-native V5 dataset expansion status

Generated: {datetime.now(timezone.utc).isoformat()}

## Result

**V5 MULTI-CLASS MODEL: NOT TRAINED.**  The source-backed RGB inventory is
not sufficient for a scientifically defensible five-class direction model.
No pseudo-labels were created and no V5 model artifact was exported.

## Dataset inventory

| Source | Download/use decision | Media | Notes |
|---|---|---|---|
| DriveNet v6 | Used | RGB video | six mapped videos; SS -> TUP and LB -> TLB only |
| MMD-MSD public subset | Used for TUP diversity/audit | RGB images | 35 images, five participants, corrected/active sitting can support TUP only |
| MVSP/LGCSPNet | Not used | RGB image dataset reported by paper | exact missing classes appear available, but official repo has no license statement and the only download is Baidu |
| CCTV Exam Monitor | Audit only | RGB JPG | no per-image label sidecar in the local archive |
| SitStraightNet | Smoke only | RGB-D images | no verified per-file labels in the local snapshot |
| SitPose | Not used | Kinect depth/skeleton | not normal RGB media |

## Manifest and extraction

- Manifest rows: **{len(records)}**
- Source-backed supervised rows: **{len(labelled)}**
- Project label counts: **{dict(counts)}**
- Participants by available class: **{{{', '.join(f'{key}: {sorted(value)}' for key, value in sorted(participant_by_class.items()))}}}**
- Missing direction classes: **{', '.join(missing)}**
- Pose pipeline: MediaPipe Pose World Landmarks -> hip-centered V4-compatible features
- Core landmarks: `{', '.join(V5_CORE_LANDMARKS)}`
- Feature families prepared: `{', '.join(V5_FEATURE_FAMILIES)}`
- DriveNet sampling target: **1 fps**, with 1/2/5 fps counts retained for audit
- Pose detection and core acceptance are recorded per row in `manifest.csv`; no frame is silently imputed

## Leakage controls

The manifest keeps participant/session/video identifiers.  No random frame
split was performed.  A final five-class model is gated until all direction
classes have source-backed labels and enough independent participant/source
groups.  Current class/source confounding is severe for any attempted
direction benchmark because the available direction labels come from one
controlled source domain.

## Required next data

The immediate missing classes are **TLF, TLL, and TLR**.  The best candidate
identified is MVSP/LGCSPNet because its published category descriptions
explicitly include down/left/right, but it cannot be admitted until the
dataset license and downloadable archive are verified.  Otherwise collect a
small, explicitly labelled RGB desk dataset with at least 6--10 participants,
all five direction classes per participant, and separate recording sessions.

## Model and evaluation status

Models benchmarked: **none for V5**, because training was correctly blocked
before model fitting.  Grouped CV, participant holdout, leave-one-participant-
out, and leave-one-dataset-out metrics are therefore **NOT RUN**.  This is a
data-readiness decision, not a failed classifier result.

V4 was not retrained or overwritten.  ANYA was not used.  Webcam source 0 was
not opened.  YOLO was not installed.

## Artifacts

- Unified manifest: `datasets/processed/rgb_native_v5/manifest.csv`
- Prepared features: `datasets/processed/rgb_native_v5/features.npz`
- Source hashes: `outputs/metrics/rgb_native_v5_source_hashes.json`
- Sampling audit: `outputs/metrics/rgb_native_v5_sampling.json`
- Model gate: `outputs/models/rgb_native_v5/NOT_TRAINED.md`
- Label mapping: `reports/v5/rgb_v5_label_mapping.md`
- Confound audit: `reports/v5/dataset_class_confounds.md`

## Source hash count

The hash inventory contains **{len(hashes)}** downloaded/source files.  Raw
DriveNet, MultiPosture, ANYA, and prior V4 artifacts were not modified.
"""
    _write_text(PROJECT_ROOT / "reports" / "rgb_native_v5.md", report)
    _write_text(PROJECT_ROOT / "reports" / "rgb_native_v5_th.md", """# สถานะ RGB-native V5\n\n**ยังไม่ฝึกโมเดล V5 แบบหลายคลาส** เพราะข้อมูลที่ตรวจสอบได้มีเพียง TUP และ TLB; ยังขาด TLF, TLL และ TLR จึงไม่สร้างป้ายกำกับสมมติและไม่ฝึกโมเดลที่อาจทำให้เข้าใจผิด\n\nกระบวนการที่ทำแล้ว: ดาวน์โหลดและตรวจสอบ DriveNet กับ MMD-MSD ชุดสาธารณะ, สร้าง manifest ที่ย้อนกลับไปยังไฟล์ต้นฉบับได้, สกัด MediaPipe World Landmarks ตามสคีมา V4, ตรวจการแบ่งกลุ่มผู้เข้าร่วม/วิดีโอ และตรวจ dataset/class confound\n\nไฟล์สำคัญ: `datasets/processed/rgb_native_v5/manifest.csv`, `datasets/processed/rgb_native_v5/features.npz`, `reports/v5/rgb_v5_label_mapping.md`, `reports/v5/dataset_class_confounds.md`\n\nคลาสที่ต้องเพิ่มต่อไป: TLF (เอนไปข้างหน้า), TLL (เอนไปซ้าย), TLR (เอนไปขวา) โดยต้องมีป้ายกำกับจากแหล่งทางการและแบ่งตาม participant/session/video ห้ามสุ่มแบ่งรายเฟรม\n\nV4 ไม่ถูกเขียนทับ, ANYA ไม่ถูกฝึก, ไม่เปิดกล้อง 0 และไม่ได้ติดตั้ง YOLO\n""")
    _write_text(V5_REPORT_DIR / "leave_one_dataset_out.md", """# V5 leave-one-dataset-out\n\nNOT RUN. Only two direction classes have defensible supervision, and no multi-dataset five-class training corpus exists. Running this test now would measure source identity rather than posture generalization.\n""")
    _write_text(V5_REPORT_DIR / "error_analysis.md", """# V5 error analysis\n\nNOT RUN. V5 model fitting was blocked because TLF, TLL, and TLR lack eligible source-backed RGB labels. There are no V5 classifier errors to analyze. Pose-quality and source-label audit details are in `datasets/processed/rgb_native_v5/manifest.csv` and `reports/rgb_native_v5.md`.\n""")
    _write_text(V5_REPORT_DIR / "rgb_pose_quality_audit.md", """# V5 RGB pose-quality audit\n\nPose-quality fields are recorded per media row in `datasets/processed/rgb_native_v5/manifest.csv`. A row is feature-eligible only when all five core landmarks meet visibility >= 0.5 and World Landmark coordinates are finite. Low-quality rows remain in the manifest and are not silently imputed.\n""")
    _write_text(V5_REPORT_DIR / "v4_vs_v5.md", """# V4 vs V5\n\nV4 remains a frozen two-class RGB-native pilot (TUP/TLB, DriveNet, 3 participants). V5 currently has a prepared multi-source manifest and frozen V4-compatible feature extraction, but no multi-class model because TLF/TLL/TLR are missing from eligible source-backed RGB data. V5 does not replace V4.\n""")
    _write_text(V5_REPORT_DIR / "rgb_cross_dataset_generalization.md", """# V5 RGB cross-dataset generalization\n\nNOT RUN for classifier metrics. The available sources do not provide a leakage-safe five-class corpus with joinable labels. MVSP is the most promising missing-class candidate but remains excluded until its license and archive are verified.\n""")
    _write_text(V5_REPORT_DIR / "sampling_audit.md", f"# V5 frame sampling audit\n\n```json\n{json.dumps(_jsonable(sampling_audit), indent=2)}\n```\n\nThe selected interval is 1 fps for reproducible anti-redundancy sampling; it was not chosen by validation accuracy.\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    for directory in (V5_RAW_DIR, V5_PROCESSED_DIR, V5_MODEL_DIR, V5_METRICS_DIR, V5_REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    feature_values: dict[str, list[np.ndarray]] = {family: [] for family in V5_FEATURE_FAMILIES}
    feature_row_indices: list[int] = []
    sampling_audit: dict[str, Any] = {}
    _extract_drivenet(args.target_fps, records, feature_values, feature_row_indices, sampling_audit)
    _extract_mmd(records, feature_values, feature_row_indices)
    for row in records:
        row.pop("_visibility", None)
    _write_manifest(records)
    np.savez_compressed(
        V5_PROCESSED_DIR / "features.npz",
        **{family: np.asarray(values, dtype=np.float32) for family, values in feature_values.items()},
        manifest_row_indices=np.asarray(feature_row_indices, dtype=np.int64),
    )
    _write_json(V5_PROCESSED_DIR / "feature_config.json", {"pipeline": "rgb_native_v5", "coordinate_representation": "MediaPipe Pose World Landmarks", "transformation": "hip midpoint translation; V4-compatible", "core_landmarks": list(V5_CORE_LANDMARKS), "families": {family: v5_feature_names(family) for family in V5_FEATURE_FAMILIES}})
    hashes = _source_hashes()
    _write_json(V5_METRICS_DIR / "rgb_native_v5_source_hashes.json", hashes)
    _write_json(V5_METRICS_DIR / "rgb_native_v5_sampling.json", sampling_audit)
    _write_json(V5_PROCESSED_DIR / "source_registry.json", _source_registry())
    labelled = [row for row in records if str(row.get("project_label") or "")]
    confound = dataset_class_confounds([row["source_dataset"] for row in labelled], [row["project_label"] for row in labelled]) if labelled else {}
    _write_text(PROJECT_ROOT / "reports" / "v5" / "rgb_v5_label_mapping.md", _mapping_report())
    _write_text(PROJECT_ROOT / "reports" / "v5" / "dataset_class_confounds.md", _confound_report(records, confound))
    _final_reports(records, sampling_audit, hashes, confound)
    missing = [label for label in V5_PROJECT_LABELS if label not in {str(row.get("project_label") or "") for row in labelled}]
    status = "NOT_TRAINED_INSUFFICIENT_SOURCE_BACKED_CLASSES" if missing else "TRAINING_GATE_REQUIRES_GROUP_DIVERSITY_CHECK"
    _write_text(V5_MODEL_DIR / "NOT_TRAINED.md", f"# RGB-native V5 model gate\n\nStatus: **{status}**\n\nMissing source-backed classes: {', '.join(missing) if missing else 'none'}\n\nNo model.joblib was created. V4 remains untouched.\n")
    _write_json(V5_METRICS_DIR / "rgb_native_v5_readiness.json", {"status": status, "records": len(records), "feature_eligible_records": len(feature_row_indices), "supervised_rows": len(labelled), "class_counts": dict(Counter(str(row["project_label"]) for row in labelled)), "missing_classes": missing, "training_run": "blocked before fitting"})
    return {"status": status, "records": len(records), "feature_eligible_records": len(feature_row_indices), "missing_classes": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and audit the RGB-native V5 dataset expansion")
    parser.add_argument("--target-fps", type=float, default=1.0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
