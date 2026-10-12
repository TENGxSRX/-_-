"""Summarize local RGB V1 collection and emit the training-readiness gate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.posture_ai.data.local_rgb_v1 import LOCAL_PARTICIPANTS, LOCAL_POSTURES, LOCAL_SESSIONS  # noqa: E402

MANIFEST_PATH = ROOT / "datasets" / "processed" / "local_rgb_v1" / "manifest.csv"
SUMMARY_JSON = ROOT / "outputs" / "metrics" / "local_rgb_v1_summary.json"
REPORT_EN = ROOT / "reports" / "v5" / "local_rgb_dataset_status.md"
REPORT_TH = ROOT / "reports" / "v5" / "local_rgb_dataset_status_th.md"


def _read_rows() -> list[dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _float_values(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        values.append(value)
    return values


def summarize() -> dict:
    rows = _read_rows()
    present = [row for row in rows if row.get("status") not in ("", "WAITING_FOR_VIDEO")]
    present_by_posture = Counter(row["project_label"] for row in present)
    present_participants = sorted(set(row["participant_id"] for row in present))
    present_sessions = sorted(set(f"{row['participant_id']}/{row['session_id']}" for row in present))
    durations = _float_values(present, "duration_seconds")
    pose_rates = _float_values(present, "pose_detection_rate")
    core_rates = _float_values(present, "core_acceptance_rate")
    independent_groups = {label: sorted(set(row["participant_id"] for row in present if row["project_label"] == label)) for label in LOCAL_POSTURES}
    gate = {
        "all_five_classes_have_video": all(present_by_posture.get(label, 0) > 0 for label in LOCAL_POSTURES),
        "each_class_has_three_participants": all(len(independent_groups[label]) >= 3 for label in LOCAL_POSTURES),
        "at_least_six_participants": len(present_participants) >= 6,
        "participant_ids_available": bool(present_participants),
        "grouped_split_possible": len(present_participants) >= 3,
        "pose_detection_acceptable": bool(pose_rates) and min(pose_rates) >= 0.90,
        "core_acceptance_acceptable": bool(core_rates) and min(core_rates) >= 0.85,
        "raw_hashes_recorded": (ROOT / "outputs" / "metrics" / "local_rgb_v1_hashes.json").exists(),
        "source_labels_missing": False,
        "obvious_confounding": False,
    }
    if not present:
        status = "WAITING_FOR_LOCAL_RGB_DATA"
    elif all(gate.values()):
        status = "LOCAL_V5_DATA_READY"
    else:
        status = "LOCAL_V5_DATA_PARTIALLY_READY"
    result = {
        "status": status,
        "participants_target": len(LOCAL_PARTICIPANTS),
        "participants_with_video": len(present_participants),
        "participants_with_video_ids": present_participants,
        "sessions_target": len(LOCAL_PARTICIPANTS) * len(LOCAL_SESSIONS),
        "sessions_with_video": len(present_sessions),
        "videos_present": len(present),
        "videos_expected": len(rows),
        "posture_videos_present": {label: present_by_posture.get(label, 0) for label in LOCAL_POSTURES},
        "independent_participants_by_class": independent_groups,
        "usable_frames": int(sum(int(row.get("core_accepted_frames") or 0) for row in present)),
        "pose_detection": {"mean": sum(pose_rates) / len(pose_rates) if pose_rates else None, "minimum": min(pose_rates) if pose_rates else None},
        "core_acceptance": {"mean": sum(core_rates) / len(core_rates) if core_rates else None, "minimum": min(core_rates) if core_rates else None},
        "duration_seconds": {"minimum": min(durations) if durations else None, "mean": sum(durations) / len(durations) if durations else None, "maximum": max(durations) if durations else None},
        "camera_views": sorted(set(row.get("camera_view", "") for row in present if row.get("camera_view", ""))),
        "gate": gate,
        "training_performed": False,
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_reports(result)
    return result


def _write_reports(result: dict) -> None:
    gate_lines = "\n".join(f"- `{key}`: {'PASS' if value else 'WAITING/FAIL'}" for key, value in result["gate"].items())
    REPORT_EN.parent.mkdir(parents=True, exist_ok=True)
    REPORT_EN.write_text(
        f"""# Local RGB V1 dataset status

Status: **{result['status']}**

## Collection inventory

- Participants: **{result['participants_with_video']} / {result['participants_target']}** with videos
- Sessions: **{result['sessions_with_video']} / {result['sessions_target']}** with videos
- Videos: **{result['videos_present']} / {result['videos_expected']}**
- Usable frames after pose pre-flight: **{result['usable_frames']}**
- Camera views observed: `{', '.join(result['camera_views']) or 'none yet'}`

| Posture | Videos present | Independent participants |
|---|---:|---:|
""" + "\n".join(f"| {label} | {result['posture_videos_present'][label]} / 20 | {len(result['independent_participants_by_class'][label])} |" for label in LOCAL_POSTURES) + f"""

## Quality summary

- Pose detection: `{result['pose_detection']}`
- Five-core acceptance: `{result['core_acceptance']}`
- Duration: `{result['duration_seconds']}`
- Raw media modification by validation: `False`

## Local V5 gate

{gate_lines}

Training is intentionally **not performed** by this collection system. The
first future experiment must be LOCAL ONLY, with participant-grouped splitting;
public data may be compared later after the local-only baseline.
""",
        encoding="utf-8",
    )
    REPORT_TH.write_text(
        f"""# สถานะชุดข้อมูล Local RGB V1

สถานะ: **{result['status']}**

- ผู้เข้าร่วมที่มีวิดีโอ: **{result['participants_with_video']} / {result['participants_target']}**
- sessions ที่มีวิดีโอ: **{result['sessions_with_video']} / {result['sessions_target']}**
- วิดีโอ: **{result['videos_present']} / {result['videos_expected']}**
- usable frames หลัง pre-flight: **{result['usable_frames']}**

| ท่า | วิดีโอที่มี | ผู้เข้าร่วมอิสระ |
|---|---:|---:|
""" + "\n".join(f"| {label} | {result['posture_videos_present'][label]} / 20 | {len(result['independent_participants_by_class'][label])} |" for label in LOCAL_POSTURES) + f"""

ยังไม่มีการฝึกโมเดล การตัดสินใจนี้เป็น data-readiness gate เท่านั้น ต้องให้ครบ
ทั้ง 5 ท่าและมีหลาย participant ก่อนจึงจะเปิด training ได้ การทดลองแรกในอนาคต
ต้องเป็น LOCAL ONLY และแบ่ง train/validation/test ตาม participant
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = summarize()
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Status: {result['status']}")
        print(f"Participants: {result['participants_with_video']} / {result['participants_target']}")
        print(f"Videos: {result['videos_present']} / {result['videos_expected']}")
        for label, count in result["posture_videos_present"].items():
            print(f"{label}: {count} / 20")


if __name__ == "__main__":
    main()
