"""Audit controlled CCTV V2 data before any future V5 training.

This command is deliberately a gate, not a trainer.  It reads human-labelled
rows and anonymous telemetry, reports missing evidence, and exits successfully
even when the honest result is ``NOT_READY_FOR_CCTV_V5_TRAINING``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.data.cctv_collection import CCTV_EXTRA_SAMPLE_COLUMNS, CCTV_MANUAL_LABELS, CCTV_COLLECTION_SCHEMA_VERSION
from src.posture_ai.data.cctv_metrics import (
    aggregate_detection_metrics,
    detection_metrics_by_scenario,
    frozen_v4_metrics,
    per_person_core_metrics,
    temporal_stability_metrics,
    tracking_metrics,
)


BASE_COLUMNS = {
    "schema_version", "sample_id", "session_id", "subject_id", "clip_id", "human_label",
    "timestamp_seconds", "frame_age_ms", "inference_time_ms", "source_timestamp_seconds",
    "mediapipe_timestamp_seconds", "frame_index", "source_type", "camera_id", "camera_profile",
    "frame_width", "frame_height", "pose_detected", "core_valid", "core_visibility_json",
    "quality_flag", "quality_warnings_json", "v4_prediction", "v4_confidence",
    "v4_probabilities_json", "geometry_json", "calibration_status",
}


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        events.append(value)
    return events


def _float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _flatten_visibility(rows: list[dict[str, str]]) -> None:
    for row in rows:
        try:
            values = json.loads(row.get("core_visibility_json", "{}"))
        except json.JSONDecodeError:
            values = {}
        if not isinstance(values, dict):
            values = {}
        for landmark in ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            value = values.get(landmark)
            if isinstance(value, dict):
                value = value.get("visibility")
            row[landmark + "_visibility"] = "" if value is None else str(value)


def _detection_records(events: list[dict[str, Any]]) -> list[dict[str, int]]:
    records = []
    for event in events:
        expected = event.get("expected_people_count")
        subjects = event.get("subjects") or []
        if expected is None or not isinstance(subjects, list):
            continue
        records.append({
            "expected_people_count": int(expected),
            "detected_people_count": len(subjects),
            "core_valid_people_count": sum(bool(subject.get("core_valid")) for subject in subjects if isinstance(subject, dict)),
        })
    return records


def audit(root: Path) -> dict[str, Any]:
    samples, fieldnames = _read_csv(root / "samples.csv")
    _flatten_visibility(samples)
    event_paths = sorted((root / "sessions").glob("*_collection_events.jsonl")) if (root / "sessions").exists() else []
    events = _read_jsonl(event_paths)
    session_paths = sorted((root / "sessions").glob("*.json")) if (root / "sessions").exists() else []
    participants = sorted({row.get("subject_id", "") for row in samples if row.get("subject_id")})
    labels = Counter((row.get("human_posture_label") or row.get("human_label") or "").upper() for row in samples)
    expected = sorted({int(row["expected_people_count"]) for row in samples if str(row.get("expected_people_count", "")).isdigit()})
    positions = sorted({row.get("target_position_id", "") for row in samples if row.get("target_position_id")})
    blockers: list[str] = []
    warnings: list[str] = []
    required_columns = {
        "session_id", "clip_id", "subject_id", "expected_people_count", "target_subject_id",
        "human_posture_label", "target_position_id", "pose_index", "track_id", "label_source",
    }
    missing_columns = sorted(required_columns - set(fieldnames))
    if not samples:
        blockers.append("NO_HUMAN_LABELLED_CCTV_V2_SAMPLES")
    if missing_columns:
        blockers.append("MISSING_SCHEMA_COLUMNS:" + ",".join(missing_columns))
    if samples and len(participants) < 3:
        blockers.append("NEED_AT_LEAST_3_ANONYMOUS_PARTICIPANTS_FOR_GROUPED_V5_GATE")
    if samples and any(label not in labels for label in ("TUP", "TLB")):
        blockers.append("BOTH_TUP_AND_TLB_HUMAN_LABELS_REQUIRED")
    if samples and len(positions) < 2:
        blockers.append("NEED_AT_LEAST_2_EXPLICIT_CCTV_POSITIONS")
    if samples and any(row.get("label_source") != "MANUAL_OPERATOR_ONLY" for row in samples):
        blockers.append("NON_MANUAL_LABEL_SOURCE_PRESENT")
    if samples and any(row.get("v4_prediction") == (row.get("human_posture_label") or row.get("human_label")) and row.get("label_source") != "MANUAL_OPERATOR_ONLY" for row in samples):
        blockers.append("PREDICTION_USED_AS_LABEL_OR_SOURCE_NOT_EXPLICIT")
    if samples and any(int(row.get("expected_people_count", "0") or 0) not in (1, 2, 3, 4) for row in samples):
        blockers.append("INVALID_EXPECTED_PEOPLE_COUNT")
    if samples and any(not row.get("session_id") or not row.get("clip_id") for row in samples):
        blockers.append("MISSING_GROUP_KEYS")
    if samples and not any(name not in BASE_COLUMNS and name not in set(CCTV_EXTRA_SAMPLE_COLUMNS) for name in fieldnames):
        blockers.append("NO_FEATURE_COLUMNS")
    if len(participants) < 2 and samples:
        warnings.append("Grouped validation cannot rotate with fewer than two participants")

    v4_rows = [row for row in samples if (row.get("human_posture_label") or row.get("human_label", "")).upper() in {"TUP", "TLB"} and row.get("v4_prediction") and row.get("core_valid") == "1"]
    v4_result = frozen_v4_metrics(
        [(row.get("human_posture_label") or row.get("human_label", "")).upper() for row in v4_rows],
        [row["v4_prediction"] for row in v4_rows],
    )
    detection_records = _detection_records(events)
    status = "READY_FOR_CCTV_V5_TRAINING" if not blockers else "NOT_READY_FOR_CCTV_V5_TRAINING"
    return {
        "status": status,
        "schema_version_expected": CCTV_COLLECTION_SCHEMA_VERSION,
        "samples_csv": str(root / "samples.csv"),
        "sample_count": len(samples),
        "session_count": len(session_paths),
        "participant_codes": participants,
        "participant_count": len(participants),
        "human_label_counts": dict(labels),
        "expected_people_counts": expected,
        "explicit_position_ids": positions,
        "event_count": len(events),
        "blockers": blockers,
        "warnings": warnings,
        "grouping_strategy": "PARTICIPANT_INDEPENDENT_SUBJECT_GROUPED; no random frame split",
        "group_key_integrity": "PASS" if samples and all(row.get("subject_id") and row.get("session_id") and row.get("clip_id") for row in samples) else "NOT_READY",
        "detection_metrics": aggregate_detection_metrics(detection_records),
        "detection_metrics_by_expected_people": detection_metrics_by_scenario(detection_records),
        "target_core_metrics": per_person_core_metrics(samples, group_fields=("subject_id", "target_position_id", "distance_band", "expected_people_count")),
        "v4_metrics_on_valid_human_labelled_rows": v4_result,
        "temporal_metrics": temporal_stability_metrics(samples),
        "tracking_metrics": tracking_metrics(events),
        "training_performed": False,
        "v4_prediction_used_as_ground_truth": False,
        "biometric_identity_used": False,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# CCTV V5 training readiness audit",
        "",
        f"สถานะ: **{result['status']}**",
        "",
        "เอกสารนี้เป็น readiness gate เท่านั้น ไม่ train V5 และไม่เปลี่ยน V4",
        "",
        f"- samples: {result['sample_count']} | sessions: {result['session_count']} | participants: {result['participant_count']}",
        f"- labels: `{result['human_label_counts']}`",
        f"- expected people values: `{result['expected_people_counts']}`",
        f"- explicit positions: `{result['explicit_position_ids']}`",
        f"- grouping: `{result['grouping_strategy']}`",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in result["blockers"] or ["NONE"])
    lines.extend(["", "## Metrics status", "", f"- detection: `{result['detection_metrics']['status']}`", f"- target core: `{result['target_core_metrics']['status']}`", f"- frozen V4: `{result['v4_metrics_on_valid_human_labelled_rows']['status']}`", f"- temporal: `{result['temporal_metrics']['status']}`", f"- tracking: `{result['tracking_metrics']['status']}`", "", "## Integrity rules", "", "- human labels must be selected by an operator", "- V4 prediction, Geometry and Temporal are diagnostic fields only", "- no random frame split; use participant/group split", "- do not start V5 until this report is READY and a separate training review approves it", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit controlled CCTV V2 data before V5 training")
    parser.add_argument("--data-root", default="data/controlled_cctv_v2")
    parser.add_argument("--report", default="reports/cctv_v5_training_readiness_th.md")
    parser.add_argument("--json", default="outputs/metrics/cctv_v5_training_readiness.json")
    args = parser.parse_args()
    result = audit(Path(args.data_root))
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_report(Path(args.report), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
