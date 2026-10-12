"""Human-labelled RGB robustness collection for the frozen V4 pilot.

This module deliberately keeps human labels separate from V4 predictions.
It stores only accepted feature rows by default; raw video recording is not
part of the writer and must be implemented as a separately explicit feature.
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CONTROLLED_SCHEMA_VERSION = "controlled_rgb_v5_v1"
CONTROLLED_LABELS = ("TUP", "TLB")
CORE_LANDMARKS = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
SUBJECT_ID_PATTERN = re.compile(r"^P[0-9]{3,}$", re.IGNORECASE)
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")

BASE_SAMPLE_COLUMNS = (
    "schema_version",
    "sample_id",
    "session_id",
    "subject_id",
    "clip_id",
    "human_label",
    "timestamp_seconds",
    "frame_age_ms",
    "inference_time_ms",
    "source_timestamp_seconds",
    "mediapipe_timestamp_seconds",
    "frame_index",
    "source_type",
    "camera_id",
    "camera_profile",
    "frame_width",
    "frame_height",
    "pose_detected",
    "core_valid",
    "core_visibility_json",
    "quality_flag",
    "quality_warnings_json",
    "v4_prediction",
    "v4_confidence",
    "v4_probabilities_json",
    "geometry_json",
    "calibration_status",
)


def validate_subject_id(value: str) -> str:
    """Require an anonymous participant code such as P001."""

    text = str(value).strip().upper()
    if not SUBJECT_ID_PATTERN.fullmatch(text):
        raise ValueError("subject_id must be an anonymous code such as P001")
    return text


def validate_safe_id(value: str, *, field_name: str = "id") -> str:
    text = str(value).strip()
    if not SAFE_ID_PATTERN.fullmatch(text):
        raise ValueError(f"{field_name} contains unsupported characters")
    return text


def make_session_id(subject_id: str, clip_id: str, *, now: datetime | None = None) -> str:
    subject = validate_subject_id(subject_id)
    clip = validate_safe_id(clip_id, field_name="clip_id")
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{subject}_{clip}"


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def compact_json(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def evaluate_collection_quality(
    *,
    pose_detected: bool,
    core_visibility: Mapping[str, Any] | None,
    feature_values: Sequence[float] | None,
    feature_names: Sequence[str] | None,
    frame_width: int,
    frame_height: int,
    min_visibility: float = 0.5,
) -> dict[str, Any]:
    """Apply transparent, label-independent acceptance gates to one frame."""

    reasons: list[str] = []
    warnings: list[str] = []
    visibility = dict(core_visibility or {})
    if not pose_detected:
        reasons.append("POSE_LOST")
    missing_or_low = []
    for name in CORE_LANDMARKS:
        item = visibility.get(name)
        if isinstance(item, Mapping):
            visible = bool(item.get("visible", False))
            value = _finite_float(item.get("visibility"))
        else:
            # The multi-person runtime intentionally stores a compact scalar
            # map.  Treat it as visibility only after the explicit core gate
            # has been applied; no low-quality landmark is silently accepted.
            value = _finite_float(item)
            visible = value is not None
        if not visible or value is None or value < float(min_visibility):
            missing_or_low.append(name)
    if missing_or_low:
        reasons.append("CORE_INVALID:" + ",".join(missing_or_low))
    if int(frame_width) <= 0 or int(frame_height) <= 0:
        reasons.append("INVALID_FRAME_DIMENSIONS")
    if feature_values is None or feature_names is None or len(feature_values) != len(feature_names) or not feature_names:
        reasons.append("V4_FEATURE_UNAVAILABLE")
    else:
        numeric = np.asarray(feature_values, dtype=float)
        if numeric.ndim != 1 or not np.isfinite(numeric).all():
            reasons.append("NONFINITE_V4_FEATURES")
    if not pose_detected and "POSE_LOST" not in reasons:
        reasons.append("POSE_LOST")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "quality_flag": "ACCEPTED" if not reasons else "REJECTED_" + reasons[0].split(":", 1)[0],
        "core_landmarks": list(CORE_LANDMARKS),
        "min_visibility": float(min_visibility),
    }


class ControlledSampleWriter:
    """Append-only CSV/JSONL writer for a single controlled collection root."""

    def __init__(
        self,
        root: str | Path,
        *,
        feature_names: Sequence[str],
        allowed_labels: Sequence[str] = CONTROLLED_LABELS,
        schema_version: str = CONTROLLED_SCHEMA_VERSION,
        extra_columns: Sequence[str] = (),
    ):
        self.root = Path(root)
        self.samples_path = self.root / "samples.csv"
        self.rejections_path = self.root / "rejections.jsonl"
        self.sessions_dir = self.root / "sessions"
        self.feature_names = tuple(str(name) for name in feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        self.allowed_labels = tuple(str(label).upper() for label in allowed_labels)
        if not self.allowed_labels or len(set(self.allowed_labels)) != len(self.allowed_labels):
            raise ValueError("allowed_labels must be non-empty and unique")
        self.schema_version = validate_safe_id(schema_version, field_name="schema_version")
        self.extra_columns = tuple(str(name) for name in extra_columns)
        if len(set(self.extra_columns)) != len(self.extra_columns) or any(name in BASE_SAMPLE_COLUMNS or name in self.feature_names for name in self.extra_columns):
            raise ValueError("extra_columns must be unique and must not overlap existing columns")
        self.columns = BASE_SAMPLE_COLUMNS + self.extra_columns + self.feature_names
        self.root.mkdir(parents=True, exist_ok=True)
        if self.samples_path.exists():
            with self.samples_path.open("r", encoding="utf-8", newline="") as handle:
                existing = tuple(next(csv.reader(handle), []))
            if existing != self.columns:
                raise ValueError("existing controlled samples.csv schema does not match V4 feature order")
        else:
            with self.samples_path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(self.columns)

    @property
    def sample_count(self) -> int:
        if not self.samples_path.exists():
            return 0
        with self.samples_path.open("r", encoding="utf-8", newline="") as handle:
            return max(0, sum(1 for _ in handle) - 1)

    def append_rejection(self, record: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.rejections_path.open("a", encoding="utf-8") as handle:
            handle.write(compact_json(dict(record)) + "\n")

    def append_sample(
        self,
        *,
        sample_id: str,
        session_id: str,
        subject_id: str,
        clip_id: str,
        human_label: str,
        timestamp_seconds: float,
        frame_index: int,
        source_type: str,
        camera_id: str | None,
        camera_profile: str | None,
        frame_width: int,
        frame_height: int,
        core_visibility: Mapping[str, Any],
        quality_flag: str,
        quality_warnings: Sequence[str],
        v4_prediction: str | None,
        v4_confidence: float | None,
        v4_probabilities: Mapping[str, Any] | None,
        geometry: Mapping[str, Any] | None,
        calibration_status: str,
        feature_values: Sequence[float],
        frame_age_ms: float | None = None,
        inference_time_ms: float | None = None,
        source_timestamp_seconds: float | None = None,
        mediapipe_timestamp_seconds: float | None = None,
        extra_values: Mapping[str, Any] | None = None,
    ) -> None:
        subject = validate_subject_id(subject_id)
        clip = validate_safe_id(clip_id, field_name="clip_id")
        label = str(human_label).upper()
        if label not in self.allowed_labels:
            raise ValueError(f"human_label must be one of {self.allowed_labels}")
        values = np.asarray(feature_values, dtype=float)
        if values.shape != (len(self.feature_names),) or not np.isfinite(values).all():
            raise ValueError("feature_values do not match the configured V4 feature schema")
        row: dict[str, Any] = {name: "" for name in self.columns}
        row.update({
            "schema_version": self.schema_version,
            "sample_id": validate_safe_id(sample_id, field_name="sample_id"),
            "session_id": validate_safe_id(session_id, field_name="session_id"),
            "subject_id": subject,
            "clip_id": clip,
            "human_label": label,
            "timestamp_seconds": f"{float(timestamp_seconds):.6f}",
            "frame_age_ms": "" if frame_age_ms is None else f"{float(frame_age_ms):.6f}",
            "inference_time_ms": "" if inference_time_ms is None else f"{float(inference_time_ms):.6f}",
            "source_timestamp_seconds": "" if source_timestamp_seconds is None else f"{float(source_timestamp_seconds):.6f}",
            "mediapipe_timestamp_seconds": "" if mediapipe_timestamp_seconds is None else f"{float(mediapipe_timestamp_seconds):.6f}",
            "frame_index": int(frame_index),
            "source_type": str(source_type),
            "camera_id": "" if camera_id is None else str(camera_id),
            "camera_profile": "" if camera_profile is None else str(camera_profile),
            "frame_width": int(frame_width),
            "frame_height": int(frame_height),
            "pose_detected": 1,
            "core_valid": 1,
            "core_visibility_json": compact_json(core_visibility),
            "quality_flag": str(quality_flag),
            "quality_warnings_json": compact_json(list(quality_warnings)),
            "v4_prediction": "" if v4_prediction is None else str(v4_prediction),
            "v4_confidence": "" if v4_confidence is None else f"{float(v4_confidence):.8f}",
            "v4_probabilities_json": compact_json(v4_probabilities or {}),
            "geometry_json": compact_json(geometry or {}),
            "calibration_status": str(calibration_status),
        })
        row.update({name: f"{float(value):.10g}" for name, value in zip(self.feature_names, values)})
        extras = dict(extra_values or {})
        unknown = set(extras) - set(self.extra_columns)
        if unknown:
            raise ValueError(f"extra_values contain unknown columns: {sorted(unknown)}")
        for name in self.extra_columns:
            row[name] = "" if name not in extras or extras[name] is None else str(extras[name])
        with self.samples_path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self.columns).writerow(row)

    def write_session_metadata(self, metadata: Mapping[str, Any]) -> Path:
        session_id = validate_safe_id(str(metadata.get("session_id", "")), field_name="session_id")
        payload = dict(metadata)
        payload.update({
            "schema_version": self.schema_version,
            "session_id": session_id,
            "human_labels_are_manual": True,
            "v4_predictions_are_not_ground_truth": True,
            "raw_video_recorded": False,
            "feature_names": list(self.feature_names),
            "sample_count_in_root_at_close": self.sample_count,
        })
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        path = self.sessions_dir / f"{session_id}.json"
        path.write_text(json.dumps(_json_value(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        return path


__all__ = [
    "BASE_SAMPLE_COLUMNS",
    "CONTROLLED_LABELS",
    "CONTROLLED_SCHEMA_VERSION",
    "CORE_LANDMARKS",
    "ControlledSampleWriter",
    "compact_json",
    "evaluate_collection_quality",
    "make_session_id",
    "validate_safe_id",
    "validate_subject_id",
]
