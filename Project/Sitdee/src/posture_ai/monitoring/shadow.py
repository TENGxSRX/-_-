"""Bounded, anonymous shadow-observation telemetry and analysis.

This module is intentionally separate from the person-centric runtime.  It
observes the runtime's output; it does not alter tracking, V4 inference,
geometry, temporal logic, model artifacts, or raw data.  Shadow observations
are engineering evidence only and must never be used as automatic labels.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import numpy as np

from src.posture_ai.inference.video_source import redact_url


SHADOW_SCHEMA_VERSION = "person_shadow_v1"
IMPORTANT_LANDMARKS = (
    "nose",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
LOWER_BODY_LANDMARKS = (
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
)
MISSING_CODE = {
    "left_hip": "LEFT_HIP_MISSING",
    "right_hip": "RIGHT_HIP_MISSING",
    "left_knee": "LEFT_KNEE_MISSING",
    "right_knee": "RIGHT_KNEE_MISSING",
    "left_ankle": "LEFT_ANKLE_MISSING",
    "right_ankle": "RIGHT_ANKLE_MISSING",
}
UNKNOWN_REASON_CODES = frozenset({
    "POSE_LOST",
    "CORE_INVALID",
    *MISSING_CODE.values(),
    "LOW_LOWER_BODY_VISIBILITY",
    "DESK_OCCLUSION_SUSPECTED",
    "INSUFFICIENT_ACTIVITY_GEOMETRY",
    "ACTIVITY_TRANSITION_UNCERTAIN",
})

_SAFE_SOURCE_KEYS = frozenset({
    "source_type", "source", "camera_id", "source_status", "connected",
    "frames_received", "decode_failures", "consecutive_failures", "reported_fps",
    "effective_received_fps", "resolution", "last_frame_age_seconds",
    "reconnect_count", "late_frame_count", "overwritten_frame_count", "backend",
    "buffer_control_requested", "buffer_control_applied",
})


def _json_safe(value: Any) -> Any:
    """Convert numpy values while refusing NaN/Infinity in long-run logs."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def safe_source_metadata(health: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep useful health data but never credential-bearing RTSP configuration."""

    safe = {str(key): _json_safe(value) for key, value in dict(health or {}).items() if str(key) in _SAFE_SOURCE_KEYS}
    if "source" in safe and safe["source"] is not None:
        safe["source"] = redact_url(str(safe["source"]))
    return safe


def landmark_visibility(landmarks: Any) -> tuple[dict[str, float | None], dict[str, bool]]:
    """Return only the important landmark visibility/presence diagnostics."""

    visibility: dict[str, float | None] = {}
    present: dict[str, bool] = {}
    for name in IMPORTANT_LANDMARKS:
        point = landmarks.by_name(name)
        point_present = bool(point is not None and point.present)
        present[name] = point_present
        raw_visibility = None if point is None else point.visibility
        visibility[name] = None if raw_visibility is None else float(raw_visibility)
    return visibility, present


def derive_unknown_reasons(subject: Any, *, visibility_threshold: float = 0.5) -> dict[str, Any]:
    """Explain UNKNOWN conservatively from current observable evidence.

    ``DESK_OCCLUSION_SUSPECTED`` deliberately is never a fact claim.  It is
    emitted only when the upper/core pose and both hips are available while a
    knee or ankle is absent/low visibility, a pattern consistent with desk
    occlusion but also possible for framing or detector limitations.
    """

    activity_state = str(getattr(subject, "activity_state", "UNKNOWN")).upper()
    tracking_state = str(getattr(subject, "tracking_state", "TEMPORARILY_LOST"))
    evidence = dict(getattr(subject, "activity_evidence", {}) or {})
    if activity_state != "UNKNOWN":
        return {"reason_codes": [], "desk_occlusion_suspected": False}
    if tracking_state not in {"NEW", "ACTIVE"}:
        return {"reason_codes": ["POSE_LOST"], "desk_occlusion_suspected": False}

    reasons: list[str] = []
    if not bool(getattr(subject, "core_valid", False)):
        reasons.append("CORE_INVALID")

    visibility, present = landmark_visibility(subject.landmarks)
    lower_low_visibility = False
    knee_or_ankle_impaired = False
    hip_ok = True
    for name in LOWER_BODY_LANDMARKS:
        if not present[name]:
            reasons.append(MISSING_CODE[name])
            if name.endswith("knee") or name.endswith("ankle"):
                knee_or_ankle_impaired = True
            if name.endswith("hip"):
                hip_ok = False
            continue
        value = visibility[name]
        if value is not None and value < float(visibility_threshold):
            lower_low_visibility = True
            if name.endswith("knee") or name.endswith("ankle"):
                knee_or_ankle_impaired = True
            if name.endswith("hip"):
                hip_ok = False
    if lower_low_visibility:
        reasons.append("LOW_LOWER_BODY_VISIBILITY")

    source_reason = str(evidence.get("reason", ""))
    if source_reason in {"AMBIGUOUS_LOWER_BODY_GEOMETRY", "DEGENERATE_KNEE_GEOMETRY"}:
        reasons.append("INSUFFICIENT_ACTIVITY_GEOMETRY")
    if evidence.get("candidate_state") is not None or str(getattr(subject, "activity_raw_state", "UNKNOWN")).upper() in {"SITTING", "STANDING"}:
        reasons.append("ACTIVITY_TRANSITION_UNCERTAIN")

    desk_suspected = bool(
        bool(getattr(subject, "core_valid", False))
        and hip_ok
        and knee_or_ankle_impaired
    )
    if desk_suspected:
        reasons.append("DESK_OCCLUSION_SUSPECTED")
    if not reasons:
        reasons.append("INSUFFICIENT_ACTIVITY_GEOMETRY")
    return {"reason_codes": list(dict.fromkeys(reasons)), "desk_occlusion_suspected": desk_suspected}


def _v4_status(subject: Any) -> tuple[str, bool, str | None]:
    result = dict(getattr(subject, "v4_result", {}) or {})
    prediction = getattr(subject, "v4_prediction", None)
    if prediction is not None:
        return "EXECUTED", True, None
    if str(result.get("status", "")).upper() == "ERROR":
        return "ERROR", False, str(result.get("detail", "V4_ERROR"))
    activity = str(getattr(subject, "activity_state", "UNKNOWN")).upper()
    if activity == "STANDING":
        return "SKIPPED", False, "STANDING_POSTURE_MONITOR_PAUSED"
    if activity == "UNKNOWN":
        return "SKIPPED", False, "ACTIVITY_UNKNOWN_POSTURE_MONITOR_PAUSED"
    return "SKIPPED", False, "POSE_OR_CORE_INVALID"


def _events_for_subject(events: Iterable[Mapping[str, Any]], subject_id: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for event in events:
        if str(event.get("subject_id", "")) == str(subject_id):
            result.append(_json_safe(dict(event)))
    return result


def build_subject_observation(
    subject: Any,
    *,
    session_id: str,
    timestamp_seconds: float,
    timestamp_utc: str,
    frame_index: int,
    source_health: Mapping[str, Any] | None,
    inference_fps: float | None,
    frame_age_ms: float | None,
    events: Iterable[Mapping[str, Any]] = (),
    visibility_threshold: float = 0.5,
) -> dict[str, Any]:
    """Build one anonymous shadow observation from an existing PoseSubject."""

    visibility, present = landmark_visibility(subject.landmarks)
    unknown = derive_unknown_reasons(subject, visibility_threshold=visibility_threshold)
    status, executed, skipped_reason = _v4_status(subject)
    temporal = _json_safe(dict(getattr(subject, "temporal_state", {}) or {}))
    return _json_safe({
        "schema_version": SHADOW_SCHEMA_VERSION,
        "record_type": "person_observation",
        "session_id": str(session_id),
        "timestamp_seconds": float(timestamp_seconds),
        "timestamp_utc": str(timestamp_utc),
        "frame_index": int(frame_index),
        "track_id": str(subject.subject_id),
        "source": safe_source_metadata(source_health),
        "source_state": safe_source_metadata(source_health).get("source_status"),
        "frame_age_ms": None if frame_age_ms is None else float(frame_age_ms),
        "inference_fps": None if inference_fps is None else float(inference_fps),
        "pose_detected": str(getattr(subject, "tracking_state", "")) in {"NEW", "ACTIVE"},
        "core_valid": bool(getattr(subject, "core_valid", False)),
        "pose_quality": float(getattr(subject, "pose_quality", 0.0)),
        "landmark_visibility": visibility,
        "landmark_present": present,
        "activity": {
            "raw_state": str(getattr(subject, "activity_raw_state", "UNKNOWN")),
            "state": str(getattr(subject, "activity_state", "UNKNOWN")),
            "confidence": float(getattr(subject, "activity_confidence", 0.0)),
            "method": str(getattr(subject, "activity_method", "NOT_EVALUATED")),
            "evidence": _json_safe(dict(getattr(subject, "activity_evidence", {}) or {})),
            "unknown_reasons": unknown["reason_codes"],
            "desk_occlusion_suspected": unknown["desk_occlusion_suspected"],
        },
        "v4": {
            "execution_status": status,
            "executed": executed,
            "skipped_reason": skipped_reason,
            "prediction": None if getattr(subject, "v4_prediction", None) is None else str(subject.v4_prediction),
            "probabilities": _json_safe(dict(getattr(subject, "v4_probabilities", {}) or {})),
        },
        "geometry": _json_safe(dict(getattr(subject, "geometry", {}) or {})),
        "temporal": temporal,
        "tracking": {
            "state": str(getattr(subject, "tracking_state", "UNKNOWN")),
            "last_seen_at": getattr(subject, "last_seen_at", None),
            "lifecycle_events": _events_for_subject(events, str(subject.subject_id)),
        },
        "labels": {"status": "NOT_A_LABEL", "human_label": None},
        "raw_video_recorded": False,
        "snapshot_recorded": False,
    })


def build_frame_observation(
    *,
    session_id: str,
    timestamp_seconds: float,
    timestamp_utc: str,
    frame_index: int,
    source_health: Mapping[str, Any] | None,
    frame_age_ms: float | None,
    inference_fps: float | None,
    inference_time_ms: float | None,
    subject_count: int,
    events: Iterable[Mapping[str, Any]],
    detected_pose_count: int | None = None,
    dropped_due_to_capacity: int = 0,
) -> dict[str, Any]:
    """Record frame/source telemetry without storing pixels or a raw URL."""

    return _json_safe({
        "schema_version": SHADOW_SCHEMA_VERSION,
        "record_type": "frame_observation",
        "session_id": str(session_id),
        "timestamp_seconds": float(timestamp_seconds),
        "timestamp_utc": str(timestamp_utc),
        "frame_index": int(frame_index),
        "source": safe_source_metadata(source_health),
        "source_state": safe_source_metadata(source_health).get("source_status"),
        "frame_age_ms": None if frame_age_ms is None else float(frame_age_ms),
        "inference_fps": None if inference_fps is None else float(inference_fps),
        "inference_time_ms": None if inference_time_ms is None else float(inference_time_ms),
        "detected_pose_count": None if detected_pose_count is None else int(detected_pose_count),
        "active_track_count": int(subject_count),
        "dropped_due_to_capacity": int(dropped_due_to_capacity),
        "events": [_json_safe(dict(event)) for event in events],
        "raw_video_recorded": False,
    })


class RotatingJsonlWriter:
    """A bounded append-only writer that rotates then stops instead of deleting.

    Stopping at the configured cap is deliberate: shadow mode must never
    silently erase a difficult observation period to continue indefinitely.
    The runner reports ``log_capacity_reached`` in its final manifest.
    """

    def __init__(self, directory: str | Path, stem: str, *, max_bytes: int, max_files: int, flush_seconds: float = 5.0) -> None:
        if int(max_bytes) < 1024:
            raise ValueError("max_bytes must be at least 1024")
        if int(max_files) < 1:
            raise ValueError("max_files must be at least 1")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stem = str(stem)
        self.max_bytes = int(max_bytes)
        self.max_files = int(max_files)
        self.flush_seconds = max(0.1, float(flush_seconds))
        self._handle = None
        self._file_index = -1
        self._current_bytes = 0
        self._last_flush_seconds = 0.0
        self.records_written = 0
        self.bytes_written = 0
        self.capacity_reached = False
        self.files: list[str] = []

    def _open_next(self) -> bool:
        if self._file_index + 1 >= self.max_files:
            self.capacity_reached = True
            return False
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
        self._file_index += 1
        path = self.directory / f"{self.stem}_{self._file_index:03d}.jsonl"
        self._handle = path.open("w", encoding="utf-8", newline="\n")
        self._current_bytes = 0
        self.files.append(path.name)
        return True

    def write(self, record: Mapping[str, Any], *, now_seconds: float) -> bool:
        encoded = json.dumps(_json_safe(dict(record)), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        size = len(encoded.encode("utf-8"))
        if size > self.max_bytes:
            raise ValueError(f"one {self.stem} record exceeds max_bytes")
        if self._handle is None and not self._open_next():
            return False
        if self._current_bytes and self._current_bytes + size > self.max_bytes:
            if not self._open_next():
                return False
        assert self._handle is not None
        self._handle.write(encoded)
        self._current_bytes += size
        self.bytes_written += size
        self.records_written += 1
        if float(now_seconds) - self._last_flush_seconds >= self.flush_seconds:
            self.flush(now_seconds=now_seconds)
        return True

    def flush(self, *, now_seconds: float) -> None:
        if self._handle is not None:
            self._handle.flush()
        self._last_flush_seconds = float(now_seconds)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def summary(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "files": list(self.files),
            "records_written": int(self.records_written),
            "bytes_written": int(self.bytes_written),
            "max_bytes_per_file": int(self.max_bytes),
            "max_files": int(self.max_files),
            "capacity_reached": bool(self.capacity_reached),
            "retention_behavior": "ROTATE_THEN_STOP_NO_AUTOMATIC_DELETION",
        }


@dataclass
class ShadowReviewQueue:
    """Queues difficult anonymous observations for explicitly human labelling."""

    unknown_review_seconds: float = 10.0
    cooldown_seconds: float = 30.0
    near_boundary_margin: float = 0.15
    repeated_lower_body_threshold: int = 3
    _unknown_since: dict[str, float] = field(default_factory=dict)
    _lower_body_failures: dict[str, int] = field(default_factory=dict)
    _last_activity_raw: dict[str, str] = field(default_factory=dict)
    _last_v4_prediction: dict[str, str | None] = field(default_factory=dict)
    _last_risk_level: dict[str, str | None] = field(default_factory=dict)
    _last_emitted: dict[tuple[str, str], float] = field(default_factory=dict)

    def _eligible(self, track_id: str, kind: str, timestamp: float) -> bool:
        key = (track_id, kind)
        previous = self._last_emitted.get(key)
        if previous is not None and timestamp - previous < self.cooldown_seconds:
            return False
        self._last_emitted[key] = timestamp
        return True

    def _item(self, record: Mapping[str, Any], kind: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return _json_safe({
            "schema_version": SHADOW_SCHEMA_VERSION,
            "record_type": "human_review_queue",
            "review_status": "NEEDS_HUMAN_LABEL",
            "review_kind": str(kind),
            "session_id": record.get("session_id"),
            "timestamp_seconds": record.get("timestamp_seconds"),
            "timestamp_utc": record.get("timestamp_utc"),
            "track_id": record.get("track_id"),
            "frame_index": record.get("frame_index"),
            "details": dict(details or {}),
            "snapshot_recorded": False,
            "raw_video_recorded": False,
            "automatic_label": None,
        })

    def inspect(self, record: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Return de-duplicated review items; never create a label."""

        timestamp = float(record["timestamp_seconds"])
        track_id = str(record["track_id"])
        activity = dict(record.get("activity", {}))
        raw_state = str(activity.get("raw_state", "UNKNOWN"))
        state = str(activity.get("state", "UNKNOWN"))
        reasons = set(activity.get("unknown_reasons", []) or [])
        output: list[dict[str, Any]] = []

        if state == "UNKNOWN":
            start = self._unknown_since.setdefault(track_id, timestamp)
            if timestamp - start >= self.unknown_review_seconds and self._eligible(track_id, "PROLONGED_UNKNOWN", timestamp):
                output.append(self._item(record, "PROLONGED_UNKNOWN", {"duration_seconds": timestamp - start, "unknown_reasons": sorted(reasons)}))
        else:
            self._unknown_since.pop(track_id, None)

        lower_failure = bool(reasons & ({"LOW_LOWER_BODY_VISIBILITY"} | set(MISSING_CODE.values())))
        self._lower_body_failures[track_id] = self._lower_body_failures.get(track_id, 0) + 1 if lower_failure else 0
        if self._lower_body_failures[track_id] >= self.repeated_lower_body_threshold and self._eligible(track_id, "REPEATED_LOWER_BODY_UNAVAILABLE", timestamp):
            output.append(self._item(record, "REPEATED_LOWER_BODY_UNAVAILABLE", {"consecutive_observations": self._lower_body_failures[track_id], "unknown_reasons": sorted(reasons)}))

        previous_raw = self._last_activity_raw.get(track_id)
        if previous_raw in {"SITTING", "STANDING"} and raw_state in {"SITTING", "STANDING"} and raw_state != previous_raw and self._eligible(track_id, "ACTIVITY_RAW_FLIP", timestamp):
            output.append(self._item(record, "ACTIVITY_RAW_FLIP", {"previous_raw_state": previous_raw, "raw_state": raw_state}))
        self._last_activity_raw[track_id] = raw_state

        lifecycle = (record.get("tracking", {}) or {}).get("lifecycle_events", []) or []
        for event in lifecycle:
            name = str(event.get("event", ""))
            if name in {"PERSON_TEMPORARILY_LOST", "PERSON_REACQUIRED"} and self._eligible(track_id, name, timestamp):
                output.append(self._item(record, name, {"lifecycle_event": dict(event)}))

        v4 = dict(record.get("v4", {}) or {})
        probabilities = dict(v4.get("probabilities", {}) or {})
        tup = probabilities.get("TUP")
        tlb = probabilities.get("TLB")
        if v4.get("executed") and tup is not None and tlb is not None and abs(float(tup) - float(tlb)) <= self.near_boundary_margin and self._eligible(track_id, "V4_NEAR_DECISION_BOUNDARY", timestamp):
            output.append(self._item(record, "V4_NEAR_DECISION_BOUNDARY", {"TUP": float(tup), "TLB": float(tlb), "margin": self.near_boundary_margin}))
        prediction = None if v4.get("prediction") is None else str(v4["prediction"])
        previous_prediction = self._last_v4_prediction.get(track_id)
        if prediction is not None and previous_prediction is not None and prediction != previous_prediction and self._eligible(track_id, "V4_RAW_PREDICTION_FLIP", timestamp):
            output.append(self._item(record, "V4_RAW_PREDICTION_FLIP", {"previous_prediction": previous_prediction, "prediction": prediction}))
        self._last_v4_prediction[track_id] = prediction

        directional = dict((record.get("geometry", {}) or {}).get("directional_indicators", {}) or {})
        active_geometry = [name for name in ("forward", "left", "right") if bool(directional.get(name))]
        if prediction == "TUP" and active_geometry and self._eligible(track_id, "V4_GEOMETRY_DIAGNOSTIC_DISAGREEMENT", timestamp):
            output.append(self._item(record, "V4_GEOMETRY_DIAGNOSTIC_DISAGREEMENT", {"v4_prediction": prediction, "geometry_indicators": active_geometry, "note": "Geometry is experimental and is not a ground-truth contradiction."}))

        risk_level = str((record.get("temporal", {}) or {}).get("risk_level", ""))
        previous_risk = self._last_risk_level.get(track_id)
        if risk_level in {"WARNING", "HIGH_DURATION_RISK"} and previous_risk not in {"WARNING", "HIGH_DURATION_RISK"} and self._eligible(track_id, "WARNING_TRANSITION", timestamp):
            output.append(self._item(record, "WARNING_TRANSITION", {"previous_risk_level": previous_risk, "risk_level": risk_level}))
        self._last_risk_level[track_id] = risk_level
        return output


def _read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in {path.name}:{line_number}") from exc
                if isinstance(record, dict):
                    records.append(record)
    return records


def _percent(value: int | float, total: int | float) -> float | None:
    return None if not total else float(value) * 100.0 / float(total)


def _distribution(values: Iterable[str], *, denominator: int | None = None) -> dict[str, dict[str, float | int | None]]:
    counts = Counter(str(item) for item in values)
    total = int(denominator if denominator is not None else sum(counts.values()))
    return {key: {"count": int(count), "percentage": _percent(count, total)} for key, count in sorted(counts.items())}


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not cleaned:
        return {"median": None, "p95": None}
    return {"median": float(median(cleaned)), "p95": float(np.percentile(cleaned, 95))}


def _flip_metrics(observations: list[Mapping[str, Any]], selector) -> dict[str, float | int | None]:
    per_track: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in observations:
        per_track[str(item.get("track_id"))].append(item)
    flips = 0
    opportunities = 0
    for items in per_track.values():
        previous = None
        for item in sorted(items, key=lambda current: float(current.get("timestamp_seconds", 0.0))):
            current = selector(item)
            if current is None:
                continue
            if previous is not None:
                opportunities += 1
                flips += int(current != previous)
            previous = current
    return {"count": int(flips), "opportunities": int(opportunities), "rate_per_100_comparisons": _percent(flips * 100, opportunities)}


def _bilateral_available(observations: list[Mapping[str, Any]], names: tuple[str, ...], threshold: float) -> dict[str, float | int | None]:
    count = 0
    for record in observations:
        values = dict(record.get("landmark_visibility", {}) or {})
        present = dict(record.get("landmark_present", {}) or {})
        ok = all(bool(present.get(name)) and (values.get(name) is None or float(values[name]) >= threshold) for name in names)
        count += int(ok)
    return {"available_count": int(count), "available_percentage": _percent(count, len(observations))}


def _bottlenecks(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rank only conditions supported by recorded telemetry, not assumptions."""

    observations = int(summary["observation_count"])
    activity = summary["activity"]["distribution"]
    unknown_pct = float((activity.get("UNKNOWN") or {}).get("percentage") or 0.0)
    landmarks = summary["landmark_availability"]
    knee_pct = float((landmarks.get("knees_bilateral") or {}).get("available_percentage") or 0.0)
    ankle_pct = float((landmarks.get("ankles_bilateral") or {}).get("available_percentage") or 0.0)
    pose_pct = float(summary["pose"]["detection_rate_percentage"] or 0.0)
    core_pct = float(summary["pose"]["core_validity_rate_percentage"] or 0.0)
    events = summary["tracking"]["lifecycle_events"]
    source = summary["source"]
    ranked: list[tuple[float, str, list[str]]] = []
    if observations and unknown_pct >= 20.0 and min(knee_pct, ankle_pct) < 70.0:
        ranked.append((unknown_pct + (100.0 - min(knee_pct, ankle_pct)), "LOWER_BODY_VISIBILITY", [f"activity UNKNOWN {unknown_pct:.1f}%", f"knees bilateral {knee_pct:.1f}%", f"ankles bilateral {ankle_pct:.1f}%"]))
    if observations and (pose_pct < 90.0 or core_pct < 80.0):
        ranked.append((200.0 - pose_pct - core_pct, "POSE", [f"pose detected {pose_pct:.1f}%", f"core valid {core_pct:.1f}%"]))
    frame_count = max(1, int(summary["frame_count"]))
    tracking_disruptions = int(events.get("PERSON_TEMPORARILY_LOST", 0)) + int(events.get("PERSON_REACQUIRED", 0))
    if tracking_disruptions > 0:
        ranked.append((100.0 * tracking_disruptions / frame_count, "TRACKING", [f"temporary lost + reacquired {tracking_disruptions} across {frame_count} inference frames"]))
    if observations and unknown_pct >= 20.0 and knee_pct >= 70.0 and ankle_pct >= 70.0:
        ranked.append((unknown_pct, "ACTIVITY_GATE", [f"activity UNKNOWN {unknown_pct:.1f}% despite bilateral knee/ankle availability >=70%"]))
    if int(summary["v4"]["executed_count"]) > 0 and int(summary["v4"]["raw_prediction_flips"]["count"]) > 0:
        ranked.append((float(summary["v4"]["raw_prediction_flips"]["rate_per_100_comparisons"] or 0.0), "POSTURE_CLASSIFIER", [f"raw V4 flips {summary['v4']['raw_prediction_flips']['count']}", "Engineering stability signal only; not an accuracy result."]))
    if int(summary["temporal"]["warning_transitions"]) > 0:
        ranked.append((float(summary["temporal"]["warning_transitions"]), "TEMPORAL_LOGIC", [f"warning transitions {summary['temporal']['warning_transitions']}; requires attended review before any change."]))
    if int(source.get("reconnects", 0)) or int(source.get("decode_failures", 0)) or int(source.get("stale_events", 0)):
        ranked.append((float(int(source.get("reconnects", 0)) + int(source.get("decode_failures", 0)) + int(source.get("stale_events", 0))), "NETWORK", [f"reconnects {source.get('reconnects', 0)}", f"decode failures {source.get('decode_failures', 0)}", f"stale events {source.get('stale_events', 0)}"]))
    ranked.append((0.01, "INSUFFICIENT_GROUND_TRUTH", ["Shadow telemetry contains no human labels by design; it cannot measure activity or posture accuracy."]))
    return [
        {"rank": index + 1, "category": category, "evidence": evidence}
        for index, (_score, category, evidence) in enumerate(sorted(ranked, key=lambda item: (-item[0], item[1])))
    ]


def _collection_recommendation(bottlenecks: list[Mapping[str, Any]]) -> dict[str, Any]:
    primary = str(bottlenecks[0]["category"]) if bottlenecks else "INSUFFICIENT_GROUND_TRUTH"
    recommendations = {
        "LOWER_BODY_VISIBILITY": "เก็บ human-labelled ชุดเล็ก 3 คนแบบไม่ระบุตัวตน: ยืน/นั่งอย่างละ 20–30 วินาที ในตำแหน่งที่เห็นขาชัด 1 จุด และหลังโต๊ะ 1 จุดต่อคน เพื่อแยกปัญหามุมกล้อง/โต๊ะจาก activity gate.",
        "POSE": "เก็บ human-labelled ชุดสั้นที่ครอบคลุมระยะและตำแหน่งกล้องต่างกัน โดยบันทึกเฉพาะ telemetry และฉลากที่ผู้สังเกตให้ ไม่ใช้ prediction เป็นฉลาก.",
        "TRACKING": "ทดสอบแบบมีผู้สังเกต 2–3 คน เดินเข้า/ออกและตัดกันช่วงสั้น แล้วทำฉลากเหตุการณ์ track lost/reacquired ด้วยคน.",
        "ACTIVITY_GATE": "เก็บ human-labelled นั่ง/ยืนช่วงสั้นที่มองเห็นขาครบ เพื่อทดสอบ geometry gate ก่อนพิจารณาแก้ threshold หรือสร้างชุดข้อมูล.",
        "POSTURE_CLASSIFIER": "เก็บตัวอย่างนั่ง TUP/TLB ที่มนุษย์ตรวจทานจากตำแหน่งกล้องจริงก่อนประเมินว่าต้องปรับโมเดลหรือไม่.",
        "TEMPORAL_LOGIC": "ตรวจทานรายการ WARNING ใน review queue โดยมนุษย์ก่อนปรับเวลาหน่วงของ temporal engine.",
        "NETWORK": "เก็บ telemetry ช่วงสั้นในเวลาที่พบปัญหาเครือข่าย แล้วเทียบ reconnect/decode/stale กับเวลาจริงก่อนเปลี่ยนค่ากล้องหรือเครือข่าย.",
        "INSUFFICIENT_GROUND_TRUTH": "เริ่มชุด human-labelled ที่เล็กที่สุด: 1–3 คน, กิจกรรม/ท่าที่กำหนดชัดเจน, ผู้สังเกตบันทึกฉลากเอง และไม่ใช้ผล AI เป็นฉลาก.",
    }
    return {"primary_driver": primary, "smallest_next_collection": recommendations.get(primary, recommendations["INSUFFICIENT_GROUND_TRUTH"])}


def analyze_shadow_session(session_dir: str | Path, *, visibility_threshold: float = 0.5) -> dict[str, Any]:
    """Analyze a finished shadow session without contacting a camera or model."""

    root = Path(session_dir)
    observations = _read_jsonl(root.glob("observations_*.jsonl"))
    frames = _read_jsonl(root.glob("frames_*.jsonl"))
    reviews = _read_jsonl(root.glob("review_queue_*.jsonl"))
    manifest_path = root / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    timestamps = [float(item.get("timestamp_seconds", 0.0)) for item in (frames or observations)]
    activities = [str((item.get("activity") or {}).get("state", "UNKNOWN")) for item in observations]
    reason_counts = Counter(reason for item in observations for reason in ((item.get("activity") or {}).get("unknown_reasons", []) or []))
    prediction_values = [str((item.get("v4") or {}).get("prediction")) for item in observations if (item.get("v4") or {}).get("executed") and (item.get("v4") or {}).get("prediction") is not None]
    confidences = [max(float(value) for value in ((item.get("v4") or {}).get("probabilities", {}) or {}).values()) for item in observations if (item.get("v4") or {}).get("executed") and (item.get("v4") or {}).get("probabilities")]
    lifecycle = Counter(str(event.get("event")) for frame in frames for event in (frame.get("events", []) or []) if str(event.get("event", "")).startswith("PERSON_"))
    source_events = [item for item in _read_jsonl(root.glob("source_events_*.jsonl"))]
    health_values = [dict((frame.get("source") or {})) for frame in frames]
    final_health = dict(manifest.get("final_source_health") or {})
    latest_health = health_values[-1] if health_values else final_health
    state_events = sum(1 for item in source_events if str(item.get("source_state")) == "SOURCE_STALE")
    pose_detected = sum(bool(item.get("pose_detected")) for item in observations)
    core_valid = sum(bool(item.get("core_valid")) for item in observations)
    risk_levels = [str((item.get("temporal") or {}).get("risk_level", "")) for item in observations]
    warning_transitions = sum(bool((item.get("temporal") or {}).get("warning_event")) for item in observations)
    result: dict[str, Any] = {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "session_dir": str(root),
        "session_id": manifest.get("session_id") or (observations[0].get("session_id") if observations else None),
        "session_duration_seconds": None if not timestamps else max(timestamps) - min(timestamps),
        "observation_count": len(observations),
        "frame_count": len(frames),
        "review_queue_count": len(reviews),
        "tracking": {
            "unique_anonymous_track_slots": sorted({str(item.get("track_id")) for item in observations}),
            "max_simultaneous_tracks": max((int(item.get("active_track_count", 0)) for item in frames), default=0),
            "max_detected_pose_count": max((int(item.get("detected_pose_count") or 0) for item in frames), default=0),
            "dropped_due_to_capacity": int(sum(int(item.get("dropped_due_to_capacity") or 0) for item in frames)),
            "lifecycle_events": dict(lifecycle),
            "suspected_fragmentation": {
                "appeared": int(lifecycle.get("PERSON_APPEARED", 0)),
                "temporarily_lost": int(lifecycle.get("PERSON_TEMPORARILY_LOST", 0)),
                "reacquired": int(lifecycle.get("PERSON_REACQUIRED", 0)),
                "exited": int(lifecycle.get("PERSON_EXITED", 0)),
                "interpretation": "Proxy only; ID switches require human track ground truth and are not measured.",
            },
        },
        "pose": {
            "detection_rate_percentage": _percent(pose_detected, len(observations)),
            "core_validity_rate_percentage": _percent(core_valid, len(observations)),
        },
        "landmark_availability": {
            "nose": _bilateral_available(observations, ("nose",), visibility_threshold),
            "shoulders_bilateral": _bilateral_available(observations, ("left_shoulder", "right_shoulder"), visibility_threshold),
            "hips_bilateral": _bilateral_available(observations, ("left_hip", "right_hip"), visibility_threshold),
            "knees_bilateral": _bilateral_available(observations, ("left_knee", "right_knee"), visibility_threshold),
            "ankles_bilateral": _bilateral_available(observations, ("left_ankle", "right_ankle"), visibility_threshold),
        },
        "activity": {
            "distribution": _distribution(activities, denominator=len(observations)),
            "raw_state_flips": _flip_metrics(observations, lambda item: (item.get("activity") or {}).get("raw_state")),
        },
        "unknown_reasons": {
            "denominator_unknown_observations": int(sum(value == "UNKNOWN" for value in activities)),
            "ranked": [
                {"reason": reason, "count": int(count), "percentage_of_unknown_observations": _percent(count, sum(value == "UNKNOWN" for value in activities))}
                for reason, count in reason_counts.most_common()
            ],
        },
        "v4": {
            "executed_count": len(prediction_values),
            "prediction_distribution": _distribution(prediction_values),
            "confidence": _quantiles(confidences),
            "raw_prediction_flips": _flip_metrics(observations, lambda item: (item.get("v4") or {}).get("prediction") if (item.get("v4") or {}).get("executed") else None),
        },
        "temporal": {
            "observe_observation_count": int(sum(level == "OBSERVE" for level in risk_levels)),
            "warning_observation_count": int(sum(level in {"WARNING", "HIGH_DURATION_RISK"} for level in risk_levels)),
            "warning_transitions": int(warning_transitions),
            "longest_poor_posture_duration_seconds": max((float((item.get("temporal") or {}).get("duration_seconds") or 0.0) for item in observations), default=0.0),
        },
        "source": {
            "frame_age_ms": _quantiles(item.get("frame_age_ms") for item in frames),
            "inference_fps": _quantiles(item.get("inference_fps") for item in frames),
            "reconnects": int(final_health.get("reconnect_count", latest_health.get("reconnect_count", 0)) or 0),
            "decode_failures": int(final_health.get("decode_failures", latest_health.get("decode_failures", 0)) or 0),
            "stale_events": int(state_events + sum(str(item.get("source_state")) == "SOURCE_STALE" for item in frames)),
        },
        "safety": {
            "raw_video_recorded": False,
            "automatic_labels_created": False,
            "review_items_require_human_label": all(item.get("review_status") == "NEEDS_HUMAN_LABEL" for item in reviews),
        },
    }
    result["bottlenecks"] = _bottlenecks(result)
    result["data_collection_recommendation"] = _collection_recommendation(result["bottlenecks"])
    return _json_safe(result)


def render_analysis_th(analysis: Mapping[str, Any]) -> str:
    """Create a concise Thai Markdown report from measured shadow telemetry."""

    lines = [
        "# รายงาน Shadow Observation", "",
        f"Session: `{analysis.get('session_id')}`",
        f"ระยะเวลา: `{analysis.get('session_duration_seconds')}` วินาที | observations: `{analysis.get('observation_count')}` | inference frames: `{analysis.get('frame_count')}`",
        "", "## ข้อจำกัดสำคัญ", "",
        "ข้อมูลนี้เป็น telemetry วิศวกรรมแบบไม่ระบุตัวตน ไม่บันทึกวิดีโอดิบ และไม่ใช่ ground truth หรือฉลากสำหรับ train โมเดล.",
        "", "## ผลที่วัดได้", "",
        f"- pose detected: `{analysis['pose']['detection_rate_percentage']}`% | core valid: `{analysis['pose']['core_validity_rate_percentage']}`%",
        f"- activity: `{analysis['activity']['distribution']}`",
        f"- V4 executed: `{analysis['v4']['executed_count']}` | distribution: `{analysis['v4']['prediction_distribution']}`",
        f"- source frame age: `{analysis['source']['frame_age_ms']}` ms | inference FPS: `{analysis['source']['inference_fps']}`",
        "", "## Landmark availability", "",
    ]
    for name, values in analysis["landmark_availability"].items():
        lines.append(f"- {name}: `{values['available_percentage']}`% ({values['available_count']})")
    lines.extend(["", "## UNKNOWN reasons", ""])
    for item in analysis["unknown_reasons"]["ranked"]:
        lines.append(f"- `{item['reason']}`: {item['count']} ({item['percentage_of_unknown_observations']}% ของ UNKNOWN observations)")
    lines.extend(["", "## Bottleneck ranking", ""])
    for item in analysis["bottlenecks"]:
        lines.append(f"{item['rank']}. **{item['category']}** — {'; '.join(item['evidence'])}")
    recommendation = analysis["data_collection_recommendation"]
    lines.extend(["", "## ชุดเก็บข้อมูลที่เล็กที่สุดที่แนะนำ", "", f"ตัวขับหลัก: **{recommendation['primary_driver']}**", "", recommendation["smallest_next_collection"], ""])
    return "\n".join(lines)
