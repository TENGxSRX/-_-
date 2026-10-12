"""Backward-compatible JSON telemetry for anonymous multi-person frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.posture_ai.inference.video_source import redact_url

from .schema import MultiPersonFrameResult


def _safe_source_metadata(source_metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep source telemetry useful while excluding credential-bearing fields."""

    if source_metadata is None:
        return None
    allowed = {
        "source_type",
        "source",
        "camera_id",
        "source_status",
        "connected",
        "frames_received",
        "decode_failures",
        "consecutive_failures",
        "reported_fps",
        "effective_received_fps",
        "resolution",
        "last_frame_age_seconds",
        "reconnect_count",
        "late_frame_count",
        "overwritten_frame_count",
        "backend",
        "buffer_control_requested",
        "buffer_control_applied",
        "network_packet_loss",
    }
    safe = {str(key): value for key, value in source_metadata.items() if str(key) in allowed}
    if "source" in safe:
        safe["source"] = redact_url(str(safe["source"]))
    return safe


def _result_to_event(
    result,
    *,
    frame_index: int | None = None,
    inference_skipped: bool = False,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not hasattr(result, "to_dict"):
        raise TypeError("result must expose to_dict()")
    event = result.to_dict()
    if frame_index is not None:
        event["frame_index"] = int(frame_index)
    event["inference_skipped"] = bool(inference_skipped)
    safe_source = _safe_source_metadata(source_metadata)
    if safe_source is not None:
        event["source_health"] = safe_source
    return event


def frame_to_event(
    result: MultiPersonFrameResult,
    *,
    frame_index: int | None = None,
    inference_skipped: bool = False,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one frame while clearly marking skipped inference frames."""

    return _result_to_event(
        result,
        frame_index=frame_index,
        inference_skipped=inference_skipped,
        source_metadata=source_metadata,
    )


def seat_frame_to_event(
    result,
    *,
    frame_index: int | None = None,
    inference_skipped: bool = False,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize a seat-aware frame while retaining anonymous debug subjects."""

    event = _result_to_event(
        result,
        frame_index=frame_index,
        inference_skipped=inference_skipped,
        source_metadata=source_metadata,
    )
    event["tracking_mode"] = "seat"
    return event


def legacy_single_person_to_event(record: dict[str, Any], *, camera_id: str = "CAM01") -> dict[str, Any]:
    """Wrap an old single-person record without changing its original fields."""

    subject = {
        "subject_id": "PERSON_01",
        "tracking_state": "ACTIVE" if record.get("pose_visible", True) else "TEMPORARILY_LOST",
        "core_valid": bool(record.get("pose_visible", True)),
        "pose_quality": record.get("pose_quality", record.get("visibility", 0.0)),
        "v4_prediction": record.get("direction", record.get("label")),
        "v4_probabilities": record.get("probabilities", {}),
        "geometry": record.get("geometry", {}),
        "temporal_state": record.get("temporal_state", record),
    }
    return {
        "timestamp_seconds": float(record.get("timestamp_seconds", record.get("timestamp", 0.0))),
        "camera_id": camera_id,
        "tracking_status": "LEGACY_SINGLE_PERSON_COMPATIBILITY",
        "subject_count": 1,
        "subjects": [subject],
        "legacy_record": dict(record),
        "inference_skipped": bool(record.get("inference_skipped", False)),
    }


class MultiPersonTelemetry:
    """Small in-memory event buffer suitable for a low-spec prototype."""

    def __init__(self, *, max_events: int = 2000) -> None:
        if int(max_events) < 1:
            raise ValueError("max_events must be at least 1")
        self.max_events = int(max_events)
        self.events: list[dict[str, Any]] = []

    def add_frame(
        self,
        result: MultiPersonFrameResult,
        *,
        frame_index: int | None = None,
        inference_skipped: bool = False,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = _result_to_event(
            result,
            frame_index=frame_index,
            inference_skipped=inference_skipped,
            source_metadata=source_metadata,
        )
        self.events.append(event)
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]
        return event

    def write_jsonl(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
