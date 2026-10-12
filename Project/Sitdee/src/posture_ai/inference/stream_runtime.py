"""Source-agnostic V4 runtime for future RTSP/file/USB sessions.

This is a conservative companion to the existing webcam validation runner.
It reuses the frozen V4 processor and temporal engine, while keeping source
health fields separate from posture-risk fields.  It is not started by tests
or setup scripts.
"""

from __future__ import annotations

import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig
from src.posture_ai.inference.pipeline import create_pose_landmarker
from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner
from src.posture_ai.inference.video_source import (
    SOURCE_CLOSED,
    SOURCE_FAILED,
    VideoSource,
)
from src.posture_ai.inference.webcam_v1 import (
    LiveFrameProcessor,
    LiveSessionAccumulator,
    draw_live_overlay,
    prepare_display_frame,
    should_process_frame,
    write_live_outputs,
)


DEFAULT_RTSP_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "rtsp_profiles.yaml"
RTSP_PRESETS = {
    "RTSP_DESKTOP": {
        "process_every": 1,
        "inference_max_width": None,
        "inference_max_height": None,
        "stale_after_seconds": 2.0,
        "low_latency_buffer_request": True,
    },
    "RTSP_BALANCED": {
        "process_every": 2,
        "inference_max_width": 1280,
        "inference_max_height": 720,
        "stale_after_seconds": 2.0,
        "low_latency_buffer_request": True,
    },
    "RTSP_LOW_LATENCY": {
        "process_every": 1,
        "inference_max_width": 960,
        "inference_max_height": 540,
        "stale_after_seconds": 1.0,
        "low_latency_buffer_request": True,
        "latest_frame_only": True,
        "clean_ui": True,
    },
}


def resolve_rtsp_preset(name: str = "RTSP_DESKTOP", *, config_path: str | Path = DEFAULT_RTSP_CONFIG) -> dict[str, Any]:
    preset_name = str(name).upper()
    values = dict(RTSP_PRESETS.get(preset_name, {}))
    path = Path(config_path)
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        configured = (raw.get("presets") or {}).get(preset_name)
        if isinstance(configured, dict):
            values.update(configured)
    if not values:
        raise ValueError(f"unknown RTSP runtime preset: {name}")
    values["name"] = preset_name
    values["process_every"] = int(values.get("process_every", 1))
    if values["process_every"] < 1:
        raise ValueError("RTSP process_every must be at least 1")
    values["stale_after_seconds"] = float(values.get("stale_after_seconds", 2.0))
    return values


def resize_for_inference(frame: np.ndarray, *, max_width: int | None = None, max_height: int | None = None) -> tuple[np.ndarray, float]:
    """Create an optional inference copy without changing the source frame."""

    if frame is None or frame.size == 0:
        return frame, 1.0
    limits = [float(value) for value in (max_width, max_height) if value is not None and float(value) > 0]
    if not limits:
        return frame, 1.0
    height, width = frame.shape[:2]
    scale = min(1.0, min(limits[0] / width, limits[-1] / height) if len(limits) == 2 else limits[0] / (width if max_width is not None else height))
    if scale >= 1.0:
        return frame, 1.0
    resized = cv2.resize(frame, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    return resized, float(scale)


def resize_for_display(frame: np.ndarray, *, max_width: int = 1280, max_height: int = 720) -> tuple[np.ndarray, float]:
    """Resize only the UI copy while preserving the source frame unchanged."""

    return resize_for_inference(frame, max_width=max_width, max_height=max_height)


def _source_fields(health: dict[str, Any], *, source_type: str, camera_id: str | None, frame_scale: float | None = None) -> dict[str, Any]:
    fields = {
        "source_type": source_type,
        "camera_id": camera_id,
        "source_status": health.get("source_status"),
        "source_resolution": health.get("resolution"),
        "source_reported_fps": health.get("reported_fps"),
        "source_effective_fps": health.get("effective_received_fps"),
        "source_reconnect_count": health.get("reconnect_count"),
        "source_last_frame_age_seconds": health.get("last_frame_age_seconds"),
    }
    if frame_scale is not None:
        fields["inference_frame_scale"] = float(frame_scale)
    return fields


def run_source_posture_session(
    source: str | int,
    *,
    model_dir: str | Path = "outputs/models/rgb_native_v4",
    pose_model: str | Path = "metadata/pose_landmarker_heavy.task",
    config_path: str | Path = "configs/posture_engine.yaml",
    rtsp_config_path: str | Path = DEFAULT_RTSP_CONFIG,
    profile: str | None = None,
    camera_id: str | None = None,
    preset: str = "RTSP_DESKTOP",
    max_frames: int | None = None,
    output_root: str | Path = "outputs/live",
    max_reconnects: int = 5,
    display: bool = False,
    window_name: str = "Office Posture - Source Runtime",
) -> dict[str, Any]:
    """Run the frozen V4 pipeline over a unified USB/file/RTSP source."""

    if max_frames is not None and int(max_frames) < 1:
        raise ValueError("max_frames must be positive when provided")
    runtime = resolve_rtsp_preset(preset, config_path=rtsp_config_path)
    config = PostureEngineConfig.from_yaml(config_path)
    video = VideoSource(
        source,
        camera_id=camera_id,
        profile=profile,
        config_path=rtsp_config_path,
        reconnect=True,
        max_reconnects=max_reconnects,
        stale_after_seconds=runtime["stale_after_seconds"],
        low_latency=bool(runtime.get("low_latency_buffer_request", False)),
    )
    if not video.open():
        raise RuntimeError(f"Source could not be opened: {video.spec.source_display}")

    runner = RGBNativeV4Runner(model_dir)
    landmarker = create_pose_landmarker(pose_model, running_mode="video")
    processor = LiveFrameProcessor(runner, landmarker, config)
    session = LiveSessionAccumulator(frame_interval_seconds=config.frame_interval_seconds)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []
    source_events: list[dict[str, Any]] = []
    frame_index = 0
    processed_frames = 0
    skipped_frames = 0
    last_record: dict[str, Any] | None = None
    last_image_pose = None
    started = time.perf_counter()
    last_timestamp = 0.0
    health_before_close: dict[str, Any] | None = None

    try:
        while max_frames is None or frame_index < max_frames:
            result = video.read()
            last_timestamp = max(last_timestamp, float(result.timestamp_seconds))
            if not result.ok or result.frame is None:
                event = {
                    "timestamp": float(result.timestamp_seconds),
                    "source_status": result.state,
                    "source_reason": result.reason,
                    "source_health": result.health,
                    "pose_recorded": False,
                }
                source_events.append(event)
                records.append(event)
                if result.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                    break
                continue

            frame = result.frame
            health = result.health
            if should_process_frame(frame_index, runtime["process_every"], last_record is not None):
                inference_frame, scale = resize_for_inference(
                    frame,
                    max_width=runtime.get("inference_max_width"),
                    max_height=runtime.get("inference_max_height"),
                )
                fps = float(health.get("effective_received_fps") or health.get("reported_fps") or 0.0)
                record, image_pose = processor.process(
                    inference_frame,
                    frame_index=frame_index,
                    timestamp_seconds=result.timestamp_seconds,
                    fps=fps,
                )
                record.update(_source_fields(health, source_type=video.source_type, camera_id=video.camera_id, frame_scale=scale))
                record["inference_skipped"] = False
                session.add(record)
                last_record = deepcopy(record)
                last_image_pose = image_pose
                processed_frames += 1
            else:
                record = deepcopy(last_record) if last_record is not None else {"timestamp": result.timestamp_seconds}
                record.update({
                    "frame": int(frame_index),
                    "timestamp": float(result.timestamp_seconds),
                    "inference_skipped": True,
                    "inference_skip_reason": "rtsp_runtime_process_every",
                    **_source_fields(health, source_type=video.source_type, camera_id=video.camera_id),
                })
                skipped_frames += 1
            records.append(record)

            if display:
                display_frame = prepare_display_frame(frame)
                annotated = draw_live_overlay(display_frame, record, last_image_pose, calibration=False, mirror=True)
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            frame_index += 1
    finally:
        landmarker.close()
        health_before_close = video.health()
        video.close()
        if display:
            cv2.destroyAllWindows()

    elapsed = max(0.0, time.perf_counter() - started)
    summary = session.summary(total_duration_seconds=max(last_timestamp, elapsed))
    final_health = health_before_close or video.health()
    summary.update({
        "session_id": session_id,
        "source_type": video.source_type,
        "camera_id": video.camera_id,
        "source": video.spec.source_display,
        "source_profile": video.spec.profile,
        "source_status": final_health.get("source_status"),
        "source_health": final_health,
        "source_health_event_count": len(source_events),
        "source_reconnect_count": final_health.get("reconnect_count", 0),
        "source_closed_cleanly": True,
        "source_stale_threshold_seconds": runtime["stale_after_seconds"],
        "stream_preset": runtime["name"],
        "displayed_frames": frame_index,
        "processed_frames": processed_frames,
        "inference_skipped_frames": skipped_frames,
        "software_status": "READY_FOR_HARDWARE_TEST",
        "hardware_status": "SOURCE_SESSION_COMPLETED" if records else "SOURCE_NOT_READY",
        "interpretation": "Descriptive engineering telemetry; source health is separate from posture risk and is not a medical score.",
    })
    paths = write_live_outputs(records, summary, output_root=output_root, session_id=session_id)
    return {"summary": summary, "paths": paths, "records": records, "source_health_events": source_events}


__all__ = ["RTSP_PRESETS", "resize_for_display", "resize_for_inference", "resolve_rtsp_preset", "run_source_posture_session"]
