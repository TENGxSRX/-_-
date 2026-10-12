"""Interactive, measured single-person RTSP AI Lab around frozen V4."""

from __future__ import annotations

import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput
from src.posture_ai.inference.pipeline import create_pose_landmarker
from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner
from src.posture_ai.inference.stream_runtime import resize_for_display, resize_for_inference, resolve_rtsp_preset
from src.posture_ai.inference.video_source import SOURCE_CLOSED, SOURCE_FAILED, VideoSource
from src.posture_ai.inference.webcam_v1 import (
    CORE_LANDMARKS,
    LiveFrameProcessor,
    LiveSessionAccumulator,
    SKELETON_CONNECTIONS,
    prepare_display_frame,
    write_live_outputs,
)
from src.posture_ai.pose.landmarks import LANDMARK_INDEX


def source_status_text(health: dict[str, Any]) -> str:
    """Return a UI-safe source health label; no URL or secret is included."""

    return str(health.get("source_status") or "SOURCE_UNKNOWN")


def profile_for_source(source: str | int, profile: str | None) -> str | None:
    """Apply a Hikvision profile only to an RTSP-like source."""

    raw = str(source).strip().lower()
    if raw in {"rtsp", "hikvision", "camera"} or raw.startswith(("rtsp://", "rtsps://")):
        return profile
    return None


def merge_source_metadata(record: dict[str, Any], health: dict[str, Any], *, source_type: str, camera_id: str | None, profile: str | None, frame_scale: float | None = None) -> dict[str, Any]:
    """Add non-secret source fields without changing posture fields."""

    merged = dict(record)
    merged.update({
        "source_type": str(source_type),
        "camera_id": camera_id,
        "source_profile": profile,
        "source_status": source_status_text(health),
        "source_resolution": health.get("resolution"),
        "source_reported_fps": health.get("reported_fps"),
        "source_effective_fps": health.get("effective_received_fps"),
        "source_last_frame_age_seconds": health.get("last_frame_age_seconds"),
        "source_reconnect_count": health.get("reconnect_count", 0),
    })
    if frame_scale is not None:
        merged["inference_frame_scale"] = float(frame_scale)
    return merged


def summarize_timing(
    *,
    elapsed_seconds: float,
    display_count: int,
    inference_count: int,
    capture_age_ms: Sequence[float],
    inference_time_ms: Sequence[float],
    render_time_ms: Sequence[float],
    source_fps: float,
    frames_captured: int,
    frames_dropped: int,
    stale_events: int,
) -> dict[str, Any]:
    """Produce auditable latency metrics from observed values only."""

    elapsed = max(float(elapsed_seconds), 1e-9)

    def stats(values: Sequence[float]) -> dict[str, float | None]:
        clean = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=float)
        if clean.size == 0:
            return {"median_ms": None, "p95_ms": None, "mean_ms": None, "max_ms": None}
        return {
            "median_ms": float(np.median(clean)),
            "p95_ms": float(np.percentile(clean, 95)),
            "mean_ms": float(np.mean(clean)),
            "max_ms": float(np.max(clean)),
        }

    age = stats(capture_age_ms)
    return {
        "source_fps": float(source_fps),
        "display_fps": float(display_count / elapsed),
        "inference_fps": float(inference_count / elapsed),
        "frames_captured": int(frames_captured),
        "frames_inferred": int(inference_count),
        "frames_intentionally_dropped": int(frames_dropped),
        "stale_events": int(stale_events),
        "capture_age_ms": age,
        "frame_age_ms": age,
        "inference_time_ms": stats(inference_time_ms),
        "render_time_ms": stats(render_time_ms),
        "measurement_note": "Frame age is the observed wait from latest-frame publication to AI-loop read; exact camera capture timestamp is not exposed by OpenCV.",
    }


def monotonic_pose_timestamp_seconds(candidate: float, previous: float | None, *, minimum_step_seconds: float = 0.001) -> float:
    """Keep MediaPipe video timestamps strictly increasing when RTSP bursts arrive."""

    value = max(0.0, float(candidate))
    if previous is None:
        return value
    return max(value, float(previous) + max(1e-6, float(minimum_step_seconds)))


def _lab_text(output: np.ndarray, text: str, x: int, y: int, *, scale: float = 0.55, color: tuple[int, int, int] = (255, 255, 255), thickness: int = 1) -> None:
    cv2.putText(output, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(2, thickness + 2), cv2.LINE_AA)
    cv2.putText(output, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _lab_panel(output: np.ndarray, x: int, y: int, width: int, height: int, color: tuple[int, int, int] = (25, 35, 45), alpha: float = 0.72) -> None:
    overlay = output.copy()
    cv2.rectangle(overlay, (max(0, x), max(0, y)), (min(output.shape[1] - 1, x + width), min(output.shape[0] - 1, y + height)), color, -1)
    output[:] = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0)


def _draw_core_landmarks(output: np.ndarray, image_pose, *, mirror: bool, detailed: bool) -> None:
    if image_pose is None:
        return
    height, width = output.shape[:2]
    coords: dict[int, tuple[int, int]] = {}
    indices = range(len(image_pose.points)) if detailed else [LANDMARK_INDEX[name] for name in CORE_LANDMARKS]
    for index in indices:
        point = image_pose.points[index]
        if point is None or not point.present:
            continue
        normalized_x = 1.0 - float(point.x) if mirror else float(point.x)
        coords[index] = (int(np.clip(normalized_x, 0.0, 1.0) * max(width - 1, 1)), int(np.clip(float(point.y), 0.0, 1.0) * max(height - 1, 1)))
        color = (0, 220, 80) if (point.visibility is not None and float(point.visibility) >= 0.5) else (0, 120, 255)
        cv2.circle(output, coords[index], 5 if not detailed else 3, color, -1)
    connections = SKELETON_CONNECTIONS if detailed else (
        (LANDMARK_INDEX["left_shoulder"], LANDMARK_INDEX["right_shoulder"]),
        (LANDMARK_INDEX["left_shoulder"], LANDMARK_INDEX["left_hip"]),
        (LANDMARK_INDEX["right_shoulder"], LANDMARK_INDEX["right_hip"]),
        (LANDMARK_INDEX["left_hip"], LANDMARK_INDEX["right_hip"]),
    )
    for left, right in connections:
        if left in coords and right in coords:
            cv2.line(output, coords[left], coords[right], (255, 190, 0), 2 if not detailed else 1)


def draw_ai_lab_overlay(
    frame: np.ndarray,
    record: dict[str, Any],
    image_pose=None,
    *,
    mirror: bool = False,
    calibration: bool = False,
    debug: bool = False,
    detailed_landmarks: bool = False,
) -> np.ndarray:
    """Draw a readable Clean View; D and L expose research diagnostics."""

    output = frame.copy()
    height, width = output.shape[:2]
    _draw_core_landmarks(output, image_pose, mirror=mirror, detailed=detailed_landmarks)
    pose_detected = bool(record.get("pose_detected"))
    core_valid = bool(record.get("core_landmark_accepted"))
    pose_status = "GOOD" if core_valid else ("PARTIAL" if pose_detected else "LOST")
    raw_label = str(record.get("ai_label") or "UNCERTAIN")
    confidence = record.get("ai_confidence")
    low_confidence = str(record.get("status", "")) == "AI_LOW_CONFIDENCE"
    display_label = "UNCERTAIN" if low_confidence or confidence is None else raw_label
    confidence_text = "--" if confidence is None else f"{100.0 * float(confidence):.1f}%"
    ai_color = (35, 100, 40) if display_label == "TUP" else ((35, 55, 120) if display_label == "TLB" else (70, 60, 35))
    ai_width = min(430, max(360, width // 3))
    _lab_panel(output, 20, 20, ai_width, 175, ai_color, 0.82)
    _lab_text(output, "AI POSTURE", 40, 60, scale=0.75, thickness=2)
    _lab_text(output, display_label, 40, 122, scale=1.35, thickness=3)
    _lab_text(output, confidence_text, 40, 170, scale=0.86, thickness=2)

    compact_x = min(width - 255, ai_width + 55)
    _lab_panel(output, compact_x, 20, 225, 92, (25, 45, 55), 0.82)
    _lab_text(output, f"POSE: {pose_status}", compact_x + 18, 58, scale=0.64, color=(80, 240, 100) if pose_status == "GOOD" else (0, 190, 255), thickness=2)
    _lab_text(output, f"CORE: {'VALID' if core_valid else 'CHECK'}", compact_x + 18, 92, scale=0.48)

    _lab_panel(output, compact_x, 125, 225, 135, (25, 45, 55), 0.82)
    display_fps = float(record.get("display_fps", 0.0) or 0.0)
    inference_fps = float(record.get("inference_fps", 0.0) or 0.0)
    frame_age = record.get("frame_age_ms")
    age_text = "--" if frame_age is None else f"{float(frame_age):.0f} ms"
    _lab_text(output, f"FPS     {display_fps:.1f}", compact_x + 18, 158, scale=0.48)
    _lab_text(output, f"AI FPS  {inference_fps:.1f}", compact_x + 18, 188, scale=0.48)
    _lab_text(output, f"FRAME AGE  {age_text}", compact_x + 18, 218, scale=0.43)
    _lab_text(output, f"CALIBRATION: {record.get('calibration_status', 'WAITING')}", compact_x + 18, 245, scale=0.37)

    geometry_y = max(270, height - 128)
    _lab_panel(output, 20, geometry_y, 305, 100, (35, 40, 65), 0.78)
    _lab_text(output, "GEOMETRY / EXPERIMENTAL", 38, geometry_y + 28, scale=0.48, thickness=1)
    _lab_text(output, f"Forward  {'YES' if record.get('forward_indicator') else 'NO'}", 38, geometry_y + 55, scale=0.44)
    _lab_text(output, f"Left     {'YES' if record.get('left_indicator') else 'NO'}", 155, geometry_y + 55, scale=0.44)
    _lab_text(output, f"Right    {'YES' if record.get('right_indicator') else 'NO'}", 38, geometry_y + 82, scale=0.44)
    _lab_text(output, f"TEMPORAL: {record.get('risk_state', 'POSE_LOST')}", 155, geometry_y + 82, scale=0.39)

    source_text = f"SOURCE {record.get('source_type', 'UNKNOWN')} | {record.get('source_status', 'SOURCE_UNKNOWN')} | D Debug  L Landmarks  Q Quit"
    _lab_text(output, source_text, 20, height - 12, scale=0.39)
    if debug:
        _lab_panel(output, max(20, width - 355), max(270, height - 250), 335, 225, (15, 25, 35), 0.86)
        probabilities = record.get("ai_probabilities") or {}
        resolution = record.get("source_resolution") or {}
        debug_lines = [
            "DEBUG",
            f"RAW AI: {raw_label}  CONF: {confidence_text}",
            f"TUP p={100.0 * float(probabilities.get('TUP', 0.0)):.1f}%  TLB p={100.0 * float(probabilities.get('TLB', 0.0)):.1f}%",
            f"INFERENCE: {float(record.get('inference_time_ms', 0.0) or 0.0):.1f} ms",
            f"RENDER: {float(record.get('render_time_ms', 0.0) or 0.0):.1f} ms",
            f"SOURCE FPS: {float(record.get('source_effective_fps', 0.0) or 0.0):.1f}",
            f"RESOLUTION: {resolution.get('width', 0)}x{resolution.get('height', 0)}",
            f"DROPPED OLD: {int(record.get('frames_intentionally_dropped', 0) or 0)}",
            f"RECONNECTS: {int(record.get('source_reconnect_count', 0) or 0)}",
        ]
        for index, line in enumerate(debug_lines):
            _lab_text(output, line, max(35, width - 340), max(300, height - 220) + index * 22, scale=0.38)
    return output


def run_rtsp_ai_lab_session(
    source: str | int = "rtsp",
    *,
    model_dir: str | Path = "outputs/models/rgb_native_v4",
    pose_model: str | Path = "metadata/pose_landmarker_heavy.task",
    config_path: str | Path = "configs/posture_engine.yaml",
    rtsp_config_path: str | Path = "configs/rtsp_profiles.yaml",
    profile: str = "HIKVISION_MAIN",
    camera_id: str | None = None,
    preset: str = "RTSP_LOW_LATENCY",
    max_people: int = 1,
    max_frames: int | None = None,
    max_seconds: float | None = None,
    max_reconnects: int = 5,
    output_root: str | Path = "outputs/rtsp_ai_lab",
    display: bool = True,
    mirror: bool = False,
    debug: bool = False,
    detailed_landmarks: bool = False,
    inference_width: int | None = None,
    inference_height: int | None = None,
    display_width: int = 1280,
    display_height: int = 720,
) -> dict[str, Any]:
    """Run the interactive lab using async latest-frame semantics for RTSP."""

    if int(max_people) != 1:
        raise ValueError("RTSP AI Lab robustness mode is intentionally single-person; use the multiperson entry point separately")
    if max_frames is not None and int(max_frames) < 1:
        raise ValueError("max_frames must be positive")
    if max_seconds is not None and float(max_seconds) <= 0:
        raise ValueError("max_seconds must be positive")
    if int(display_width) < 320 or int(display_height) < 240:
        raise ValueError("display dimensions are too small")
    runtime = resolve_rtsp_preset(preset, config_path=rtsp_config_path)
    if inference_width is not None:
        if int(inference_width) < 160:
            raise ValueError("inference_width must be at least 160 pixels")
        runtime["inference_max_width"] = int(inference_width)
    if inference_height is not None:
        if int(inference_height) < 120:
            raise ValueError("inference_height must be at least 120 pixels")
        runtime["inference_max_height"] = int(inference_height)
    config = PostureEngineConfig.from_yaml(config_path)
    video = VideoSource(
        source,
        camera_id=camera_id,
        profile=profile_for_source(source, profile),
        config_path=rtsp_config_path,
        reconnect=True,
        max_reconnects=max_reconnects,
        stale_after_seconds=runtime["stale_after_seconds"],
        low_latency=bool(runtime.get("low_latency_buffer_request", False)),
    )
    latest = LatestFrameInput(video, asynchronous=video.source_type == "RTSP")
    if not latest.open():
        raise RuntimeError(f"Source could not be opened: {video.spec.source_display}")

    runner = RGBNativeV4Runner(model_dir)
    landmarker = create_pose_landmarker(pose_model, running_mode="video")
    processor = LiveFrameProcessor(runner, landmarker, config, calibration=True)
    session = LiveSessionAccumulator(frame_interval_seconds=config.frame_interval_seconds)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []
    source_events: list[dict[str, Any]] = []
    capture_ages: list[float] = []
    inference_times: list[float] = []
    render_times: list[float] = []
    frame_index = 0
    processed_frames = 0
    displayed_frames = 0
    stale_events = 0
    last_record: dict[str, Any] | None = None
    last_image_pose = None
    started = time.perf_counter()
    last_timestamp = 0.0
    source_time_origin: float | None = None
    closed_reason = "LIMIT_OR_USER"
    last_pose_timestamp: float | None = None
    try:
        if display:
            cv2.namedWindow("Office Posture - RTSP AI Lab", cv2.WINDOW_NORMAL)
        while max_frames is None or frame_index < int(max_frames):
            if max_seconds is not None and time.perf_counter() - started >= float(max_seconds):
                closed_reason = "MAX_SECONDS"
                break
            result = latest.read()
            if not result.ok or result.frame is None:
                health = result.health
                if result.reason == "no_new_frame" or health.get("source_status") == "SOURCE_STALE":
                    stale_events += 1
                event = {
                    "timestamp": float(result.timestamp_seconds),
                    "source_status": result.state,
                    "source_reason": result.reason,
                    "source_health": health,
                    "pose_recorded": False,
                }
                source_events.append(event)
                if result.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                    closed_reason = result.reason or result.state
                    break
                continue

            frame = result.frame
            health = result.health
            if result.frame_age_ms is not None:
                capture_ages.append(float(result.frame_age_ms))
            source_timestamp = float(result.timestamp_seconds)
            if source_time_origin is None:
                source_time_origin = source_timestamp
            session_source_timestamp = max(0.0, source_timestamp - source_time_origin)
            last_timestamp = max(last_timestamp, session_source_timestamp)
            inference_started = time.perf_counter()
            inference_frame, scale = resize_for_inference(
                frame,
                max_width=runtime.get("inference_max_width"),
                max_height=runtime.get("inference_max_height"),
            )
            fps = float(health.get("effective_received_fps") or health.get("reported_fps") or 0.0)
            pose_timestamp = monotonic_pose_timestamp_seconds(session_source_timestamp, last_pose_timestamp)
            last_pose_timestamp = pose_timestamp
            record, image_pose = processor.process(
                inference_frame,
                frame_index=frame_index,
                timestamp_seconds=pose_timestamp,
                fps=fps,
            )
            inference_time_ms = (time.perf_counter() - inference_started) * 1000.0
            inference_times.append(inference_time_ms)
            record = merge_source_metadata(record, health, source_type=video.source_type, camera_id=video.camera_id, profile=video.spec.profile, frame_scale=scale)
            record.update({
                "inference_skipped": False,
                "ai_lab_status": "LIVE_SINGLE_PERSON",
                "frame_age_ms": result.frame_age_ms,
                "capture_age_ms": result.frame_age_ms,
                "source_timestamp_seconds": source_timestamp,
                "mediapipe_timestamp_seconds": float(pose_timestamp),
                "inference_time_ms": inference_time_ms,
                "frames_intentionally_dropped": latest.dropped_frames,
                "calibration_status": "READY" if record.get("core_landmark_accepted") else "WAITING",
            })
            processed_frames += 1
            elapsed_now = max(time.perf_counter() - started, 1e-9)
            record["display_fps"] = displayed_frames / elapsed_now
            record["inference_fps"] = processed_frames / elapsed_now
            render_started = time.perf_counter()
            if display:
                display_frame, _display_scale = resize_for_display(
                    frame,
                    max_width=int(display_width),
                    max_height=int(display_height),
                )
                annotated = draw_ai_lab_overlay(
                    prepare_display_frame(display_frame, mirror=mirror),
                    record,
                    image_pose,
                    mirror=mirror,
                    debug=debug,
                    detailed_landmarks=detailed_landmarks,
                )
                cv2.imshow("Office Posture - RTSP AI Lab", annotated)
                displayed_frames += 1
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    closed_reason = "USER_EXIT"
                    break
                if key == ord("d"):
                    debug = not debug
                elif key == ord("l"):
                    detailed_landmarks = not detailed_landmarks
            render_time_ms = (time.perf_counter() - render_started) * 1000.0
            render_times.append(render_time_ms)
            record["render_time_ms"] = render_time_ms
            record["display_fps"] = displayed_frames / max(time.perf_counter() - started, 1e-9)
            session.add(record)
            last_record = deepcopy(record)
            last_image_pose = image_pose
            records.append(record)
            frame_index += 1
    finally:
        landmarker.close()
        final_health = latest.health()
        dropped_frames = latest.dropped_frames
        latest.close()
        if display:
            cv2.destroyAllWindows()

    elapsed = max(0.0, time.perf_counter() - started)
    timing = summarize_timing(
        elapsed_seconds=elapsed,
        display_count=displayed_frames,
        inference_count=processed_frames,
        capture_age_ms=capture_ages,
        inference_time_ms=inference_times,
        render_time_ms=render_times,
        source_fps=float(final_health.get("effective_received_fps") or final_health.get("reported_fps") or 0.0),
        frames_captured=int(final_health.get("frames_received", 0) or 0),
        frames_dropped=dropped_frames,
        stale_events=stale_events,
    )
    summary = session.summary(total_duration_seconds=max(last_timestamp, elapsed))
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
        "stream_preset": runtime["name"],
        "inference_resolution_limit": {"width": runtime.get("inference_max_width"), "height": runtime.get("inference_max_height")},
        "displayed_frames": displayed_frames,
        "processed_frames": processed_frames,
        "inference_skipped_frames": 0,
        "frames_intentionally_dropped": dropped_frames,
        "max_people": 1,
        "raw_video_recorded": False,
        "recording_enabled": False,
        "debug_mode_at_close": bool(debug),
        "detailed_landmarks_at_close": bool(detailed_landmarks),
        "closed_reason": closed_reason,
        "timing_metrics": timing,
        "software_status": "RTSP_AI_LAB_READY_FOR_LIVE_VIEW",
        "hardware_status": "SOURCE_SESSION_COMPLETED" if processed_frames else "REAL_STREAM_NOT_TESTED",
        "interpretation": "Engineering telemetry only. V4 AI posture and geometry indicators are separate; no medical or generalization claim.",
    })
    paths = write_live_outputs(records + source_events, summary, output_root=output_root, session_id=f"rtsp_ai_lab_{session_id}")
    return {"summary": summary, "paths": paths, "records": records + source_events, "source_health_events": source_events}


__all__ = [
    "draw_ai_lab_overlay",
    "merge_source_metadata",
    "profile_for_source",
    "run_rtsp_ai_lab_session",
    "source_status_text",
    "summarize_timing",
    "monotonic_pose_timestamp_seconds",
]
