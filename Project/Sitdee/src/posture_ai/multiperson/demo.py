"""Explicitly launched multi-person/seat-aware runtime over one VideoSource."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig
from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput
from src.posture_ai.inference.stream_runtime import resize_for_display, resize_for_inference
from src.posture_ai.inference.video_source import SOURCE_CLOSED, SOURCE_FAILED, SOURCE_RTSP, VideoSource

from .mediapipe_tasks import MultiPersonMediaPipeBackend
from .overlay import draw_multiperson_overlay
from .runtime import MultiPersonRuntime, MultiPersonRuntimeConfig, resolve_performance_preset, should_process_frame
from .seating import SeatAwareRuntime, SeatAwareRuntimeConfig, draw_seat_layout, load_seat_layout
from .telemetry import MultiPersonTelemetry
from .tracking import core_quality_summary


def parse_source(value: str) -> int | str:
    """Interpret numeric values as camera indices and other values as source names/paths."""

    try:
        return int(value)
    except ValueError:
        return value


def run_multiperson_demo(
    source: str | int,
    *,
    max_people: int = 3,
    performance_preset: str = "RTSP_CCTV",
    process_every: int | None = None,
    model_dir: str | Path = "outputs/models/rgb_native_v4",
    pose_model: str | Path = "metadata/pose_landmarker_heavy.task",
    config_path: str | Path = "configs/posture_engine.yaml",
    rtsp_config_path: str | Path = "configs/rtsp_profiles.yaml",
    profile: str | None = None,
    camera_id: str | None = None,
    telemetry_path: str | Path | None = None,
    max_frames: int | None = None,
    max_seconds: float | None = None,
    max_reconnects: int = 5,
    tracking_mode: str = "generic",
    seat_layout_path: str | Path | None = None,
    display: bool = True,
    display_width: int = 1280,
    display_height: int = 720,
    debug: bool = False,
    latest_frame_only: bool | None = None,
) -> dict[str, Any]:
    """Run multi-person V4 over USB, file, or RTSP through the Desktop VideoSource."""

    mode = str(tracking_mode).lower()
    if mode not in {"generic", "seat"}:
        raise ValueError("tracking_mode must be 'generic' or 'seat'")
    if max_people < 1 or max_people > 4:
        raise ValueError("max_people must be between 1 and 4")
    if max_frames is not None and int(max_frames) < 1:
        raise ValueError("max_frames must be positive")
    if max_seconds is not None and float(max_seconds) <= 0:
        raise ValueError("max_seconds must be positive")
    if int(max_reconnects) < 0:
        raise ValueError("max_reconnects must not be negative")
    if int(display_width) < 320 or int(display_height) < 240:
        raise ValueError("display dimensions are too small")
    if mode == "seat" and seat_layout_path is None:
        raise ValueError("seat tracking requires an explicit real seat-layout config")
    source_value = parse_source(str(source)) if isinstance(source, str) else int(source)
    performance = resolve_performance_preset(performance_preset, process_every)
    video = VideoSource(
        source_value,
        camera_id=camera_id,
        profile=profile,
        config_path=rtsp_config_path,
        reconnect=True,
        max_reconnects=max_reconnects,
        stale_after_seconds=1.0 if performance["name"] == "RTSP_CCTV" else 2.0,
        low_latency=True,
    )
    runner = RGBNativeV4Runner(model_dir)
    engine_config = PostureEngineConfig.from_yaml(config_path)
    if mode == "seat":
        layout_camera_id, zones = load_seat_layout(seat_layout_path)
        runtime_camera_id = camera_id or layout_camera_id
        runtime = SeatAwareRuntime(
            runner,
            zones,
            runtime_config=SeatAwareRuntimeConfig(
                max_people=max_people,
                camera_id=runtime_camera_id,
            ),
            posture_config=engine_config,
        )
    else:
        runtime_camera_id = camera_id or video.camera_id or "CAM01"
        zones = []
        runtime = MultiPersonRuntime(
            runner,
            runtime_config=MultiPersonRuntimeConfig(
                max_people=max_people,
                visibility_threshold=engine_config.minimum_pose_visibility,
                camera_id=runtime_camera_id,
            ),
            posture_config=engine_config,
        )
    backend = MultiPersonMediaPipeBackend(
        pose_model,
        max_people=max_people,
        running_mode="video",
        visibility_threshold=engine_config.minimum_pose_visibility,
    )
    use_latest_frame = video.source_type == SOURCE_RTSP if latest_frame_only is None else bool(latest_frame_only)
    source_input = LatestFrameInput(video, asynchronous=use_latest_frame)
    if not source_input.open():
        backend.close()
        raise RuntimeError(f"Could not open explicit source: {video.spec.source_display}")
    telemetry = MultiPersonTelemetry(max_events=2000)

    frame_index = 0
    processed_frames = 0
    skipped_frames = 0
    started = time.perf_counter()
    last_timestamp = 0.0
    last_pose_timestamp: float | None = None
    last_result = None
    last_frame = None
    health_before_close: dict[str, Any] | None = None
    frame_ages: list[float] = []
    inference_times: list[float] = []
    detected_people_counts: list[int] = []
    valid_core_counts: list[int] = []
    rejected_pose_counts: list[int] = []
    quality_values: list[float] = []
    core_visibility_values: dict[str, list[float]] = {name: [] for name in ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")}
    head_visible_count = 0
    shoulders_visible_count = 0
    hips_visible_count = 0
    source_events: list[dict[str, Any]] = []
    stale_events = 0
    lifecycle_event_counts: Counter[str] = Counter()
    activity_event_counts: Counter[str] = Counter()
    activity_state_counts: Counter[str] = Counter()
    tracking_state_counts: Counter[str] = Counter()
    active_track_counts: list[int] = []
    person_ids_seen: set[str] = set()
    try:
        while (max_frames is None or frame_index < max_frames) and (max_seconds is None or time.perf_counter() - started < float(max_seconds)):
            read_result = source_input.read()
            if not read_result.ok or read_result.frame is None:
                if read_result.reason == "no_new_frame" or read_result.state == "SOURCE_STALE":
                    stale_events += 1
                if len(source_events) < 200:
                    source_events.append({
                        "timestamp_seconds": float(read_result.timestamp_seconds),
                        "source_status": read_result.state,
                        "source_reason": read_result.reason,
                    })
                if read_result.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                    break
                continue

            frame = read_result.frame
            last_frame = frame
            health = read_result.health
            timestamp_seconds = float(read_result.timestamp_seconds)
            last_timestamp = max(last_timestamp, timestamp_seconds)
            if read_result.frame_age_ms is not None:
                frame_ages.append(float(read_result.frame_age_ms))
            if should_process_frame(frame_index, performance["process_every"], has_previous_result=last_result is not None):
                inference_started = time.perf_counter()
                inference_frame, _scale = resize_for_inference(
                    frame,
                    max_width=performance["capture_width"],
                    max_height=performance["capture_height"],
                )
                if last_pose_timestamp is None:
                    pose_timestamp = max(0.0, timestamp_seconds)
                else:
                    pose_timestamp = max(timestamp_seconds, last_pose_timestamp + 0.001)
                last_pose_timestamp = pose_timestamp
                detections = backend.detect(
                    inference_frame,
                    timestamp_ms=int(round(pose_timestamp * 1000.0)),
                )
                detected_people_counts.append(len(detections))
                valid_core_counts.append(sum(1 for detection in detections if detection.core_valid))
                rejected_pose_counts.append(sum(1 for detection in detections if not detection.core_valid))
                for detection in detections:
                    quality = core_quality_summary(detection, threshold=engine_config.minimum_pose_visibility)
                    quality_values.append(float(quality["pose_quality_score"]))
                    head_visible_count += int(bool(quality["HEAD_VISIBLE"]))
                    shoulders_visible_count += int(bool(quality["SHOULDERS_VISIBLE"]))
                    hips_visible_count += int(bool(quality["HIPS_VISIBLE"]))
                    for name, value in quality["core_visibility"].items():
                        if value is not None:
                            core_visibility_values[name].append(float(value))
                last_result = runtime.process_frame(
                    detections,
                    pose_timestamp,
                    camera_id=runtime_camera_id,
                )
                active_track_counts.append(len(last_result.subjects))
                for event in getattr(last_result, "events", ()):
                    event_name = str(event.get("event", "UNKNOWN"))
                    if event_name.startswith("PERSON_"):
                        lifecycle_event_counts[event_name] += 1
                    else:
                        activity_event_counts[event_name] += 1
                for subject in last_result.subjects:
                    person_ids_seen.add(subject.subject_id)
                    activity_state_counts[subject.activity_state] += 1
                    tracking_state_counts[subject.tracking_state] += 1
                telemetry.add_frame(
                    last_result,
                    frame_index=frame_index,
                    inference_skipped=False,
                    source_metadata=health,
                )
                processed_frames += 1
                inference_times.append((time.perf_counter() - inference_started) * 1000.0)
                skipped = False
            else:
                if last_result is None:
                    frame_index += 1
                    continue
                skipped_result = replace(last_result, timestamp_seconds=timestamp_seconds, events=())
                telemetry.add_frame(
                    skipped_result,
                    frame_index=frame_index,
                    inference_skipped=True,
                    source_metadata=health,
                )
                skipped_frames += 1
                skipped = True

            if display:
                display_frame, _display_scale = resize_for_display(
                    frame,
                    max_width=int(display_width),
                    max_height=int(display_height),
                )
                if mode == "seat":
                    annotated = draw_seat_layout(display_frame, zones, result=last_result)
                    status = f"SEAT-AWARE V2 | {performance['name']} | seats={len(last_result.seats)}"
                else:
                    annotated = draw_multiperson_overlay(display_frame, last_result, debug=debug)
                    status = f"PERSON-CENTRIC | {performance['name']} | people={len(last_result.subjects)}"
                if skipped:
                    status += " | INFERENCE SKIPPED"
                cv2.putText(annotated, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, "Q = quit", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.imshow("Multi-Person Posture Prototype V2", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
            frame_index += 1
    finally:
        backend.close()
        health_before_close = source_input.health()
        dropped_frames = source_input.dropped_frames
        source_input.close()
        if display:
            cv2.destroyAllWindows()

    if telemetry_path is not None:
        telemetry.write_jsonl(telemetry_path)
    elapsed = max(0.0, time.perf_counter() - started)
    final_health = health_before_close or video.health()
    source_fps = float(final_health.get("effective_received_fps") or final_health.get("reported_fps") or 0.0)
    inference_fps = float(processed_frames / max(elapsed, 1e-9))
    mean_detected = float(sum(detected_people_counts) / len(detected_people_counts)) if detected_people_counts else 0.0
    max_detected = int(max(detected_people_counts)) if detected_people_counts else 0
    mean_valid_core = float(sum(valid_core_counts) / len(valid_core_counts)) if valid_core_counts else 0.0
    max_valid_core = int(max(valid_core_counts)) if valid_core_counts else 0
    mean_rejected = float(sum(rejected_pose_counts) / len(rejected_pose_counts)) if rejected_pose_counts else 0.0
    pose_detection_rate = float(sum(count > 0 for count in detected_people_counts) / len(detected_people_counts)) if detected_people_counts else 0.0
    core_rate = float(sum(valid for valid in valid_core_counts) / max(sum(detected_people_counts), 1)) if detected_people_counts else 0.0
    total_detections = max(sum(detected_people_counts), 1)
    quality_stats = {
        "mean_pose_quality_score": None if not quality_values else float(np.mean(quality_values)),
        "head_visible_rate": float(head_visible_count / total_detections) if detected_people_counts else 0.0,
        "shoulders_visible_rate": float(shoulders_visible_count / total_detections) if detected_people_counts else 0.0,
        "hips_visible_rate": float(hips_visible_count / total_detections) if detected_people_counts else 0.0,
        "core_ok_rate": core_rate,
        "core_partial_rate": float(sum(rejected_pose_counts) / total_detections) if detected_people_counts else 0.0,
        "landmark_mean_visibility": {
            name: (None if not values else float(np.mean(values)))
            for name, values in core_visibility_values.items()
        },
    }
    return {
        "status": "SEAT_AWARE_SOFTWARE_READY" if mode == "seat" else "MULTI_PERSON_SOFTWARE_READY",
        "hardware_status": "REAL_RTSP_SHORT_SOURCE_RUN" if video.source_type == SOURCE_RTSP else "HARDWARE_NOT_VALIDATED",
        "source": video.spec.source_display,
        "source_type": video.source_type,
        "camera_id": runtime_camera_id,
        "source_status": final_health.get("source_status"),
        "source_health": final_health,
        "max_people": int(max_people),
        "tracking_mode": mode,
        "seat_layout_path": None if mode == "generic" else str(seat_layout_path),
        "performance": performance,
        "captured_frames": int(frame_index),
        "processed_frames": int(processed_frames),
        "inference_skipped_frames": int(skipped_frames),
        "telemetry_events": len(telemetry.events),
        "telemetry_path": None if telemetry_path is None else str(telemetry_path),
        "elapsed_seconds": elapsed,
        "last_frame_available": last_frame is not None,
        "latest_frame_only": bool(use_latest_frame),
        "source_fps": source_fps,
        "inference_fps": inference_fps,
        "frame_age_ms": {
            "median_ms": None if not frame_ages else float(np.median(frame_ages)),
            "p95_ms": None if not frame_ages else float(np.percentile(frame_ages, 95)),
        },
        "inference_time_ms": {
            "median_ms": None if not inference_times else float(np.median(inference_times)),
            "p95_ms": None if not inference_times else float(np.percentile(inference_times, 95)),
        },
        "frames_captured": int(final_health.get("frames_received", 0) or 0),
        "frames_inferred": int(processed_frames),
        "frames_intentionally_dropped": int(dropped_frames),
        "stale_events": int(stale_events),
        "source_health_events": int(len(source_events)),
        "poses_detected_rate": pose_detection_rate,
        "core_valid_rate": core_rate,
        "core_quality": quality_stats,
        "detected_people": {
            "mean_per_inferred_frame": mean_detected,
            "max": max_detected,
            "per_inferred_frame": detected_people_counts,
        },
        "valid_core_poses": {
            "mean_per_inferred_frame": mean_valid_core,
            "max": max_valid_core,
            "per_inferred_frame": valid_core_counts,
        },
        "rejected_poses": {
            "mean_per_inferred_frame": mean_rejected,
            "per_inferred_frame": rejected_pose_counts,
        },
        "tracking_events": dict(lifecycle_event_counts),
        "activity_events": dict(activity_event_counts),
        "activity_state_distribution": dict(activity_state_counts),
        "tracking_state_distribution": dict(tracking_state_counts),
        "active_track_count": {
            "mean": None if not active_track_counts else float(np.mean(active_track_counts)),
            "max": 0 if not active_track_counts else int(max(active_track_counts)),
            "min": 0 if not active_track_counts else int(min(active_track_counts)),
        },
        "anonymous_person_ids_seen": sorted(person_ids_seen),
        "id_fragmentation_proxy": {
            "appeared_events": int(lifecycle_event_counts.get("PERSON_APPEARED", 0)),
            "reacquired_events": int(lifecycle_event_counts.get("PERSON_REACQUIRED", 0)),
            "temporary_lost_events": int(lifecycle_event_counts.get("PERSON_TEMPORARILY_LOST", 0)),
            "exited_events": int(lifecycle_event_counts.get("PERSON_EXITED", 0)),
            "id_switch_accuracy": "NOT_MEASURED_NO_MANUAL_PERSON_GROUND_TRUTH",
        },
        "source_events": source_events,
        "interpretation": "Engineering telemetry only. Person detection, core quality, posture classification and track stability are separate metrics; no medical or generalization claim.",
    }
