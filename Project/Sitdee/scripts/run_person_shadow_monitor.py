"""Long-running, no-video shadow observation over the frozen person runtime.

This runner records only anonymous engineering telemetry.  It intentionally
does not write frames, video, labels, model updates, or credentials.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig  # noqa: E402
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput  # noqa: E402
from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner  # noqa: E402
from src.posture_ai.inference.stream_runtime import resize_for_inference  # noqa: E402
from src.posture_ai.inference.video_source import SOURCE_CLOSED, SOURCE_FAILED, VideoSource  # noqa: E402
from src.posture_ai.monitoring.shadow import (  # noqa: E402
    RotatingJsonlWriter,
    ShadowReviewQueue,
    build_frame_observation,
    build_subject_observation,
    safe_source_metadata,
)
from src.posture_ai.multiperson.mediapipe_tasks import MultiPersonMediaPipeBackend  # noqa: E402
from src.posture_ai.multiperson.runtime import MultiPersonRuntime, MultiPersonRuntimeConfig, resolve_performance_preset, should_process_frame  # noqa: E402


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _make_session_directory(root: Path, requested_id: str | None) -> tuple[str, Path]:
    session_id = requested_id or f"shadow_{datetime.now().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    directory = root / session_id
    if directory.exists():
        raise FileExistsError(f"Shadow session directory already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    return session_id, directory


def run_shadow_monitor(args: argparse.Namespace) -> dict:
    if args.max_people < 1 or args.max_people > 4:
        raise ValueError("max_people must be between 1 and 4")
    if args.duration_minutes <= 0:
        raise ValueError("duration_minutes must be positive")
    if args.max_log_mib < 1 or args.max_log_files < 1:
        raise ValueError("log limits must be positive")
    session_id, session_dir = _make_session_directory(Path(args.output_root), args.session_id)
    max_bytes = int(args.max_log_mib * 1024 * 1024)
    observations_writer = RotatingJsonlWriter(session_dir, "observations", max_bytes=max_bytes, max_files=args.max_log_files, flush_seconds=args.flush_seconds)
    frames_writer = RotatingJsonlWriter(session_dir, "frames", max_bytes=max_bytes, max_files=args.max_log_files, flush_seconds=args.flush_seconds)
    review_writer = RotatingJsonlWriter(session_dir, "review_queue", max_bytes=max(1024 * 1024, max_bytes // 8), max_files=args.max_log_files, flush_seconds=args.flush_seconds)
    source_event_writer = RotatingJsonlWriter(session_dir, "source_events", max_bytes=max(1024 * 1024, max_bytes // 16), max_files=args.max_log_files, flush_seconds=args.flush_seconds)
    engine_config = PostureEngineConfig.from_yaml(args.config)
    performance = resolve_performance_preset(args.preset, args.process_every)
    source = VideoSource(
        _parse_source(args.source), camera_id=args.camera_id, profile=args.profile,
        config_path=args.rtsp_config, reconnect=True, max_reconnects=args.max_reconnects,
        stale_after_seconds=1.0 if performance["name"] == "RTSP_CCTV" else 2.0, low_latency=True,
    )
    runner = RGBNativeV4Runner(args.model_dir)
    runtime_camera_id = args.camera_id or source.camera_id or "CAM01"
    runtime = MultiPersonRuntime(
        runner,
        runtime_config=MultiPersonRuntimeConfig(max_people=args.max_people, visibility_threshold=engine_config.minimum_pose_visibility, camera_id=runtime_camera_id),
        posture_config=engine_config,
    )
    backend = MultiPersonMediaPipeBackend(args.pose_model, max_people=args.max_people, running_mode="video", visibility_threshold=engine_config.minimum_pose_visibility)
    source_input = LatestFrameInput(source, asynchronous=source.source_type == "RTSP")
    review_queue = ShadowReviewQueue(unknown_review_seconds=args.unknown_review_seconds, cooldown_seconds=args.review_cooldown_seconds, near_boundary_margin=args.near_boundary_margin)
    manifest = {
        "schema_version": "person_shadow_manifest_v1", "session_id": session_id, "started_utc": _timestamp_utc(),
        "source_type": source.source_type, "camera_id": runtime_camera_id, "profile": args.profile,
        "max_people": args.max_people, "tracking_mode": "generic", "preset": performance,
        "raw_video_recorded": False, "snapshots_enabled": False, "automatic_labels_created": False,
        "v4_mode": "FROZEN_TUP_TLB_ONLY", "activity_method": "GEOMETRY_EXPERIMENTAL",
        "log_retention": "ROTATE_THEN_STOP_NO_AUTOMATIC_DELETION",
    }
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.perf_counter()
    deadline = started + args.duration_minutes * 60.0
    frame_index = 0
    processed = 0
    skipped = 0
    last_pose_timestamp: float | None = None
    last_result = None
    inference_starts: deque[float] = deque(maxlen=60)
    last_source_event_key: tuple[str, str] | None = None
    last_source_event_at = -999.0
    stopped_reason = "DURATION_COMPLETED"
    final_health: dict = {}
    latest_frame_dropped = 0
    max_detected_pose_count = 0
    capacity_drop_count = 0
    try:
        if not source_input.open():
            raise RuntimeError(f"Could not open explicit source: {source.spec.source_display}")
        while time.perf_counter() < deadline:
            read = source_input.read()
            now = time.perf_counter()
            if not read.ok or read.frame is None:
                health = safe_source_metadata(read.health)
                key = (str(read.state), str(read.reason))
                if key != last_source_event_key or now - last_source_event_at >= 2.0:
                    source_event_writer.write({
                        "schema_version": "person_shadow_v1", "record_type": "source_event", "session_id": session_id,
                        "timestamp_seconds": max(0.0, now - started), "timestamp_utc": _timestamp_utc(),
                        "source_state": read.state, "source_reason": read.reason, "source": health,
                        "raw_video_recorded": False,
                    }, now_seconds=now - started)
                    last_source_event_key, last_source_event_at = key, now
                if read.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                    stopped_reason = f"SOURCE_STOPPED_{read.state}"
                    break
                continue
            frame = read.frame
            timestamp = float(read.timestamp_seconds)
            health = safe_source_metadata(read.health)
            if not should_process_frame(frame_index, performance["process_every"], has_previous_result=last_result is not None):
                skipped += 1
                frame_index += 1
                continue
            inference_started = time.perf_counter()
            inference_frame, _ = resize_for_inference(frame, max_width=performance["capture_width"], max_height=performance["capture_height"])
            pose_timestamp = max(0.0, timestamp) if last_pose_timestamp is None else max(timestamp, last_pose_timestamp + 0.001)
            last_pose_timestamp = pose_timestamp
            detections = backend.detect(inference_frame, timestamp_ms=int(round(pose_timestamp * 1000.0)))
            last_result = runtime.process_frame(detections, pose_timestamp, camera_id=runtime_camera_id)
            detected_pose_count = runtime.tracker.last_detected_pose_count
            dropped_due_to_capacity = runtime.tracker.last_dropped_due_to_capacity
            max_detected_pose_count = max(max_detected_pose_count, detected_pose_count)
            capacity_drop_count += dropped_due_to_capacity
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            processed += 1
            inference_starts.append(time.perf_counter())
            fps = None if len(inference_starts) < 2 else (len(inference_starts) - 1) / max(1e-9, inference_starts[-1] - inference_starts[0])
            age_ms = read.frame_age_ms
            frame_record = build_frame_observation(
                session_id=session_id, timestamp_seconds=pose_timestamp, timestamp_utc=_timestamp_utc(), frame_index=frame_index,
                source_health=health, frame_age_ms=age_ms, inference_fps=fps, inference_time_ms=inference_ms,
                subject_count=len(last_result.subjects), events=last_result.events,
                detected_pose_count=detected_pose_count, dropped_due_to_capacity=dropped_due_to_capacity,
            )
            if not frames_writer.write(frame_record, now_seconds=time.perf_counter() - started):
                stopped_reason = "LOG_CAPACITY_REACHED"
                break
            observation_capacity_reached = False
            for subject in last_result.subjects:
                record = build_subject_observation(
                    subject, session_id=session_id, timestamp_seconds=pose_timestamp, timestamp_utc=_timestamp_utc(), frame_index=frame_index,
                    source_health=health, inference_fps=fps, frame_age_ms=age_ms, events=last_result.events,
                    visibility_threshold=engine_config.activity_gate.visibility_threshold,
                )
                if not observations_writer.write(record, now_seconds=time.perf_counter() - started):
                    stopped_reason = "LOG_CAPACITY_REACHED"
                    observation_capacity_reached = True
                    break
                for review in review_queue.inspect(record):
                    review_writer.write(review, now_seconds=time.perf_counter() - started)
            if observation_capacity_reached:
                break
            frame_index += 1
    except KeyboardInterrupt:
        stopped_reason = "INTERRUPTED_BY_USER"
    finally:
        try:
            final_health = safe_source_metadata(source_input.health())
            latest_frame_dropped = source_input.dropped_frames
        finally:
            source_input.close()
            backend.close()
            for writer in (observations_writer, frames_writer, review_writer, source_event_writer):
                writer.close()
    manifest.update({
        "ended_utc": _timestamp_utc(), "stopped_reason": stopped_reason, "elapsed_seconds": time.perf_counter() - started,
        "captured_frame_attempts": frame_index, "inference_frames": processed, "skipped_frames": skipped,
        "latest_frame_dropped": latest_frame_dropped, "final_source_health": final_health,
        "tracking_capacity": {"max_detected_pose_count": max_detected_pose_count, "dropped_due_to_capacity": capacity_drop_count},
        "writers": {"observations": observations_writer.summary(), "frames": frames_writer.summary(), "review_queue": review_writer.summary(), "source_events": source_event_writer.summary()},
    })
    manifest["log_capacity_reached"] = any(item["capacity_reached"] for item in manifest["writers"].values())
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "SHADOW_SESSION_COMPLETED", "session_dir": str(session_dir), "session_id": session_id, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description="Anonymous no-video shadow monitor for frozen V4")
    parser.add_argument("--source", default="rtsp", help="rtsp, a video path, or a USB index; no source is opened until run")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default="HIKVISION_MAIN")
    parser.add_argument("--camera-id", default="CAM02")
    parser.add_argument("--duration-minutes", type=float, default=60.0)
    parser.add_argument("--max-people", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--preset", default="RTSP_CCTV", choices=("DESKTOP", "BALANCED", "LOW_SPEC", "RTSP_CCTV"))
    parser.add_argument("--process-every", type=int, default=None)
    parser.add_argument("--max-reconnects", type=int, default=5)
    parser.add_argument("--model-dir", default="outputs/models/rgb_native_v4")
    parser.add_argument("--pose-model", default="metadata/pose_landmarker_heavy.task")
    parser.add_argument("--config", default="configs/posture_engine.yaml")
    parser.add_argument("--rtsp-config", default="configs/rtsp_profiles.yaml")
    parser.add_argument("--output-root", default="outputs/shadow_sessions")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--max-log-mib", type=int, default=64, help="per JSONL stream/file; rotation stops at max files without deleting logs")
    parser.add_argument("--max-log-files", type=int, default=6)
    parser.add_argument("--flush-seconds", type=float, default=5.0)
    parser.add_argument("--unknown-review-seconds", type=float, default=10.0)
    parser.add_argument("--review-cooldown-seconds", type=float, default=30.0)
    parser.add_argument("--near-boundary-margin", type=float, default=0.15)
    args = parser.parse_args()
    try:
        result = run_shadow_monitor(args)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "SHADOW_SESSION_FAILED", "detail": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
