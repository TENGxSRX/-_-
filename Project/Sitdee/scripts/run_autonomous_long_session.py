"""Visible, local-only YOLO/ByteTrack long-session observation lab."""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from collections import Counter, OrderedDict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.detection.yolo_optional import UltralyticsPersonTracker, YOLOUnavailableError  # noqa: E402
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput  # noqa: E402
from src.posture_ai.inference.video_source import SOURCE_CLOSED, SOURCE_FAILED, VideoSource  # noqa: E402
from src.posture_ai.monitoring.long_session import (  # noqa: E402
    CandidateSelector, CandidateSelectorConfig, LiveSessionState, LocalDashboardServer,
    atomic_write_json, performance_summary,
)
from src.posture_ai.monitoring.shadow import RotatingJsonlWriter, safe_source_metadata  # noqa: E402
from src.posture_ai.inference.pose_v2_stage import MediaPipeCropPoseBackend, PoseV2Stage  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TelemetryCapacityReached(RuntimeError):
    """A required append-only writer cannot accept another record."""


def write_required(writer: RotatingJsonlWriter, record: dict[str, Any], *, now_seconds: float) -> None:
    if not writer.write(record, now_seconds=now_seconds):
        raise TelemetryCapacityReached(writer.stem)


def session_id_now() -> str:
    return datetime.now().strftime("SESSION_%Y%m%d_%H%M%S")


def resize(frame: Any, max_width: int) -> Any:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    return cv2.resize(frame, (max_width, max(1, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)


def profile_for_source(source: str, profile: str | None) -> str | None:
    """Pass Hikvision profile only for RTSP sources; file paths stay files."""

    value = str(source).strip().lower()
    if value.startswith(("rtsp://", "rtsps://")) or value in {"rtsp", "hikvision", "camera"}:
        return profile
    return None


def draw_gui(frame: Any, boxes: list[Any], state: dict[str, Any], *, cuda_name: str) -> Any:
    output = frame.copy()
    for box in boxes:
        track_id = "?" if box.track_id is None else str(box.track_id)
        cv2.rectangle(output, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), (0, 220, 80), 2)
        cv2.putText(output, f"TRACK {track_id}  {box.confidence:.0%}", (round(box.x1), max(20, round(box.y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 2, cv2.LINE_AA)
    lines = [
        f"PEOPLE NOW {state['people_now']} | FPS {state.get('fps') or 0:.1f} | LATENCY {state.get('latency_median_ms') or 0:.0f} ms",
        f"SESSION {state['duration_seconds']/60:.1f} min | SOURCE {state['source_state']} | CUDA {cuda_name}",
        f"CANDIDATES {sum((state.get('data_collection') or {}).get('candidate_counts', {}).values())} | LOST {state['lost_events']} | RETURN {state['reacquired_events']}",
        "PRIVACY: RAW VIDEO OFF | ANONYMOUS TRACK IDs ONLY | Q/ESC TO STOP",
    ]
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 1050), 92), (20, 24, 30), -1)
    for index, line in enumerate(lines):
        cv2.putText(output, line, (10, 21 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def report_th(summary: dict[str, Any]) -> str:
    perf, collection = summary["performance"], summary["data_reduction"]
    return "\n".join([
        "# รายงาน Autonomous Long Session", "",
        f"Session: `{summary['session_id']}`", f"สถานะสิ้นสุด: **{summary['stopped_reason']}**",
        "", "## MEASURED", "",
        f"- ระยะเวลา: `{summary['duration_seconds']:.1f}` วินาที",
        f"- เฟรมที่ประมวลผล: `{summary['processed_frames']}`",
        f"- คนพร้อมกันสูงสุด: `{summary['people']['max_simultaneous']}`",
        f"- new/lost/reacquired: `{summary['tracking']['new']}/{summary['tracking']['lost']}/{summary['tracking']['reacquired']}`",
        f"- FPS ล่าสุด: `{perf['fps']}` | latency: `{perf['latency_ms']}`",
        f"- reconnect: `{summary['source'].get('reconnect_count', 0)}`",
        "", "## DATA REDUCTION", "",
        f"- quality accepted: `{collection['quality_accepted_samples']}`",
        f"- duplicates rejected: `{collection['duplicates_rejected']}`",
        f"- candidate counts: `{collection['candidate_counts']}`",
        "", "## INFERRED", "",
        f"- lifecycle pattern counts: `{summary['behavior_patterns']['counts']}`",
        "- LEAVE/RETURN เป็น pattern จากวงจร track ไม่ใช่ตัวตนจริงหรือพฤติกรรมที่ยืนยันแล้ว",
        "", "## PSEUDO_LABEL", "", "- ไม่ได้สร้าง pseudo-label ใน session นี้",
        "", "## UNKNOWN", "",
        "- Pose/posture: `POSE_PIPELINE_PENDING`",
        "- OOD: `NOT_EVALUATED_NO_REFERENCE_DISTRIBUTION`",
        "- ไม่มีการวัด posture accuracy หรือ ground truth",
        "", "## PRIVACY", "",
        "ไม่บันทึกวิดีโอดิบ ภาพนิ่ง ใบหน้า ชื่อบุคคล หรือ credential ของ RTSP",
    ]) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    session_id = args.session_id or session_id_now()
    session_dir = Path(args.output_root) / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    started_utc, started = utc_now(), time.perf_counter()
    # Construct optional runtime dependencies before starting background UI
    # services, so a model/configuration failure cannot leave a dashboard
    # thread running by itself.
    tracker = UltralyticsPersonTracker(args.weights, device=args.device, confidence=args.confidence, iou=args.iou, tracker="bytetrack.yaml")
    pose_stage = None
    if bool(getattr(args, "enable_pose_v2", False)):
        pose_backend = MediaPipeCropPoseBackend(
            getattr(args, "pose_model", "metadata/pose_landmarker_heavy.task"),
            visibility_threshold=float(getattr(args, "pose_visibility_threshold", 0.5)),
        )
        pose_stage = PoseV2Stage(
            pose_backend,
            max_pose_per_second=float(getattr(args, "max_pose_per_second", 2.0)),
            ambiguity_iou=float(args.occlusion_iou_threshold),
            visibility_threshold=float(getattr(args, "pose_visibility_threshold", 0.5)),
        )
    source = VideoSource(args.source, camera_id=args.camera_id, profile=profile_for_source(args.source, args.profile), config_path=args.rtsp_config,
                         reconnect=True, max_reconnects=args.max_reconnects, stale_after_seconds=args.stale_after_seconds, low_latency=True)
    source_input = LatestFrameInput(source, asynchronous=True)
    state = LiveSessionState(session_id, started_utc)
    dashboard = LocalDashboardServer(state, host="127.0.0.1", port=args.dashboard_port)
    dashboard_url = dashboard.start()
    if not args.no_browser:
        webbrowser.open(dashboard_url)

    max_bytes = int(args.max_log_mib * 1024 * 1024)
    telemetry = RotatingJsonlWriter(session_dir, "telemetry", max_bytes=max_bytes, max_files=args.max_log_files, flush_seconds=args.flush_seconds)
    candidates = RotatingJsonlWriter(session_dir, "candidates", max_bytes=max(1024 * 1024, max_bytes // 4), max_files=args.max_log_files, flush_seconds=args.flush_seconds)
    pose_writer = None if pose_stage is None else RotatingJsonlWriter(
        session_dir, "pose_telemetry", max_bytes=max(1024 * 1024, max_bytes // 2),
        max_files=args.max_log_files, flush_seconds=args.flush_seconds,
    )
    selector = CandidateSelector(CandidateSelectorConfig(
        representative_interval_seconds=args.representative_interval_seconds,
        transition_distance=args.transition_distance,
        low_confidence_threshold=args.low_confidence_threshold,
        stable_confidence_threshold=args.stable_confidence_threshold,
        occlusion_iou_threshold=args.occlusion_iou_threshold,
        event_cooldown_seconds=args.event_cooldown_seconds,
    ))
    manifest = {
        "schema_version": "autonomous_long_session_v1", "session_id": session_id,
        "started_utc": started_utc, "dashboard_url": dashboard_url,
        "source_type": "RTSP", "camera_id": args.camera_id, "profile": args.profile,
        "detector": Path(args.weights).name, "tracker": "bytetrack.yaml", "person_class_only": True,
        "pose_pipeline": "POSE_V2_ENABLED_TELEMETRY_ONLY" if pose_stage is not None else "POSE_V2_OPTIONAL_DISABLED", "v4_loaded": False,
        "raw_video_recorded": False, "snapshots_recorded": False,
        "identity": "ANONYMOUS_TRANSIENT_TRACK_IDS_ONLY", "automatic_labels_created": False,
        "shadow_learning": "SHADOW_DATA_POOL_BUILDING_NO_MODEL_TRAINING",
    }
    atomic_write_json(session_dir / "session_manifest.json", manifest)

    active_ids: set[int] = set()
    seen_ids: set[int] = set()
    missing_ids: set[int] = set()
    first_seen: OrderedDict[int, float] = OrderedDict()
    recent_events: deque[str] = deque(maxlen=40)
    latency_window: deque[float] = deque(maxlen=600)
    inference_times: deque[float] = deque(maxlen=120)
    patterns: Counter[str] = Counter()
    new_count = lost_count = reacquired_count = short_lived = processed = 0
    max_people = source_failures = 0
    stopped_reason = "DURATION_COMPLETED"
    last_telemetry_at = last_snapshot_at = -1e12
    final_health: dict[str, Any] = {}
    deadline = started + args.duration_minutes * 60.0
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    try:
        if not source_input.open():
            raise RuntimeError(f"RTSP source unavailable: {source.spec.source_display}")
        if not args.no_display:
            cv2.namedWindow("Autonomous CCTV Long Session", cv2.WINDOW_NORMAL)
        while time.perf_counter() < deadline:
            read = source_input.read()
            wall_now = time.perf_counter()
            elapsed = wall_now - started
            if not read.ok or read.frame is None:
                source_failures += 1
                health = safe_source_metadata(read.health)
                recent_events.append(f"{elapsed:.1f}s SOURCE {read.state}: {read.reason}")
                state.update(duration_seconds=elapsed, source_state=read.state, recent_events=list(recent_events))
                write_required(telemetry, {"record_type": "SOURCE_EVENT", "session_id": session_id, "timestamp_seconds": elapsed,
                        "timestamp_utc": utc_now(), "source_state": read.state, "reason": read.reason, "source": health,
                        "raw_video_recorded": False}, now_seconds=elapsed)
                if read.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                    stopped_reason = f"SOURCE_STOPPED_{read.state}"
                    break
                continue

            frame = resize(read.frame, args.max_width)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_start = time.perf_counter()
            boxes = tracker.track(frame)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - infer_start) * 1000.0
            latency_window.append(latency_ms)
            inference_times.append(time.perf_counter())
            fps = None if len(inference_times) < 2 else (len(inference_times) - 1) / max(1e-9, inference_times[-1] - inference_times[0])
            current_ids = {int(box.track_id) for box in boxes if box.track_id is not None}
            new_ids = current_ids - seen_ids
            lost_ids = active_ids - current_ids
            reacquired_ids = current_ids & missing_ids
            seen_ids.update(new_ids)
            new_count += len(new_ids); lost_count += len(lost_ids); reacquired_count += len(reacquired_ids)
            for track_id in new_ids:
                first_seen[track_id] = elapsed
                recent_events.append(f"{elapsed:.1f}s TRACK {track_id} NEW")
            for track_id in lost_ids:
                age = elapsed - first_seen.get(track_id, elapsed)
                if age < args.short_track_seconds:
                    short_lived += 1
                recent_events.append(f"{elapsed:.1f}s TRACK {track_id} LOST")
                patterns["PATTERN_TRACK_DISAPPEARED"] += 1
                item = selector.event_candidate(track_id, "TRACK_LOSS", elapsed)
                if item is not None:
                    item.update({"session_id": session_id, "timestamp_utc": utc_now()})
                    write_required(candidates, item, now_seconds=elapsed)
            for track_id in reacquired_ids:
                recent_events.append(f"{elapsed:.1f}s TRACK {track_id} REACQUIRED")
                patterns["PATTERN_TRACK_RETURNED"] += 1
            missing_ids.update(lost_ids); missing_ids.difference_update(reacquired_ids)
            active_ids = current_ids
            while len(first_seen) > args.max_track_memory:
                first_seen.popitem(last=False)
            max_people = max(max_people, len(boxes)); processed += 1

            selected = selector.inspect(boxes, timestamp_seconds=elapsed, frame_width=frame.shape[1], frame_height=frame.shape[0], new_ids=new_ids, reacquired_ids=reacquired_ids)
            if pose_stage is not None and pose_writer is not None:
                selected_ids = {int(item["anonymous_track_id"]) for item in selected}
                selected_boxes = [box for box in boxes if box.track_id is not None and int(box.track_id) in selected_ids]
                pose_results = pose_stage.process(frame, selected_boxes, elapsed, all_boxes=boxes)
                for pose_result in pose_results:
                    write_required(
                        pose_writer,
                        pose_result.to_record(
                            session_id=session_id, timestamp_seconds=elapsed,
                            timestamp_utc=utc_now(), frame_index=read.frame_index,
                        ),
                        now_seconds=elapsed,
                    )
            for item in selected:
                item.update({"session_id": session_id, "timestamp_utc": utc_now()})
                write_required(candidates, item, now_seconds=elapsed)

            health = safe_source_metadata(read.health)
            perf = performance_summary(list(latency_window))
            collection = selector.summary()
            active_lifetimes = [elapsed - first_seen[track_id] for track_id in current_ids if track_id in first_seen]
            lifetime_summary = performance_summary(active_lifetimes)
            state.update(
                duration_seconds=elapsed, source_state=health.get("source_status", "UNKNOWN"), people_now=len(boxes),
                max_people=max_people, active_tracks=sorted(current_ids), new_track_events=new_count,
                lost_events=lost_count, reacquired_events=reacquired_count, short_lived_tracks=short_lived,
                fragmentation_proxy=new_count + short_lived, processed_frames=processed, fps=fps,
                track_lifetime_seconds={"median": lifetime_summary["median"], "max": lifetime_summary["max"]},
                latency_median_ms=perf["median"], latency_p95_ms=perf["p95"],
                reconnect_count=int(health.get("reconnect_count", 0) or 0),
                overwritten_frames=int(health.get("overwritten_frame_count", 0) or 0),
                recent_events=list(recent_events), data_collection=collection,
                behavior_patterns={"status": "TRACK_LIFECYCLE_PATTERNS_ONLY", "counts": dict(patterns)},
                shadow_learning={"status": "SHADOW_DATA_POOL_READY" if selected else "SHADOW_DATA_POOL_BUILDING",
                                 "clustering": "NOT_RUN_METADATA_ONLY", "candidate_model": "NOT_TRAINED"},
                pose_pipeline="POSE_V2_ENABLED_TELEMETRY_ONLY" if pose_stage is not None else "POSE_V2_OPTIONAL_DISABLED",
            )
            if elapsed - last_telemetry_at >= 1.0 / args.telemetry_hz:
                write_required(telemetry, {
                    "record_type": "TRACKING_TELEMETRY", "session_id": session_id, "timestamp_seconds": elapsed,
                    "timestamp_utc": utc_now(), "frame_index": read.frame_index, "source": health,
                    "people_now": len(boxes), "active_track_ids": sorted(current_ids),
                    "tracks": [{"track_id": box.track_id, "bbox": [box.x1, box.y1, box.x2, box.y2], "confidence": box.confidence} for box in boxes],
                    "new_track_ids": sorted(new_ids), "lost_track_ids": sorted(lost_ids), "reacquired_track_ids": sorted(reacquired_ids),
                    "fps": fps, "latency_ms": latency_ms, "frame_age_ms": read.frame_age_ms,
                    "pose_pipeline": "POSE_PIPELINE_PENDING", "labels": "NONE", "raw_video_recorded": False,
                }, now_seconds=elapsed)
                last_telemetry_at = elapsed
            if elapsed - last_snapshot_at >= args.dashboard_refresh_seconds:
                atomic_write_json(session_dir / "live_status.json", state.snapshot())
                last_snapshot_at = elapsed
            if not args.no_display:
                cv2.imshow("Autonomous CCTV Long Session", draw_gui(frame, boxes, state.snapshot(), cuda_name=cuda_name))
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q"), ord("Q")}:
                    stopped_reason = "USER_EXIT"
                    break
    except KeyboardInterrupt:
        stopped_reason = "KEYBOARD_INTERRUPT"
    except TelemetryCapacityReached:
        stopped_reason = "TELEMETRY_CAPACITY_REACHED"
    finally:
        final_health = safe_source_metadata(source_input.health())
        source_input.close()
        dashboard.stop()
        telemetry.close(); candidates.close()
        if pose_stage is not None:
            pose_stage.close()
        if pose_writer is not None:
            pose_writer.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    duration = time.perf_counter() - started
    latency = performance_summary(list(latency_window))
    fps_value = None if len(inference_times) < 2 else (len(inference_times) - 1) / max(1e-9, inference_times[-1] - inference_times[0])
    pose_summary = {"status": "DISABLED", "v4_called": False} if pose_stage is None else pose_stage.summary()
    summary = {
        "schema_version": "autonomous_long_session_summary_v1", "session_id": session_id,
        "started_utc": started_utc, "ended_utc": utc_now(), "duration_seconds": duration,
        "stopped_reason": stopped_reason, "processed_frames": processed, "source_read_failures": source_failures,
        "people": {"max_simultaneous": max_people, "unique_track_ids_observed": len(seen_ids)},
        "tracking": {"new": new_count, "lost": lost_count, "reacquired": reacquired_count, "short_lived": short_lived,
                     "fragmentation_proxy": new_count + short_lived, "identity_switches": "UNKNOWN_NO_HUMAN_GROUND_TRUTH"},
        "performance": {"fps": fps_value, "latency_ms": latency}, "source": final_health,
        "data_reduction": selector.summary(),
        "behavior_patterns": {"status": "INFERRED_FROM_TRACK_LIFECYCLE_ONLY", "counts": dict(patterns), "sit_stand": "UNKNOWN_NO_POSE"},
        "pose_pipeline": "POSE_V2_ENABLED_TELEMETRY_ONLY" if pose_stage is not None else "POSE_V2_OPTIONAL_DISABLED",
        "pose": pose_summary,
        "shadow_learning": "SHADOW_DATA_POOL_READY_NO_MODEL_TRAINED" if selector.summary()["candidate_pool_size"] else "SHADOW_DATA_POOL_EMPTY_NO_MODEL_TRAINED",
        "safety": {"raw_video_recorded": False, "snapshots_recorded": False, "automatic_labels_created": False,
                   "ground_truth_created": False, "credentials_logged": False},
        "writers": {"telemetry": telemetry.summary(), "candidates": candidates.summary(), **({} if pose_writer is None else {"pose_telemetry": pose_writer.summary()})},
        "important_anomalies": {
            "source_read_failures": source_failures, "short_lived_tracks": short_lived,
            "telemetry_capacity_reached": telemetry.summary()["capacity_reached"],
            "candidate_capacity_reached": candidates.summary()["capacity_reached"],
            "pose_capacity_reached": False if pose_writer is None else pose_writer.summary()["capacity_reached"],
            "identity_switches": "UNKNOWN_NO_HUMAN_GROUND_TRUTH",
        },
        "artifacts": {"session_dir": str(session_dir), "manifest": "session_manifest.json", "live_status": "live_status.json",
                      "summary": "session_summary.json", "report": "session_report_th.md"},
    }
    manifest.update({"ended_utc": summary["ended_utc"], "stopped_reason": stopped_reason, "summary": "session_summary.json",
                     "pose": pose_summary, "writers": summary["writers"]})
    state.update(duration_seconds=duration, stopped_reason=stopped_reason,
                 recent_events=list(recent_events) + ([stopped_reason] if stopped_reason == "TELEMETRY_CAPACITY_REACHED" else []))
    atomic_write_json(session_dir / "live_status.json", state.snapshot())
    atomic_write_json(session_dir / "session_summary.json", summary)
    atomic_write_json(session_dir / "session_manifest.json", manifest)
    (session_dir / "session_report_th.md").write_text(report_th(summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Visible local-only YOLO/ByteTrack autonomous long session")
    parser.add_argument("--source", default="rtsp")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default="HIKVISION_MAIN")
    parser.add_argument("--camera-id", default="CAM02")
    parser.add_argument("--rtsp-config", default="configs/rtsp_profiles.yaml")
    parser.add_argument("--weights", default="outputs/yolo_tracking/weights/yolo26n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--duration-minutes", type=float, default=300.0)
    parser.add_argument("--max-reconnects", type=int, default=1000)
    parser.add_argument("--stale-after-seconds", type=float, default=1.5)
    parser.add_argument("--telemetry-hz", type=float, default=2.0)
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--dashboard-refresh-seconds", type=float, default=2.0)
    parser.add_argument("--output-root", default="outputs/long_sessions")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--max-log-mib", type=int, default=128)
    parser.add_argument("--max-log-files", type=int, default=8)
    parser.add_argument("--flush-seconds", type=float, default=5.0)
    parser.add_argument("--representative-interval-seconds", type=float, default=30.0)
    parser.add_argument("--transition-distance", type=float, default=0.12)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.45)
    parser.add_argument("--stable-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--occlusion-iou-threshold", type=float, default=0.25)
    parser.add_argument("--event-cooldown-seconds", type=float, default=3.0)
    parser.add_argument("--short-track-seconds", type=float, default=2.0)
    parser.add_argument("--max-track-memory", type=int, default=4096)
    parser.add_argument("--enable-pose-v2", action="store_true", help="Optional MediaPipe crop telemetry stage; disabled by default")
    parser.add_argument("--max-pose-per-second", type=float, default=2.0, help="Global pose-attempt budget across all tracks")
    parser.add_argument("--pose-model", default="metadata/pose_landmarker_heavy.task")
    parser.add_argument("--pose-visibility-threshold", type=float, default=0.5)
    parser.add_argument("--no-display", action="store_true", help="Smoke tests only; long sessions should keep GUI visible")
    parser.add_argument("--no-browser", action="store_true", help="Dashboard still runs; do not open the browser automatically")
    args = parser.parse_args()
    if args.duration_minutes <= 0 or args.telemetry_hz <= 0 or args.max_log_mib < 1 or args.max_log_files < 1 or args.max_pose_per_second <= 0 or not 0.0 <= args.pose_visibility_threshold <= 1.0:
        parser.error("duration, telemetry rate, and log limits must be positive")
    try:
        summary = run(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError, OSError, YOLOUnavailableError) as exc:
        print(json.dumps({"status": "AUTONOMOUS_LONG_SESSION_FAILED", "detail": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "AUTONOMOUS_LONG_SESSION_STOPPED_CAPACITY" if summary["stopped_reason"] == "TELEMETRY_CAPACITY_REACHED" else "AUTONOMOUS_LONG_SESSION_COMPLETED", "session_id": summary["session_id"],
                      "stopped_reason": summary["stopped_reason"], "artifacts": summary["artifacts"]}, ensure_ascii=False, indent=2))
    return 2 if summary["stopped_reason"] == "TELEMETRY_CAPACITY_REACHED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
