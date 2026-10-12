"""Small, isolated YOLO person-tracking experiment for the Hikvision RTSP path.

This script intentionally does only RTSP/latest-frame input, COCO-person
detection, and anonymous tracker IDs.  It does not import MediaPipe, load V4,
write training data, or record raw video.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.detection.yolo_optional import YOLOUnavailableError, UltralyticsPersonTracker  # noqa: E402
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput  # noqa: E402
from src.posture_ai.inference.video_source import SOURCE_CLOSED, SOURCE_FAILED, VideoSource  # noqa: E402


def _resize_for_tracking(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    """Resize only the inference/display copy; source frames stay untouched."""

    height, width = frame.shape[:2]
    if max_width <= 0 or width <= max_width:
        return frame, 1.0
    scale = float(max_width) / float(width)
    return cv2.resize(frame, (max_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA), scale


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {"median": float(np.median(array)), "p95": float(np.percentile(array, 95)), "max": float(np.max(array))}


def _safe_health(health: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "source_type", "camera_id", "source_status", "connected", "frames_received",
        "decode_failures", "consecutive_failures", "reported_fps", "effective_received_fps",
        "resolution", "last_frame_age_seconds", "reconnect_count", "late_frame_count",
        "overwritten_frame_count", "backend", "buffer_control_requested", "buffer_control_applied",
    )
    return {key: health.get(key) for key in fields if key in health}


CONTROLLED_MARKERS = {
    ord("b"): "BASELINE",
    ord("c"): "CROSSING",
    ord("h"): "WALK_BEHIND",
    ord("f"): "FRONT_OCCLUSION",
    ord("e"): "EXIT_REENTER",
    ord("x"): "END",
}

CONTROLLED_DIAGNOSES = {
    ord("d"): "HUMAN_OBSERVED_DETECTOR_FAILURE",
    ord("s"): "HUMAN_OBSERVED_ASSOCIATION_FAILURE",
    ord("i"): "HUMAN_OBSERVED_INCONCLUSIVE",
}


def _box_at_point(boxes: list[Any], x: int, y: int) -> Any | None:
    """Return the highest-confidence box containing a human mouse click."""

    matches = [box for box in boxes if box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2]
    return max(matches, key=lambda box: box.confidence, default=None)


def _subject_snapshots(
    boxes: list[Any],
    subject_track_ids: dict[str, int | None],
    previous_track_ids: dict[str, int | None],
    *,
    frame_width: int,
    frame_height: int,
    fps: float | None,
    latency_ms: float | None,
    timestamp_seconds: float,
) -> list[dict[str, Any]]:
    """Serialize operator-defined A/B observations without asserting identity.

    A missing matching box means the selected tracker ID is not currently
    detected; it does *not* prove a detector or association failure.  The
    human operator records that distinction with D or S.
    """

    by_track_id = {box.track_id: box for box in boxes if box.track_id is not None}
    snapshots: list[dict[str, Any]] = []
    for subject in ("A", "B"):
        track_id = subject_track_ids.get(subject)
        box = by_track_id.get(track_id)
        snapshots.append({
            "operator_subject": subject,
            "track_id": track_id,
            "previous_track_id": previous_track_ids.get(subject),
            "bbox": None if box is None else [float(box.x1), float(box.y1), float(box.x2), float(box.y2)],
            "confidence": None if box is None else float(box.confidence),
            "detection_present": box is not None,
            "active_track_count": len(by_track_id),
            "fps": fps,
            "latency_ms": latency_ms,
            "timestamp_seconds": timestamp_seconds,
            "coordinate_space": "tracking_display_pixels",
            "frame_dimensions": {"width": frame_width, "height": frame_height},
        })
    return snapshots


def _draw(
    frame: np.ndarray,
    boxes: list[Any],
    *,
    tracker_name: str,
    fps: float,
    latency_ms: float | None,
    controlled: bool,
    selected_track_id: int | None,
    human_ids: dict[str, int | None],
    current_marker: str | None,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    for box in boxes:
        left, top, right, bottom = box.clipped(width, height).x1, box.clipped(width, height).y1, box.clipped(width, height).x2, box.clipped(width, height).y2
        track = "?" if box.track_id is None else str(box.track_id)
        color = (0, 220, 60) if box.track_id != selected_track_id else (0, 210, 255)
        labels = [f"TRACK {track}"]
        labels.extend(subject for subject, identity in human_ids.items() if identity == box.track_id)
        cv2.rectangle(output, (int(left), int(top)), (int(right), int(bottom)), color, 2)
        cv2.putText(output, f"{'/'.join(labels)}  {box.confidence:.0%}", (int(left), max(20, int(top) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    text = f"YOLO PERSON TRACKING | {tracker_name} | people: {len(boxes)} | {fps:.1f} FPS"
    if latency_ms is not None:
        text += f" | {latency_ms:.0f} ms"
    cv2.rectangle(output, (0, 0), (min(width, 920), 35), (25, 25, 25), -1)
    cv2.putText(output, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    if controlled:
        marker = current_marker or "NONE"
        controls = "Click person, 1=A, 2=B | B/C/H/F/E/X=markers | D=detector S=association I=inconclusive | Q quit"
        identities = f"HUMAN OBSERVED: A={human_ids['A'] or '-'} B={human_ids['B'] or '-'} | MARKER={marker}"
        cv2.rectangle(output, (0, 38), (min(width, 1120), 82), (25, 25, 25), -1)
        cv2.putText(output, identities, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1, cv2.LINE_AA)
        cv2.putText(output, controls, (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Latest-frame YOLO person tracking only; no MediaPipe/V4/raw video")
    parser.add_argument("--source", default="rtsp", help="rtsp/profile, local video, or USB index")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default="HIKVISION_MAIN")
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--rtsp-config", default="configs/rtsp_profiles.yaml")
    parser.add_argument("--weights", default="outputs/yolo_tracking/weights/yolo26n.pt")
    parser.add_argument("--tracker", choices=("bytetrack.yaml", "botsort.yaml"), default="bytetrack.yaml")
    parser.add_argument("--device", default="0", help="Ultralytics device, normally 0 or cpu")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-reconnects", type=int, default=3)
    parser.add_argument("--telemetry", default="outputs/metrics/yolo_rtsp_tracking_telemetry.jsonl")
    parser.add_argument("--summary", default="outputs/metrics/yolo_rtsp_tracking_summary.json")
    parser.add_argument("--scenario", default="UNSPECIFIED", help="Operator note only, e.g. CROSSING_BEHIND")
    parser.add_argument("--controlled", action="store_true", help="Enable human A/B ID assignment and timestamped scenario markers; requires display")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-sync-gpu-timing", action="store_true", help="Prefer throughput timing over synchronized approximate latency")
    args = parser.parse_args()
    if args.controlled and args.no_display:
        parser.error("--controlled requires the live display; remove --no-display")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if not 0.0 <= args.confidence <= 1.0 or not 0.0 <= args.iou <= 1.0:
        parser.error("confidence and iou must be in [0, 1]")

    try:
        import torch

        tracker = UltralyticsPersonTracker(args.weights, device=args.device, confidence=args.confidence, iou=args.iou, tracker=args.tracker)
    except (FileNotFoundError, YOLOUnavailableError, ValueError, ImportError) as exc:
        print(f"YOLO_TRACKER_UNAVAILABLE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    source = VideoSource(
        args.source,
        camera_id=args.camera_id,
        profile=args.profile,
        config_path=args.rtsp_config,
        reconnect=True,
        max_reconnects=args.max_reconnects,
        stale_after_seconds=1.5,
        low_latency=True,
    )
    latest = LatestFrameInput(source, asynchronous=source.source_type == "RTSP")
    telemetry_path = Path(args.telemetry)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not latest.open():
        failure_summary = {
            "status": "YOLO_RTSP_SOURCE_OPEN_FAILED",
            "scenario": args.scenario,
            "tracker": args.tracker,
            "model_file": Path(args.weights).name,
            "person_class_only": True,
            "final_source_health": _safe_health(source.health()),
            "raw_video_recorded": False,
            "mediapipe_used": False,
            "v4_used": False,
            "telemetry_path": str(telemetry_path),
        }
        summary_path.write_text(json.dumps(failure_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(failure_summary, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    started = time.perf_counter()
    frame_index = 0
    processed = 0
    source_failures = 0
    latency_ms: list[float] = []
    frame_age_ms: list[float] = []
    person_counts: list[int] = []
    all_ids: set[int] = set()
    active_ids: set[int] = set()
    temporarily_missing: set[int] = set()
    new_id_events = 0
    lost_id_events = 0
    reacquired_id_events = 0
    final_health: dict[str, Any] = {}
    close_reason = "LIMIT_REACHED"
    gpu_available = bool(torch.cuda.is_available())
    human_ids: dict[str, int | None] = {"A": None, "B": None}
    selected_track_id: int | None = None
    latest_boxes: list[Any] = []
    scenario_events: list[dict[str, Any]] = []
    current_marker: str | None = None
    previous_marker_track_ids: dict[str, int | None] = {"A": None, "B": None}

    def record_marker(
        marker: str,
        *,
        event_type: str,
        timestamp_seconds: float,
        current_boxes: list[Any],
        frame_width: int,
        frame_height: int,
        fps: float,
        latency_ms: float,
        set_current_scenario: bool,
    ) -> None:
        """Store a human-operated scenario boundary, never an inferred identity label."""

        nonlocal current_marker, previous_marker_track_ids
        if set_current_scenario:
            current_marker = marker
        current_ids = {int(box.track_id) for box in current_boxes if box.track_id is not None}
        observation = {
            "event": "HUMAN_OBSERVED_GROUND_TRUTH",
            "event_type": event_type,
            "marker": marker,
            "scenario": current_marker,
            "timestamp_seconds": timestamp_seconds,
            "human_subject_track_ids": dict(human_ids),
            "operator_subjects": _subject_snapshots(
                current_boxes, human_ids, previous_marker_track_ids,
                frame_width=frame_width, frame_height=frame_height,
                fps=fps, latency_ms=latency_ms, timestamp_seconds=timestamp_seconds,
            ),
            "active_track_ids": sorted(current_ids),
            "active_tracks": len(current_ids),
            "active_track_count": len(current_ids),
            "fps": fps,
            "latency_ms": latency_ms,
            "raw_video_recorded": False,
        }
        scenario_events.append(observation)
        telemetry.write(json.dumps(observation, ensure_ascii=False) + "\n")
        telemetry.flush()
        previous_marker_track_ids = dict(human_ids)

    def select_track_on_click(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        """Human selection only: click a visible box before assigning it to A or B."""

        nonlocal selected_track_id
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        selected = _box_at_point(latest_boxes, x, y)
        selected_track_id = None if selected is None else selected.track_id

    try:
        if not args.no_display:
            cv2.namedWindow("YOLO Person Tracking", cv2.WINDOW_NORMAL)
            if args.controlled:
                cv2.setMouseCallback("YOLO Person Tracking", select_track_on_click)
        with telemetry_path.open("w", encoding="utf-8") as telemetry:
            while (args.max_frames is None or frame_index < args.max_frames) and (args.max_seconds is None or time.perf_counter() - started < args.max_seconds):
                read = latest.read()
                if not read.ok or read.frame is None:
                    source_failures += 1
                    telemetry.write(json.dumps({
                        "event": "SOURCE_READ_FAILURE",
                        "frame_index": frame_index,
                        "timestamp_seconds": float(read.timestamp_seconds),
                        "reason": read.reason,
                        "source_health": _safe_health(read.health),
                    }, ensure_ascii=False) + "\n")
                    telemetry.flush()
                    if read.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                        close_reason = read.reason or read.state
                        break
                    continue

                frame, _scale = _resize_for_tracking(read.frame, args.max_width)
                if gpu_available and not args.no_sync_gpu_timing:
                    torch.cuda.synchronize()
                inference_started = time.perf_counter()
                boxes = tracker.track(frame)
                latest_boxes = boxes
                if gpu_available and not args.no_sync_gpu_timing:
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - inference_started) * 1000.0
                latency_ms.append(elapsed_ms)
                if read.frame_age_ms is not None:
                    frame_age_ms.append(float(read.frame_age_ms))
                current_ids = {int(box.track_id) for box in boxes if box.track_id is not None}
                new_ids = current_ids - all_ids
                lost_ids = active_ids - current_ids
                reacquired_ids = current_ids & temporarily_missing
                new_id_events += len(new_ids)
                lost_id_events += len(lost_ids)
                reacquired_id_events += len(reacquired_ids)
                all_ids.update(current_ids)
                temporarily_missing.update(lost_ids)
                temporarily_missing.difference_update(reacquired_ids)
                active_ids = current_ids
                person_counts.append(len(boxes))
                processed += 1
                elapsed_total = max(1e-9, time.perf_counter() - started)
                fps = processed / elapsed_total
                subject_snapshots = _subject_snapshots(
                    boxes, human_ids, previous_marker_track_ids,
                    frame_width=frame.shape[1], frame_height=frame.shape[0],
                    fps=fps, latency_ms=elapsed_ms, timestamp_seconds=float(read.timestamp_seconds),
                )
                telemetry.write(json.dumps({
                    "event": "TRACKING_FRAME",
                    "event_type": "TRACKING_FRAME",
                    "frame_index": frame_index,
                    "timestamp_seconds": float(read.timestamp_seconds),
                    "scenario": args.scenario,
                    "controlled_marker": current_marker,
                    "human_subject_track_ids": dict(human_ids) if args.controlled else None,
                    "operator_subjects": subject_snapshots if args.controlled else None,
                    "tracker": args.tracker,
                    "detected_people": len(boxes),
                    "active_track_count": len(current_ids),
                    "track_ids": sorted(current_ids),
                    "new_track_ids": sorted(new_ids),
                    "lost_track_ids": sorted(lost_ids),
                    "reacquired_track_ids": sorted(reacquired_ids),
                    "track_ids_missing_from_detector": sum(box.track_id is None for box in boxes),
                    "inference_latency_ms": elapsed_ms,
                    "frame_age_ms": read.frame_age_ms,
                    "processing_fps": fps,
                    "source_health": _safe_health(read.health),
                    "raw_video_recorded": False,
                }, ensure_ascii=False, allow_nan=False) + "\n")
                telemetry.flush()
                if not args.no_display:
                    rendered = _draw(
                        frame, boxes, tracker_name=args.tracker, fps=fps, latency_ms=elapsed_ms,
                        controlled=args.controlled, selected_track_id=selected_track_id,
                        human_ids=human_ids, current_marker=current_marker,
                    )
                    cv2.imshow("YOLO Person Tracking", rendered)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {27, ord("q"), ord("Q")}:
                        close_reason = "USER_EXIT"
                        break
                    if args.controlled:
                        if key in CONTROLLED_MARKERS:
                            marker = CONTROLLED_MARKERS[key]
                            record_marker(
                                marker, event_type="SCENARIO_MARKER",
                                timestamp_seconds=float(read.timestamp_seconds), current_boxes=boxes,
                                frame_width=frame.shape[1], frame_height=frame.shape[0],
                                fps=fps, latency_ms=elapsed_ms, set_current_scenario=True,
                            )
                            if marker == "END":
                                close_reason = "CONTROLLED_END"
                                break
                        elif key in CONTROLLED_DIAGNOSES:
                            record_marker(
                                CONTROLLED_DIAGNOSES[key], event_type="DIAGNOSTIC_MARKER",
                                timestamp_seconds=float(read.timestamp_seconds), current_boxes=boxes,
                                frame_width=frame.shape[1], frame_height=frame.shape[0],
                                fps=fps, latency_ms=elapsed_ms, set_current_scenario=False,
                            )
                        elif key == ord("1") and selected_track_id is not None:
                            human_ids["A"] = selected_track_id
                            record_marker(
                                "ASSIGN_A", event_type="OPERATOR_ASSIGNMENT",
                                timestamp_seconds=float(read.timestamp_seconds), current_boxes=boxes,
                                frame_width=frame.shape[1], frame_height=frame.shape[0],
                                fps=fps, latency_ms=elapsed_ms, set_current_scenario=False,
                            )
                        elif key == ord("2") and selected_track_id is not None:
                            human_ids["B"] = selected_track_id
                            record_marker(
                                "ASSIGN_B", event_type="OPERATOR_ASSIGNMENT",
                                timestamp_seconds=float(read.timestamp_seconds), current_boxes=boxes,
                                frame_width=frame.shape[1], frame_height=frame.shape[0],
                                fps=fps, latency_ms=elapsed_ms, set_current_scenario=False,
                            )
                frame_index += 1
    except KeyboardInterrupt:
        close_reason = "KEYBOARD_INTERRUPT"
    finally:
        final_health = latest.health()
        latest.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed_total = max(1e-9, time.perf_counter() - started)
    summary = {
        "status": "YOLO_RTSP_TRACKING_COMPLETED" if processed else "YOLO_RTSP_TRACKING_NO_FRAMES",
        "scenario": args.scenario,
        "tracker": args.tracker,
        "model_file": Path(args.weights).name,
        "person_class_only": True,
        "frames_processed": processed,
        "source_read_failures": source_failures,
        "mean_processing_fps": float(processed / elapsed_total),
        "detected_people_per_frame": {
            "max": int(max(person_counts, default=0)),
            "mean": None if not person_counts else float(np.mean(person_counts)),
        },
        "track_observations": {
            "unique_track_ids": sorted(all_ids),
            "new_track_id_events": new_id_events,
            "lost_track_id_events": lost_id_events,
            "reacquired_track_id_events": reacquired_id_events,
            "id_switches": "NOT_MEASURED_NO_MANUAL_IDENTITY_GROUND_TRUTH",
            "fragmentation": "NOT_MEASURED_NO_MANUAL_IDENTITY_GROUND_TRUTH; new_track_id_events are a diagnostic proxy only",
        },
        "controlled_human_observations": scenario_events if args.controlled else "NOT_ENABLED",
        "inference_latency_ms": _summary(latency_ms),
        "frame_age_ms": _summary(frame_age_ms),
        "gpu": {
            "torch_cuda_available": gpu_available,
            "device": torch.cuda.get_device_name(0) if gpu_available else "CPU",
            "peak_torch_allocated_mib": None if not gpu_available else float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
        },
        "final_source_health": _safe_health(final_health),
        "raw_video_recorded": False,
        "mediapipe_used": False,
        "v4_used": False,
        "close_reason": close_reason,
        "telemetry_path": str(telemetry_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if processed else 2


if __name__ == "__main__":
    raise SystemExit(main())
