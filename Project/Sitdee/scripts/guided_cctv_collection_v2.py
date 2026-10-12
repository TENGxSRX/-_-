"""Attended, target-locked CCTV ground-truth collection V2.

The collector uses a human-selected anonymous target.  It never copies a V4
prediction or geometry indicator into ``human_label`` and never changes the
selected target to another visible person automatically.  Raw video and
screenshots are off by design; accepted samples contain landmarks, features,
and audit metadata only.

Run this with ``.venv_yolo`` when ``--person-backend yolo`` is selected.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.data.controlled_collection import (  # noqa: E402
    ControlledSampleWriter,
    compact_json,
    evaluate_collection_quality,
    make_session_id,
    validate_safe_id,
    validate_subject_id,
)
from src.posture_ai.data.posture_taxonomy import load_posture_taxonomy  # noqa: E402
from src.posture_ai.data.target_lock_collection import (  # noqa: E402
    TARGET_LOCK_COLLECTION_SCHEMA_VERSION,
    TARGET_LOCK_EXTRA_SAMPLE_COLUMNS,
    canonical_cctv_diagnostic,
    serialise_landmarks,
    target_lock_collection_gate,
)
from src.posture_ai.detection.yolo_optional import (  # noqa: E402
    PersonBox,
    YOLOUnavailableError,
    UltralyticsPersonTracker,
    crop_frame,
)
from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig  # noqa: E402
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput  # noqa: E402
from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner  # noqa: E402
from src.posture_ai.inference.stream_runtime import resize_for_display, resize_for_inference  # noqa: E402
from src.posture_ai.inference.video_source import SOURCE_CLOSED, SOURCE_FAILED, VideoSource  # noqa: E402
from src.posture_ai.multiperson.mediapipe_tasks import MultiPersonMediaPipeBackend  # noqa: E402
from src.posture_ai.multiperson.overlay import draw_multiperson_overlay  # noqa: E402
from src.posture_ai.multiperson.runtime import MultiPersonRuntime, MultiPersonRuntimeConfig, resolve_performance_preset, should_process_frame  # noqa: E402
from src.posture_ai.multiperson.target_lock import TargetCandidate, TargetLockController  # noqa: E402
from src.posture_ai.multiperson.telemetry import frame_to_event  # noqa: E402
from src.posture_ai.multiperson.yolo_association import associate_pose_items_to_person_boxes  # noqa: E402


DEFAULT_OUTPUT_ROOT = "data/controlled_cctv_target_lock_v2"
DEFAULT_TELEMETRY_ROOT = "outputs/live/target_lock_telemetry"


def _safe_json(value: Any) -> Any:
    """Make telemetry JSON-safe without retaining secret-bearing objects."""

    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_source_health(health: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only non-secret source health for the standalone JSONL telemetry."""

    fields = (
        "source_type", "camera_id", "source_status", "connected", "frames_received",
        "decode_failures", "consecutive_failures", "reported_fps", "effective_received_fps",
        "resolution", "last_frame_age_seconds", "reconnect_count", "late_frame_count",
        "overwritten_frame_count", "last_failure_reason", "backend",
        "buffer_control_requested", "buffer_control_applied", "reader_thread_alive",
    )
    return _safe_json({key: health.get(key) for key in fields if key in health})


def _v4_prediction(runner: RGBNativeV4Runner, subject: Any, *, visibility_threshold: float) -> dict[str, Any] | None:
    if not bool(subject.core_valid) or subject.world_landmarks is None:
        return None
    try:
        return runner.predict_world_landmarks(
            subject.world_landmarks,
            image_visibility=subject.landmarks,
            visibility_threshold=visibility_threshold,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _identity_for(person: PersonBox, subject_id: str | None) -> tuple[str, str]:
    if person.track_id is not None:
        return f"YOLO_TRACK_{int(person.track_id)}", "YOLO_TRACK"
    if subject_id:
        return f"MEDIAPIPE_{subject_id}", "MEDIAPIPE_EPHEMERAL"
    return "MEDIAPIPE_UNMATCHED", "MEDIAPIPE_EPHEMERAL"


def _active_subjects(result: Any) -> list[Any]:
    return [subject for subject in result.subjects if str(getattr(subject, "tracking_state", "")) in {"NEW", "ACTIVE"}]


def _build_target_candidates(
    subjects: list[Any],
    people: list[PersonBox],
    *,
    frame_width: int,
    frame_height: int,
    person_backend: str,
) -> tuple[list[TargetCandidate], dict[str, PersonBox], dict[str, Any], dict[str, Any]]:
    """Build lock candidates from current-frame full-frame association only."""

    people_by_key: dict[str, PersonBox] = {}
    subjects_by_key: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {
        "person_backend": person_backend,
        "person_box_count": len(people),
        "association_strategy": "MEDIAPIPE_EPHEMERAL_ONLY" if person_backend != "yolo" else "FULL_FRAME_SPATIAL_ONE_TO_ONE",
        "association_count": 0,
        "ambiguous_association_count": 0,
        "unmatched_pose_count": 0,
        "unmatched_person_count": 0,
    }
    if person_backend != "yolo":
        candidates: list[TargetCandidate] = []
        for subject in subjects:
            key = f"MEDIAPIPE_{subject.subject_id}"
            candidate = TargetCandidate(
                identity_key=key,
                identity_mode="MEDIAPIPE_EPHEMERAL",
                runtime_subject_id=subject.subject_id,
                yolo_track_id=None,
                bbox=subject.bbox,
                association_state="ASSOCIATED",
                association_score=1.0,
                pose_valid=subject.world_landmarks is not None,
                core_valid=bool(subject.core_valid),
            )
            candidates.append(candidate)
            subjects_by_key[key] = subject
        diagnostics["association_count"] = len(candidates)
        return candidates, people_by_key, subjects_by_key, diagnostics

    association = associate_pose_items_to_person_boxes(subjects, people, frame_width=frame_width, frame_height=frame_height)
    diagnostics.update({
        "association_count": len(association.associations),
        "ambiguous_association_count": sum(item.state == "AMBIGUOUS" for item in association.associations),
        "unmatched_pose_count": len(association.unmatched_pose_indices),
        "unmatched_person_count": len(association.unmatched_person_indices),
    })
    candidates = []
    for item in association.associations:
        subject = subjects[item.pose_index]
        person = people[item.person_index]
        key, mode = _identity_for(person, subject.subject_id)
        candidate = TargetCandidate(
            identity_key=key,
            identity_mode=mode,
            runtime_subject_id=subject.subject_id,
            yolo_track_id=person.track_id,
            bbox=person.normalized(frame_width, frame_height),
            association_state=item.state,
            association_score=item.score,
            pose_valid=subject.world_landmarks is not None,
            core_valid=bool(subject.core_valid),
        )
        candidates.append(candidate)
        people_by_key[key] = person
        subjects_by_key[key] = subject
    return candidates, people_by_key, subjects_by_key, diagnostics


def _padded_crop_box(box: PersonBox, *, padding_fraction: float) -> PersonBox:
    pad = max(0.0, float(padding_fraction))
    return PersonBox(
        box.x1 - box.width * pad,
        box.y1 - box.height * pad,
        box.x2 + box.width * pad,
        box.y2 + box.height * pad,
        box.confidence,
        box.class_id,
        box.track_id,
    )


def _crop_comparison(
    crop_backend: MultiPersonMediaPipeBackend | None,
    runner: RGBNativeV4Runner,
    frame: np.ndarray,
    box: PersonBox | None,
    *,
    timestamp_ms: int,
    visibility_threshold: float,
    full_prediction: Mapping[str, Any] | None,
    padding_fraction: float,
) -> dict[str, Any]:
    """Diagnostic-only crop experiment; it never supplies collection features."""

    if crop_backend is None or box is None:
        return {"status": "NOT_RUN"}
    crop, clipped = crop_frame(frame, _padded_crop_box(box, padding_fraction=padding_fraction))
    try:
        detections = crop_backend.detect(crop, timestamp_ms=timestamp_ms)
    except (RuntimeError, TypeError, ValueError) as exc:
        return {"status": f"ERROR_{type(exc).__name__.upper()}"}
    if not detections:
        return {"status": "NO_CROP_POSE", "crop_box": [clipped.x1, clipped.y1, clipped.x2, clipped.y2]}
    detected = detections[0]
    crop_prediction = None
    if detected.core_valid and detected.world_landmarks is not None:
        try:
            crop_prediction = runner.predict_world_landmarks(
                detected.world_landmarks,
                image_visibility=detected.landmarks,
                visibility_threshold=visibility_threshold,
            )
        except (KeyError, TypeError, ValueError):
            crop_prediction = None
    full_features = None if full_prediction is None else full_prediction.get("feature_values")
    crop_features = None if crop_prediction is None else crop_prediction.get("feature_values")
    feature_l1 = None
    if isinstance(full_features, list) and isinstance(crop_features, list) and len(full_features) == len(crop_features):
        feature_l1 = float(np.mean(np.abs(np.asarray(full_features, dtype=float) - np.asarray(crop_features, dtype=float))))
    return {
        "status": "MEASURED_DIAGNOSTIC_ONLY",
        "crop_box": [clipped.x1, clipped.y1, clipped.x2, clipped.y2],
        "crop_pose_valid": bool(detected.world_landmarks is not None),
        "crop_core_valid": bool(detected.core_valid),
        "full_v4_prediction": None if full_prediction is None else full_prediction.get("label"),
        "crop_v4_prediction": None if crop_prediction is None else crop_prediction.get("label"),
        "v4_prediction_agrees": None if full_prediction is None or crop_prediction is None else full_prediction.get("label") == crop_prediction.get("label"),
        "v4_feature_mean_absolute_difference": feature_l1,
        "collection_features_remain": "FULL_FRAME_MEDIAPIPE_WORLD_ONLY",
    }


def _draw_yolo_and_lock(
    frame: np.ndarray,
    people: Iterable[PersonBox],
    candidates: Iterable[TargetCandidate],
    lock: Any,
    *,
    person_frame_width: int,
    person_frame_height: int,
    capture_requested: bool,
    selected_label: str | None,
    labels_help: str,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    candidate_by_track = {candidate.yolo_track_id: candidate for candidate in candidates if candidate.yolo_track_id is not None}
    for person in people:
        left, top, right, bottom = person.normalized(person_frame_width, person_frame_height)
        box = PersonBox(left * width, top * height, right * width, bottom * height, person.confidence, person.class_id, person.track_id)
        candidate = candidate_by_track.get(person.track_id)
        is_locked = lock.identity_key is not None and person.track_id is not None and lock.target_yolo_track_id == person.track_id
        color = (0, 200, 0) if candidate is not None and candidate.association_state == "ASSOCIATED" else (0, 165, 255)
        if is_locked:
            color = (0, 0, 255) if lock.auto_paused else (255, 220, 0)
        cv2.rectangle(output, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
        identifier = "TRACK ?" if person.track_id is None else f"TRACK {person.track_id}"
        state = "NO_POSE" if candidate is None else candidate.association_state
        cv2.putText(output, f"{identifier} {state}", (int(box.x1), max(18, int(box.y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

    panel_height = min(142, height)
    panel_top = max(0, height - panel_height)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, panel_top), (width, height), (30, 25, 50), -1)
    output[:] = cv2.addWeighted(overlay, 0.80, output, 0.20, 0.0)
    state = "ARMED" if capture_requested else "PAUSED"
    lines = [
        f"TARGET LOCK: {lock.state} | {state} | HUMAN GROUND TRUTH: {selected_label or 'NO_LABEL'}",
        f"TRACK: {lock.target_yolo_track_id if lock.target_yolo_track_id is not None else 'NONE'} | AUTO-PAUSE: {'YES' if lock.auto_paused else 'NO'} | {lock.pause_reason or 'READY'}",
        f"{labels_help} | 0=NO_LABEL/SKIP | SPACE=arm/pause | TAB/N=next target | C=clear target",
        "Click a separated person box to lock. Never use V4/Geometry as the human label. Q/ESC=finish.",
    ]
    for line_index, line in enumerate(lines):
        y = panel_top + 22 + 28 * line_index
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _write_telemetry(handle: Any, event: Mapping[str, Any]) -> None:
    handle.write(json.dumps(_safe_json(event), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Target-locked human CCTV collector V2; raw video is disabled")
    parser.add_argument("--source", default="rtsp", help="rtsp/profile, local video file, or explicit USB index")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default="HIKVISION_MAIN")
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--subject-id", required=True, help="Anonymous participant code such as P001")
    parser.add_argument("--clip-id", required=True, help="Safe session/clip code")
    parser.add_argument("--operator-code", default="ANONYMOUS_OPERATOR", help="Non-identifying operator code")
    parser.add_argument("--model", default="outputs/models/rgb_native_v4")
    parser.add_argument("--pose-model", default="metadata/pose_landmarker_heavy.task")
    parser.add_argument("--config", default="configs/posture_engine.yaml")
    parser.add_argument("--rtsp-config", default="configs/rtsp_profiles.yaml")
    parser.add_argument("--taxonomy", default="configs/posture_taxonomy.yaml")
    parser.add_argument("--preset", choices=("RTSP_DESKTOP", "RTSP_BALANCED", "RTSP_LOW_LATENCY", "RTSP_CCTV"), default="RTSP_CCTV")
    parser.add_argument("--max-people", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--process-every", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--max-reconnects", type=int, default=5)
    parser.add_argument("--person-backend", choices=("yolo", "mediapipe"), default="yolo")
    parser.add_argument("--yolo-weights", default="outputs/yolo_person/weights/yolo11n.pt")
    parser.add_argument("--yolo-device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--yolo-tracker", default="bytetrack.yaml")
    parser.add_argument("--yolo-confidence", type=float, default=0.35)
    parser.add_argument("--collection-mode", choices=("association", "crop-experimental"), default="association")
    parser.add_argument("--crop-padding", type=float, default=0.10)
    parser.add_argument("--reacquisition-window-seconds", type=float, default=1.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.5)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--telemetry-path", default=None)
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--no-display", action="store_true", help="Run read-only diagnostics; no target can be selected and no samples are written")
    args = parser.parse_args()

    try:
        participant_id = validate_subject_id(args.subject_id)
        clip_id = validate_safe_id(args.clip_id, field_name="clip_id")
        taxonomy = load_posture_taxonomy(args.taxonomy)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.process_every is not None and args.process_every < 1:
        parser.error("--process-every must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if args.sample_interval_seconds < 0.0 or args.reacquisition_window_seconds <= 0.0 or args.crop_padding < 0.0:
        parser.error("collection timing and crop values are invalid")

    runtime_preset = resolve_performance_preset(args.preset, args.process_every)
    posture_config = PostureEngineConfig.from_yaml(args.config)
    runner = RGBNativeV4Runner(args.model)
    writer = ControlledSampleWriter(
        args.output_root,
        feature_names=runner.expected_names,
        allowed_labels=taxonomy.collection_labels,
        schema_version=TARGET_LOCK_COLLECTION_SCHEMA_VERSION,
        extra_columns=TARGET_LOCK_EXTRA_SAMPLE_COLUMNS,
    )
    session_id = make_session_id(participant_id, clip_id)
    telemetry_path = Path(args.telemetry_path) if args.telemetry_path else Path(DEFAULT_TELEMETRY_ROOT) / f"{session_id}.jsonl"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)

    yolo_tracker = None
    if args.person_backend == "yolo":
        try:
            yolo_tracker = UltralyticsPersonTracker(
                args.yolo_weights,
                device=args.yolo_device,
                confidence=args.yolo_confidence,
                tracker=args.yolo_tracker,
            )
        except (FileNotFoundError, YOLOUnavailableError, ValueError) as exc:
            print(f"YOLO_TARGET_LOCK_BACKEND_UNAVAILABLE: {type(exc).__name__}: {exc}", file=sys.stderr)
            print("Use the isolated .venv_yolo and run the explicit setup tool before collection.", file=sys.stderr)
            return 2

    source = VideoSource(
        args.source,
        camera_id=args.camera_id,
        profile=args.profile,
        config_path=args.rtsp_config,
        reconnect=True,
        max_reconnects=args.max_reconnects,
        stale_after_seconds=1.0 if runtime_preset["name"] == "RTSP_CCTV" else 2.0,
        low_latency=True,
    )
    latest = LatestFrameInput(source, asynchronous=source.source_type == "RTSP")
    if not latest.open():
        print(f"Could not open source: {source.spec.source_display}", file=sys.stderr)
        return 2

    backend = MultiPersonMediaPipeBackend(
        args.pose_model,
        max_people=args.max_people,
        running_mode="video",
        visibility_threshold=posture_config.minimum_pose_visibility,
    )
    crop_backend = None
    if args.collection_mode == "crop-experimental":
        crop_backend = MultiPersonMediaPipeBackend(
            args.pose_model,
            max_people=1,
            running_mode="video",
            visibility_threshold=posture_config.minimum_pose_visibility,
        )
    runtime = MultiPersonRuntime(
        runner,
        runtime_config=MultiPersonRuntimeConfig(
            max_people=args.max_people,
            visibility_threshold=posture_config.minimum_pose_visibility,
            camera_id=args.camera_id or source.camera_id or "CAM01",
        ),
        posture_config=posture_config,
    )
    lock_controller = TargetLockController(yolo_reacquisition_window_seconds=args.reacquisition_window_seconds)
    selected_label: str | None = None
    capture_requested = False
    pending_click: dict[str, tuple[float, float] | None] = {"point": None}
    candidate_cycle_index = -1
    frame_index = 0
    processed_frames = 0
    accepted_samples = 0
    rejection_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    last_sample_timestamp: float | None = None
    last_rejection_timestamp: float | None = None
    source_origin: float | None = None
    last_pose_timestamp: float | None = None
    started = time.perf_counter()
    close_reason = "USER_EXIT"
    last_result = None
    current_candidates: list[TargetCandidate] = []
    current_people: list[PersonBox] = []
    current_people_by_key: dict[str, PersonBox] = {}
    current_subjects_by_key: dict[str, Any] = {}
    latest_lock = lock_controller.snapshot
    label_key_map = {str(index + 1): label for index, label in enumerate(taxonomy.collection_labels[:9])}
    labels_help = "  ".join(f"{key}={label}" for key, label in label_key_map.items())

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # The rendered image size is fixed to the last display frame.  A click
        # is still explicit; if it lies in overlapping boxes, selection rejects.
        pending_click["point"] = (float(x) / max(1.0, float(args.display_width)), float(y) / max(1.0, float(args.display_height)))

    try:
        if not args.no_display:
            cv2.namedWindow("Target-Lock CCTV Collection V2", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Target-Lock CCTV Collection V2", on_mouse)
        with telemetry_path.open("w", encoding="utf-8") as telemetry_handle:
            while (args.max_frames is None or frame_index < args.max_frames) and (args.max_seconds is None or time.perf_counter() - started < args.max_seconds):
                read_result = latest.read()
                if not read_result.ok or read_result.frame is None:
                    _write_telemetry(telemetry_handle, {
                        "event_type": "SOURCE_ONLY",
                        "timestamp_seconds": float(read_result.timestamp_seconds),
                        "frame_index": frame_index,
                        "source_health": _safe_source_health(read_result.health),
                        "target_lock": latest_lock.to_dict(),
                        "training_data_written": False,
                        "raw_video_recorded": False,
                    })
                    if read_result.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                        close_reason = read_result.reason or read_result.state
                        break
                    continue

                source_frame = read_result.frame
                inference_frame, inference_scale = resize_for_inference(
                    source_frame,
                    max_width=runtime_preset["capture_width"],
                    max_height=runtime_preset["capture_height"],
                )
                if source_origin is None:
                    source_origin = float(read_result.timestamp_seconds)
                session_timestamp = max(0.0, float(read_result.timestamp_seconds) - source_origin)
                pose_timestamp = session_timestamp if last_pose_timestamp is None else max(session_timestamp, last_pose_timestamp + 0.001)
                last_pose_timestamp = pose_timestamp
                processed = should_process_frame(frame_index, runtime_preset["process_every"], has_previous_result=last_result is not None)
                frame_inference_ms: float | None = None
                crop_result: dict[str, Any] = {"status": "NOT_RUN"}
                diagnostics: dict[str, Any] = {}

                if processed:
                    inference_started = time.perf_counter()
                    current_people = yolo_tracker.track(inference_frame) if yolo_tracker is not None else []
                    detections = backend.detect(inference_frame, timestamp_ms=int(round(pose_timestamp * 1000.0)))
                    last_result = runtime.process_frame(detections, pose_timestamp, camera_id=runtime.config.camera_id)
                    active_subjects = _active_subjects(last_result)
                    current_candidates, current_people_by_key, current_subjects_by_key, diagnostics = _build_target_candidates(
                        active_subjects,
                        current_people,
                        frame_width=int(inference_frame.shape[1]),
                        frame_height=int(inference_frame.shape[0]),
                        person_backend=args.person_backend,
                    )
                    latest_lock = lock_controller.update(current_candidates, timestamp_seconds=pose_timestamp)
                    selected_subject = current_subjects_by_key.get(latest_lock.identity_key or "")
                    prediction = None if selected_subject is None else _v4_prediction(
                        runner,
                        selected_subject,
                        visibility_threshold=posture_config.minimum_pose_visibility,
                    )
                    if args.collection_mode == "crop-experimental" and selected_subject is not None:
                        crop_result = _crop_comparison(
                            crop_backend,
                            runner,
                            inference_frame,
                            current_people_by_key.get(latest_lock.identity_key or ""),
                            timestamp_ms=int(round(pose_timestamp * 1000.0)),
                            visibility_threshold=posture_config.minimum_pose_visibility,
                            full_prediction=prediction,
                            padding_fraction=args.crop_padding,
                        )
                    frame_inference_ms = (time.perf_counter() - inference_started) * 1000.0
                    processed_frames += 1

                    quality = evaluate_collection_quality(
                        pose_detected=selected_subject is not None and selected_subject.world_landmarks is not None,
                        core_visibility={} if selected_subject is None else selected_subject.core_visibility,
                        feature_values=None if prediction is None else prediction.get("feature_values"),
                        feature_names=None if prediction is None else prediction.get("feature_names"),
                        frame_width=int(source_frame.shape[1]),
                        frame_height=int(source_frame.shape[0]),
                        min_visibility=posture_config.minimum_pose_visibility,
                    )
                    collection_gate = target_lock_collection_gate(
                        latest_lock,
                        human_label=selected_label,
                        label_source="HUMAN",
                        quality_accepted=bool(quality["accepted"]),
                    )
                    eligible_interval = last_sample_timestamp is None or pose_timestamp - last_sample_timestamp >= args.sample_interval_seconds
                    should_write = bool(capture_requested and collection_gate["accepted"] and eligible_interval and not args.no_display)
                    sample_written = False
                    if should_write and selected_subject is not None and prediction is not None:
                        selected_label = taxonomy.validate_human_label(selected_label or "")
                        canonical = canonical_cctv_diagnostic(selected_subject.world_landmarks)
                        sample_id = validate_safe_id(f"{session_id}_{frame_index:07d}", field_name="sample_id")
                        writer.append_sample(
                            sample_id=sample_id,
                            session_id=session_id,
                            subject_id=participant_id,
                            clip_id=clip_id,
                            human_label=selected_label,
                            timestamp_seconds=pose_timestamp,
                            frame_index=frame_index,
                            source_type=source.source_type,
                            camera_id=source.camera_id,
                            camera_profile=source.spec.profile,
                            frame_width=int(source_frame.shape[1]),
                            frame_height=int(source_frame.shape[0]),
                            core_visibility=selected_subject.core_visibility,
                            quality_flag=quality["quality_flag"],
                            quality_warnings=quality["warnings"],
                            v4_prediction=prediction.get("label"),
                            v4_confidence=prediction.get("confidence"),
                            v4_probabilities=prediction.get("probabilities", {}),
                            geometry=selected_subject.geometry,
                            calibration_status="MULTI_PERSON_RUNTIME_DIAGNOSTIC_ONLY",
                            feature_values=prediction["feature_values"],
                            frame_age_ms=read_result.frame_age_ms,
                            inference_time_ms=frame_inference_ms,
                            source_timestamp_seconds=read_result.timestamp_seconds,
                            mediapipe_timestamp_seconds=pose_timestamp,
                            extra_values={
                                "source_label": selected_label,
                                "canonical_label": selected_label,
                                "label_source": "HUMAN",
                                "target_lock_state": latest_lock.state,
                                "target_identity_mode": latest_lock.identity_mode,
                                "target_yolo_track_id": latest_lock.target_yolo_track_id,
                                "target_runtime_subject_id": latest_lock.runtime_subject_id,
                                "target_bbox_json": compact_json(latest_lock.target_bbox or []),
                                "target_association_state": latest_lock.association_state,
                                "target_association_score": latest_lock.association_score,
                                "target_pause_reason": latest_lock.pause_reason,
                                "yolo_enabled": yolo_tracker is not None,
                                "yolo_model": Path(args.yolo_weights).name if yolo_tracker is not None else "NOT_ENABLED",
                                "yolo_tracker": args.yolo_tracker if yolo_tracker is not None else "NOT_ENABLED",
                                "yolo_detection_count": len(current_people),
                                "yolo_tracked_count": sum(box.track_id is not None for box in current_people),
                                "association_strategy": diagnostics.get("association_strategy"),
                                "collection_mode": args.collection_mode,
                                "inference_frame_width": int(inference_frame.shape[1]),
                                "inference_frame_height": int(inference_frame.shape[0]),
                                "inference_scale": inference_scale,
                                "world_landmarks_json": compact_json(serialise_landmarks(selected_subject.world_landmarks)),
                                "cctv_canonical_status": canonical["status"],
                                "cctv_canonical_features_json": compact_json(canonical["features"]),
                                "cctv_canonical_metadata_json": compact_json(canonical["metadata"]),
                            },
                        )
                        last_sample_timestamp = pose_timestamp
                        accepted_samples += 1
                        label_counts[selected_label] += 1
                        sample_written = True
                    elif capture_requested and not collection_gate["accepted"]:
                        # Record a rate-limited rejection explanation, but keep the
                        # continuous stream telemetry separate from training rows.
                        if last_rejection_timestamp is None or pose_timestamp - last_rejection_timestamp >= args.sample_interval_seconds:
                            reasons = list(collection_gate["reasons"]) + list(quality["reasons"])
                            for reason in reasons:
                                rejection_counts[str(reason).split(":", 1)[0]] += 1
                            writer.append_rejection({
                                "schema_version": TARGET_LOCK_COLLECTION_SCHEMA_VERSION,
                                "session_id": session_id,
                                "subject_id": participant_id,
                                "clip_id": clip_id,
                                "target_lock": latest_lock.to_dict(),
                                "human_label_selected": selected_label,
                                "label_source": "HUMAN",
                                "reasons": reasons,
                                "raw_video_recorded": False,
                                "training_data_written": False,
                            })
                            last_rejection_timestamp = pose_timestamp

                    telemetry_event = frame_to_event(last_result, frame_index=frame_index, source_metadata=read_result.health)
                    telemetry_event.update({
                        "event_type": "RUNTIME_TELEMETRY",
                        "target_lock": latest_lock.to_dict(),
                        "collection_armed": capture_requested,
                        "human_label_selected": selected_label,
                        "human_label_is_training_ground_truth": False,
                        "training_data_written": sample_written,
                        "yolo": {
                            "enabled": yolo_tracker is not None,
                            "model": Path(args.yolo_weights).name if yolo_tracker is not None else "NOT_ENABLED",
                            "tracker": args.yolo_tracker if yolo_tracker is not None else "NOT_ENABLED",
                            "person_detection_count": len(current_people),
                            "tracked_person_count": sum(box.track_id is not None for box in current_people),
                            **diagnostics,
                        },
                        "crop_mediapipe_experiment": crop_result,
                        "frame_age_ms": read_result.frame_age_ms,
                        "inference_time_ms": frame_inference_ms,
                        "raw_video_recorded": False,
                    })
                    _write_telemetry(telemetry_handle, telemetry_event)
                else:
                    _write_telemetry(telemetry_handle, {
                        "event_type": "RUNTIME_TELEMETRY_SKIPPED",
                        "timestamp_seconds": pose_timestamp,
                        "frame_index": frame_index,
                        "source_health": _safe_source_health(read_result.health),
                        "target_lock": latest_lock.to_dict(),
                        "inference_skipped": True,
                        "training_data_written": False,
                        "raw_video_recorded": False,
                    })

                if not args.no_display and last_result is not None:
                    display_frame, _ = resize_for_display(inference_frame, max_width=args.display_width, max_height=args.display_height)
                    # Set click coordinates against the actual rendered size.
                    args.display_width, args.display_height = int(display_frame.shape[1]), int(display_frame.shape[0])
                    annotated = draw_multiperson_overlay(display_frame, last_result, debug=True)
                    annotated = _draw_yolo_and_lock(
                        annotated,
                        current_people,
                        current_candidates,
                        latest_lock,
                        person_frame_width=int(inference_frame.shape[1]),
                        person_frame_height=int(inference_frame.shape[0]),
                        capture_requested=capture_requested,
                        selected_label=selected_label,
                        labels_help=labels_help,
                    )
                    cv2.imshow("Target-Lock CCTV Collection V2", annotated)
                    if pending_click["point"] is not None:
                        point = pending_click.pop("point")
                        try:
                            latest_lock = lock_controller.select_at(current_candidates, point[0], point[1], timestamp_seconds=pose_timestamp)
                            capture_requested = False
                        except ValueError:
                            # The UI shows the lock as paused; no hidden fallback or
                            # alternate person is ever selected here.
                            pass
                    key_code = cv2.waitKey(1) & 0xFF
                    key = chr(key_code).upper() if key_code else ""
                    if key == "Q" or key_code == 27:
                        close_reason = "USER_EXIT"
                        break
                    if key in label_key_map:
                        selected_label = label_key_map[key]
                    elif key == "0":
                        selected_label = None
                        capture_requested = False
                    elif key_code in {9} or key == "N":
                        selectable = [candidate for candidate in current_candidates if candidate.is_unambiguous]
                        if selectable:
                            candidate_cycle_index = (candidate_cycle_index + 1) % len(selectable)
                            latest_lock = lock_controller.select(selectable[candidate_cycle_index], timestamp_seconds=pose_timestamp)
                            capture_requested = False
                    elif key == "C":
                        latest_lock = lock_controller.clear()
                        capture_requested = False
                    elif key_code == ord(" "):
                        capture_requested = not capture_requested
                frame_index += 1
    except KeyboardInterrupt:
        close_reason = "KEYBOARD_INTERRUPT"
    finally:
        try:
            backend.close()
        finally:
            if crop_backend is not None:
                crop_backend.close()
            final_health = latest.health()
            latest.close()
            if not args.no_display:
                cv2.destroyAllWindows()

    metadata = {
        "schema_version": TARGET_LOCK_COLLECTION_SCHEMA_VERSION,
        "session_id": session_id,
        "subject_id": participant_id,
        "clip_id": clip_id,
        "operator_code": str(args.operator_code)[:80],
        "taxonomy_schema_version": taxonomy.schema_version,
        "allowed_human_labels": list(taxonomy.collection_labels),
        "source_type": source.source_type,
        "camera_id": source.camera_id,
        "camera_profile": source.spec.profile,
        "source_display": source.spec.source_display,
        "preset": runtime_preset["name"],
        "max_people": args.max_people,
        "person_backend": args.person_backend,
        "yolo_weights_file": Path(args.yolo_weights).name if yolo_tracker is not None else None,
        "yolo_tracker": args.yolo_tracker if yolo_tracker is not None else None,
        "collection_mode": args.collection_mode,
        "frames_seen": frame_index,
        "frames_processed": processed_frames,
        "accepted_samples_this_session": accepted_samples,
        "label_counts_this_session": dict(label_counts),
        "rejection_counts": dict(rejection_counts),
        "last_target_lock": latest_lock.to_dict(),
        "last_source_health": _safe_source_health(final_health),
        "telemetry_path": str(telemetry_path),
        "raw_video_recorded": False,
        "screenshots_recorded": False,
        "human_labels_are_manual": True,
        "label_source_required": "HUMAN",
        "v4_predictions_are_diagnostic_only": True,
        "geometry_is_diagnostic_only": True,
        "training_performed": False,
        "close_reason": close_reason,
        "target_reassignment_policy": "NO_SILENT_REASSIGNMENT",
    }
    metadata_path = writer.write_session_metadata(metadata)
    print(json.dumps({
        "status": "TARGET_LOCK_COLLECTION_SESSION_CLOSED",
        "session_id": session_id,
        "accepted_samples_this_session": accepted_samples,
        "samples_csv": str(writer.samples_path),
        "runtime_telemetry": str(telemetry_path),
        "session_metadata": str(metadata_path),
        "raw_video_recorded": False,
        "training_performed": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
