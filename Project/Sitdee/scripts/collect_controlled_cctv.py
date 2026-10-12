"""Collect human-labelled multi-person CCTV samples without recording raw video.

The operator selects one anonymous PERSON_XX track and explicitly assigns a
posture label.  The frozen V4 prediction is stored as diagnostic context only;
it is never used as ground truth.  This command is prepared for a later,
attended camera session and is intentionally not run during automated tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.data.cctv_collection import (  # noqa: E402
    CCTV_COLLECTION_SCHEMA_VERSION,
    CCTV_EXTRA_SAMPLE_COLUMNS,
    CCTV_MANUAL_LABELS,
    ManualPersonAssignment,
    label_for_key,
)
from src.posture_ai.data.controlled_collection import (  # noqa: E402
    ControlledSampleWriter,
    evaluate_collection_quality,
    make_session_id,
    validate_safe_id,
    validate_subject_id,
)
from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig  # noqa: E402
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput  # noqa: E402
from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner  # noqa: E402
from src.posture_ai.inference.stream_runtime import resize_for_display, resize_for_inference  # noqa: E402
from src.posture_ai.inference.video_source import SOURCE_CLOSED, SOURCE_FAILED, VideoSource  # noqa: E402
from src.posture_ai.multiperson.mediapipe_tasks import MultiPersonMediaPipeBackend  # noqa: E402
from src.posture_ai.multiperson.overlay import draw_multiperson_overlay  # noqa: E402
from src.posture_ai.multiperson.runtime import MultiPersonRuntime, MultiPersonRuntimeConfig, resolve_performance_preset, should_process_frame  # noqa: E402
from src.posture_ai.multiperson.telemetry import MultiPersonTelemetry  # noqa: E402


def _draw_status(
    frame,
    *,
    selected_person: str | None,
    selected_label: str | None,
    expected_people_count: int | None,
    detected_people_count: int,
    capturing: bool,
    help_text: str,
) -> object:
    output = frame.copy()
    height, width = output.shape[:2]
    panel_height = 92
    panel_y = max(0, height - panel_height)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, panel_y), (width, height), (25, 25, 55), -1)
    output[:] = cv2.addWeighted(overlay, 0.75, output, 0.25, 0.0)
    state = "CAPTURING" if capturing else "PAUSED"
    lines = (
        f"EXPECTED: {expected_people_count or 'NOT SET'} | DETECTED: {detected_people_count} | {state}",
        f"TARGET: {selected_person or 'NONE'} | HUMAN LABEL: {selected_label or 'NONE'}",
        "1=TUP  2=TLB  3=TLF  4=TLL  5=TLR  0=IGNORE  U=UNCERTAIN",
        f"TAB/N=next target  [/]=expected count  SPACE=start/pause  R=reset  Q/ESC=finish | {help_text}",
    )
    for offset, line in enumerate(lines):
        y = panel_y + 22 + offset * 24
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _subject_prediction(runner: RGBNativeV4Runner, subject, *, visibility_threshold: float) -> dict | None:
    if not subject.core_valid or subject.world_landmarks is None:
        return None
    try:
        return runner.predict_world_landmarks(
            subject.world_landmarks,
            image_visibility=subject.landmarks,
            visibility_threshold=visibility_threshold,
        )
    except (KeyError, TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Attended, human-labelled multi-person CCTV collection; raw video is off")
    parser.add_argument("--source", default="rtsp", help="rtsp/profile, video file, or explicit USB index")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default="HIKVISION_MAIN")
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--subject-id", required=True, help="Anonymous participant code, for example P001")
    parser.add_argument("--clip-id", required=True, help="Safe clip code, for example CCTV_STAGE1_01")
    parser.add_argument("--operator-code", default="ANONYMOUS_OPERATOR", help="Non-identifying operator code")
    parser.add_argument("--model", default="outputs/models/rgb_native_v4")
    parser.add_argument("--pose-model", default="metadata/pose_landmarker_heavy.task")
    parser.add_argument("--config", default="configs/posture_engine.yaml")
    parser.add_argument("--rtsp-config", default="configs/rtsp_profiles.yaml")
    parser.add_argument("--preset", choices=("RTSP_DESKTOP", "RTSP_BALANCED", "RTSP_LOW_LATENCY", "RTSP_CCTV"), default="RTSP_CCTV")
    parser.add_argument("--max-people", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--process-every", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--max-reconnects", type=int, default=5)
    parser.add_argument("--expected-people-count", type=int, choices=(1, 2, 3, 4), default=None, help="Operator ground truth for people physically present; can also be changed with [ and ]")
    parser.add_argument("--target-seat-id", default=None, help="Optional anonymous seat code such as SEAT_01")
    parser.add_argument("--target-position-id", default=None, help="Optional safe position code such as LEFT_NEAR")
    parser.add_argument("--distance-band", choices=("NEAR", "MID", "FAR", "UNKNOWN"), default="UNKNOWN")
    parser.add_argument("--output-root", default="data/controlled_cctv_v2")
    parser.add_argument("--telemetry-path", default=None)
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--display-height", type=int, default=720)
    parser.add_argument("--no-display", action="store_true", help="Run diagnostics only; manual capture cannot occur")
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    try:
        participant_id = validate_subject_id(args.subject_id)
        clip_id = validate_safe_id(args.clip_id, field_name="clip_id")
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_people < 1 or args.max_reconnects < 0:
        parser.error("max-people and max-reconnects must be non-negative/positive")
    if args.process_every is not None and args.process_every < 1:
        parser.error("--process-every must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")

    runtime_preset = resolve_performance_preset(args.preset, args.process_every)
    posture_config = PostureEngineConfig.from_yaml(args.config)
    runner = RGBNativeV4Runner(args.model)
    target_seat_id = None if args.target_seat_id is None else validate_safe_id(args.target_seat_id, field_name="target_seat_id")
    target_position_id = None if args.target_position_id is None else validate_safe_id(args.target_position_id, field_name="target_position_id")
    writer = ControlledSampleWriter(
        args.output_root,
        feature_names=runner.expected_names,
        allowed_labels=CCTV_MANUAL_LABELS,
        schema_version=CCTV_COLLECTION_SCHEMA_VERSION,
        extra_columns=CCTV_EXTRA_SAMPLE_COLUMNS,
    )
    session_id = make_session_id(participant_id, clip_id)
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
    runtime = MultiPersonRuntime(
        runner,
        runtime_config=MultiPersonRuntimeConfig(
            max_people=args.max_people,
            visibility_threshold=posture_config.minimum_pose_visibility,
            camera_id=args.camera_id or source.camera_id or "CAM01",
        ),
        posture_config=posture_config,
    )
    telemetry = MultiPersonTelemetry(max_events=2000)
    telemetry_path = Path(args.telemetry_path) if args.telemetry_path else Path(args.output_root) / "sessions" / f"{session_id}.jsonl"
    selected_person: str | None = None
    selected_label: str | None = None
    capturing = False
    frame_index = 0
    processed_frames = 0
    accepted_samples = 0
    label_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    seen_people: set[str] = set()
    started = time.perf_counter()
    source_origin: float | None = None
    last_pose_timestamp: float | None = None
    last_result = None
    close_reason = "USER_EXIT"
    expected_people_count = args.expected_people_count
    collection_events: list[dict] = []
    try:
        if not args.no_display:
            cv2.namedWindow("Controlled CCTV Collection", cv2.WINDOW_NORMAL)
        while (args.max_frames is None or frame_index < args.max_frames) and (args.max_seconds is None or time.perf_counter() - started < args.max_seconds):
            read_result = latest.read()
            if not read_result.ok or read_result.frame is None:
                if read_result.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                    close_reason = read_result.reason or read_result.state
                    break
                continue
            frame = read_result.frame
            inference_frame, _scale = resize_for_inference(
                frame,
                max_width=runtime_preset["capture_width"],
                max_height=runtime_preset["capture_height"],
            )
            if source_origin is None:
                source_origin = float(read_result.timestamp_seconds)
            session_timestamp = max(0.0, float(read_result.timestamp_seconds) - source_origin)
            if last_pose_timestamp is None:
                pose_timestamp = session_timestamp
            else:
                pose_timestamp = max(session_timestamp, last_pose_timestamp + 0.001)
            last_pose_timestamp = pose_timestamp
            if should_process_frame(frame_index, runtime_preset["process_every"], has_previous_result=last_result is not None):
                detections = backend.detect(inference_frame, timestamp_ms=int(round(pose_timestamp * 1000.0)))
                last_result = runtime.process_frame(detections, pose_timestamp, camera_id=runtime.config.camera_id)
                telemetry.add_frame(last_result, frame_index=frame_index, source_metadata=read_result.health)
                processed_frames += 1
                for subject in last_result.subjects:
                    seen_people.add(subject.subject_id)
                subject_by_id = {subject.subject_id: subject for subject in last_result.subjects}
                ignored_subjects = [subject.subject_id for subject in last_result.subjects if subject.subject_id != selected_person]
                event = last_result.to_dict()
                event.update({
                    "schema_version": CCTV_COLLECTION_SCHEMA_VERSION,
                    "frame_index": frame_index,
                    "expected_people_count": expected_people_count,
                    "target_subject_id": selected_person,
                    "human_posture_label": selected_label,
                    "human_label_source": "MANUAL_OPERATOR_ONLY" if selected_label in CCTV_MANUAL_LABELS else "NOT_ASSIGNED",
                    "ignored_for_posture_training": ignored_subjects,
                    "v4_prediction_is_not_ground_truth": True,
                    "raw_video_recorded": False,
                })
                event["subjects"] = [
                    dict(subject, pose_index=index, track_id=subject.get("subject_id"))
                    for index, subject in enumerate(event.get("subjects", []))
                ]
                collection_events.append(event)
                if expected_people_count is not None and selected_person is not None and capturing and selected_label in CCTV_MANUAL_LABELS:
                    subject = subject_by_id.get(selected_person)
                    prediction = None if subject is None else _subject_prediction(
                        runner,
                        subject,
                        visibility_threshold=posture_config.minimum_pose_visibility,
                    )
                    if subject is None:
                        rejection_counts["SELECTED_PERSON_NOT_VISIBLE"] += 1
                    else:
                        feature_values = None if prediction is None else prediction.get("feature_values")
                        feature_names = None if prediction is None else prediction.get("feature_names")
                        gate = evaluate_collection_quality(
                            pose_detected=subject.tracking_state in {"NEW", "ACTIVE"},
                            core_visibility=subject.core_visibility,
                            feature_values=feature_values,
                            feature_names=feature_names,
                            frame_width=int(inference_frame.shape[1]),
                            frame_height=int(inference_frame.shape[0]),
                            min_visibility=posture_config.minimum_pose_visibility,
                        )
                        if gate["accepted"] and prediction is not None:
                            assignment = ManualPersonAssignment(selected_person, selected_label, args.operator_code)
                            sample_id = validate_safe_id(f"{session_id}_{selected_person}_{frame_index:06d}", field_name="sample_id")
                            pose_index = next((index for index, item in enumerate(last_result.subjects) if item.subject_id == selected_person), -1)
                            writer.append_sample(
                                sample_id=sample_id,
                                session_id=session_id,
                                subject_id=participant_id,
                                clip_id=clip_id,
                                human_label=assignment.label,
                                timestamp_seconds=pose_timestamp,
                                frame_index=frame_index,
                                source_type=source.source_type,
                                camera_id=source.camera_id,
                                camera_profile=source.spec.profile,
                                frame_width=int(inference_frame.shape[1]),
                                frame_height=int(inference_frame.shape[0]),
                                core_visibility=subject.core_visibility,
                                quality_flag=gate["quality_flag"],
                                quality_warnings=gate["warnings"],
                                v4_prediction=prediction.get("label"),
                                v4_confidence=prediction.get("confidence"),
                                v4_probabilities=prediction.get("probabilities", {}),
                                geometry=subject.geometry,
                                calibration_status="MULTI_PERSON_RUNTIME_GEOMETRY_ONLY",
                                feature_values=prediction["feature_values"],
                                frame_age_ms=read_result.frame_age_ms,
                                source_timestamp_seconds=read_result.timestamp_seconds,
                                mediapipe_timestamp_seconds=pose_timestamp,
                                extra_values={
                                    "expected_people_count": expected_people_count,
                                    "target_subject_id": assignment.person_id,
                                    "target_seat_id": target_seat_id or "",
                                    "target_position_id": target_position_id or "",
                                    "distance_band": args.distance_band,
                                    "pose_index": pose_index,
                                    "track_id": assignment.person_id,
                                    "nose_visibility": subject.core_visibility.get("nose"),
                                    "left_shoulder_visibility": subject.core_visibility.get("left_shoulder"),
                                    "right_shoulder_visibility": subject.core_visibility.get("right_shoulder"),
                                    "left_hip_visibility": subject.core_visibility.get("left_hip"),
                                    "right_hip_visibility": subject.core_visibility.get("right_hip"),
                                    "human_posture_label": assignment.label,
                                    "label_source": "MANUAL_OPERATOR_ONLY",
                                },
                            )
                            accepted_samples += 1
                            label_counts[assignment.label] += 1
                        else:
                            for reason in gate["reasons"]:
                                rejection_counts[reason.split(":", 1)[0]] += 1
                            writer.append_rejection({
                                "schema_version": CCTV_COLLECTION_SCHEMA_VERSION,
                                "session_id": session_id,
                                "participant_code": participant_id,
                                "clip_id": clip_id,
                                "person_id": selected_person,
                                "human_label_selected": selected_label,
                                "reasons": gate["reasons"],
                                "model_prediction_is_not_ground_truth": True,
                                "raw_video_recorded": False,
                            })
            else:
                if last_result is not None:
                    telemetry.add_frame(last_result, frame_index=frame_index, inference_skipped=True, source_metadata=read_result.health)

            if not args.no_display and last_result is not None:
                display_frame, _display_scale = resize_for_display(frame, max_width=args.display_width, max_height=args.display_height)
                annotated = draw_multiperson_overlay(display_frame, last_result)
                annotated = _draw_status(
                    annotated,
                    selected_person=selected_person,
                    selected_label=selected_label,
                    expected_people_count=expected_people_count,
                    detected_people_count=len(last_result.subjects),
                    capturing=capturing,
                    help_text="manual labels only; no raw video",
                )
                cv2.imshow("Controlled CCTV Collection", annotated)
                key_code = cv2.waitKey(1) & 0xFF
                key = chr(key_code).upper() if key_code else ""
                if key in {"Q"} or key_code == 27:
                    close_reason = "USER_EXIT"
                    break
                if key_code in {9} or key in {"N"}:
                    ids = [subject.subject_id for subject in last_result.subjects]
                    if ids:
                        selected_person = ids[0] if selected_person not in ids else ids[(ids.index(selected_person) + 1) % len(ids)]
                mapped = label_for_key(key)
                if mapped in CCTV_MANUAL_LABELS:
                    selected_label = mapped
                elif mapped in {"IGNORE", "UNCERTAIN"}:
                    capturing = False
                    selected_label = None
                elif key_code == ord(" "):
                    capturing = bool(expected_people_count and selected_person and selected_label in CCTV_MANUAL_LABELS) and not capturing
                elif key == "[":
                    expected_people_count = max(1, (expected_people_count or 1) - 1)
                elif key == "]":
                    expected_people_count = min(4, (expected_people_count or 0) + 1)
                elif key in {"R"}:
                    capturing = False
                    selected_label = None
            frame_index += 1
    except KeyboardInterrupt:
        close_reason = "KEYBOARD_INTERRUPT"
    finally:
        backend.close()
        final_health = latest.health()
        latest.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    telemetry.write_jsonl(telemetry_path)
    collection_events_path = Path(args.output_root) / "sessions" / f"{session_id}_collection_events.jsonl"
    collection_events_path.parent.mkdir(parents=True, exist_ok=True)
    with collection_events_path.open("w", encoding="utf-8") as handle:
        for event in collection_events:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    metadata = {
        "schema_version": CCTV_COLLECTION_SCHEMA_VERSION,
        "session_id": session_id,
        "participant_code": participant_id,
        "clip_id": clip_id,
        "operator_code": str(args.operator_code)[:80],
        "expected_people_count_at_close": expected_people_count,
        "target_seat_id": target_seat_id,
        "target_position_id": target_position_id,
        "distance_band": args.distance_band,
        "source_type": source.source_type,
        "source_display": source.spec.source_display,
        "camera_id": source.camera_id,
        "camera_profile": source.spec.profile,
        "preset": runtime_preset["name"],
        "max_people": args.max_people,
        "frames_seen": frame_index,
        "frames_processed": processed_frames,
        "accepted_samples_this_session": accepted_samples,
        "label_counts_this_session": dict(label_counts),
        "rejection_counts": dict(rejection_counts),
        "runtime_person_ids_seen": sorted(seen_people),
        "last_source_health": final_health,
        "close_reason": close_reason,
        "manual_label_options": list(CCTV_MANUAL_LABELS),
        "human_labels_are_manual": True,
        "v4_predictions_are_not_ground_truth": True,
        "biometric_identity_used": False,
        "raw_video_recorded": False,
        "training_performed": False,
        "telemetry_path": str(telemetry_path),
        "collection_events_path": str(collection_events_path),
        "ignored_visible_tracks_are_not_posture_training_labels": True,
    }
    metadata_path = writer.write_session_metadata(metadata)
    print(json.dumps({
        "status": "CONTROLLED_CCTV_COLLECTION_SESSION_CLOSED",
        "session_id": session_id,
        "accepted_samples_this_session": accepted_samples,
        "metadata": str(metadata_path),
        "samples_csv": str(writer.samples_path),
        "telemetry": str(telemetry_path),
        "raw_video_recorded": False,
        "training_performed": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
