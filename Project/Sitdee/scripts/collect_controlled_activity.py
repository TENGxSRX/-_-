"""Attended human-labelled sit/stand collection without raw video.

Labels are selected by the operator with S (SITTING) or D (STANDING).  The
experimental activity gate and V4 are diagnostic only and never become the
target label.  The command is prepared for a future attended session and is
not started automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.data.activity_collection import ACTIVITY_COLLECTION_SCHEMA_VERSION, ACTIVITY_EXTRA_SAMPLE_COLUMNS, ACTIVITY_LABELS
from src.posture_ai.data.controlled_collection import ControlledSampleWriter, evaluate_collection_quality, make_session_id, validate_safe_id, validate_subject_id
from src.posture_ai.features.canonical_cctv import CANONICAL_CCTV_SCHEMA_VERSION, build_canonical_cctv_features
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput
from src.posture_ai.inference.stream_runtime import resize_for_display, resize_for_inference
from src.posture_ai.inference.video_source import SOURCE_CLOSED, SOURCE_FAILED, VideoSource
from src.posture_ai.multiperson.mediapipe_tasks import MultiPersonMediaPipeBackend
from src.posture_ai.multiperson.overlay import draw_multiperson_overlay
from src.posture_ai.multiperson.schema import MultiPersonFrameResult
from src.posture_ai.multiperson.telemetry import MultiPersonTelemetry
from src.posture_ai.multiperson.tracking import AnonymousPoseTracker
from src.posture_ai.ergonomics.geometry import calculate_geometry


def _status(frame, *, selected: str | None, label: str | None, expected: int | None, detected: int, collecting: bool):
    output = frame.copy()
    height = output.shape[0]
    lines = (
        f"EXPECTED: {expected or 'NOT SET'} | DETECTED: {detected} | {'COLLECTING' if collecting else 'PAUSED'}",
        f"TARGET: {selected or 'NONE'} | HUMAN ACTIVITY: {label or 'NONE'}",
        "S=SITTING  D=STANDING  0=clear  TAB/N=target  SPACE=start/pause  [/]=expected  Q=finish",
        "NO RAW VIDEO | gate/V4 are diagnostic only | labels are human-confirmed",
    )
    for index, line in enumerate(lines):
        y = height - 82 + index * 20
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Human-labelled sit/stand collection; raw video is disabled")
    parser.add_argument("--source", default="rtsp")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default="HIKVISION_MAIN")
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--pose-model", default="metadata/pose_landmarker_heavy.task")
    parser.add_argument("--rtsp-config", default="configs/rtsp_profiles.yaml")
    parser.add_argument("--max-people", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--expected-people-count", type=int, choices=(1, 2, 3, 4), default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--output-root", default="data/controlled_activity_v1")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    participant = validate_subject_id(args.subject_id)
    clip_id = validate_safe_id(args.clip_id, field_name="clip_id")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    session_id = make_session_id(participant, clip_id)
    feature_names = tuple(f"canonical_{name}" for name in build_canonical_cctv_features(
        np.zeros((33, 3)), coordinate_type="mediapipe_world", scale_mode="none"
    ).names)
    writer = ControlledSampleWriter(
        args.output_root,
        feature_names=feature_names,
        allowed_labels=ACTIVITY_LABELS,
        schema_version=ACTIVITY_COLLECTION_SCHEMA_VERSION,
        extra_columns=ACTIVITY_EXTRA_SAMPLE_COLUMNS,
    )
    source = VideoSource(args.source, camera_id=args.camera_id, profile=args.profile, config_path=args.rtsp_config, reconnect=True, low_latency=True)
    latest = LatestFrameInput(source, asynchronous=source.source_type == "RTSP")
    if not latest.open():
        print("Could not open explicit source", file=sys.stderr)
        return 2
    backend = MultiPersonMediaPipeBackend(args.pose_model, max_people=args.max_people, running_mode="video")
    tracker = AnonymousPoseTracker(max_people=args.max_people)
    telemetry = MultiPersonTelemetry(max_events=2000)
    selected: str | None = None
    label: str | None = None
    collecting = False
    expected = args.expected_people_count
    frame_index = 0
    accepted = 0
    started = time.perf_counter()
    origin: float | None = None
    events: list[dict] = []
    try:
        if not args.no_display:
            cv2.namedWindow("Controlled Activity Collection", cv2.WINDOW_NORMAL)
        while args.max_frames is None or frame_index < args.max_frames:
            if args.max_seconds is not None and time.perf_counter() - started >= args.max_seconds:
                break
            read = latest.read()
            if not read.ok or read.frame is None:
                if read.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                    break
                continue
            if origin is None:
                origin = float(read.timestamp_seconds)
            timestamp = max(0.0, float(read.timestamp_seconds) - origin)
            inference, _ = resize_for_inference(read.frame, max_width=960, max_height=540)
            detections = backend.detect(inference, timestamp_ms=int(round(timestamp * 1000.0)))
            subjects = tracker.update(detections, timestamp)
            result = MultiPersonFrameResult(timestamp, subjects=subjects, camera_id=args.camera_id or source.camera_id or "CAM01", tracking_status="CONTROLLED_ACTIVITY_COLLECTION", events=tuple(tracker.last_events))
            telemetry.add_frame(result, frame_index=frame_index, source_metadata=read.health)
            if selected is None and subjects:
                selected = subjects[0].subject_id
            by_id = {subject.subject_id: subject for subject in subjects}
            if collecting and label and expected and selected:
                subject = by_id.get(selected)
                if subject is not None and subject.core_valid and subject.world_landmarks is not None:
                    try:
                        canonical = build_canonical_cctv_features(subject.world_landmarks, coordinate_type="mediapipe_world", scale_mode="none")
                        values = canonical.values.reshape(-1)
                        visibility = subject.core_visibility
                        gate = evaluate_collection_quality(
                            pose_detected=True,
                            core_visibility=visibility,
                            feature_values=values,
                            feature_names=feature_names,
                            frame_width=int(inference.shape[1]),
                            frame_height=int(inference.shape[0]),
                        )
                        if gate["accepted"]:
                            pose_index = next((index for index, item in enumerate(subjects) if item.subject_id == selected), -1)
                            writer.append_sample(
                                sample_id=validate_safe_id(f"{session_id}_{selected}_{frame_index:06d}", field_name="sample_id"),
                                session_id=session_id,
                                subject_id=participant,
                                clip_id=clip_id,
                                human_label=label,
                                timestamp_seconds=timestamp,
                                frame_index=frame_index,
                                source_type=source.source_type,
                                camera_id=source.camera_id,
                                camera_profile=source.spec.profile,
                                frame_width=int(inference.shape[1]),
                                frame_height=int(inference.shape[0]),
                                core_visibility=visibility,
                                quality_flag=gate["quality_flag"],
                                quality_warnings=gate["warnings"],
                                v4_prediction=None,
                                v4_confidence=None,
                                v4_probabilities={},
                                geometry=calculate_geometry(subject.world_landmarks),
                                calibration_status="ACTIVITY_LABELLED_NO_V4_TARGET",
                                feature_values=values,
                                frame_age_ms=read.frame_age_ms,
                                source_timestamp_seconds=read.timestamp_seconds,
                                mediapipe_timestamp_seconds=timestamp,
                                extra_values={
                                    "human_activity_label": label,
                                    "label_source": "MANUAL_OPERATOR_ONLY",
                                    "target_subject_id": selected,
                                    "track_id": selected,
                                    "pose_index": pose_index,
                                    "expected_people_count": expected,
                                    "activity_gate_state": "NOT_USED_FOR_LABEL",
                                    "activity_gate_method": "GEOMETRY_EXPERIMENTAL_DIAGNOSTIC_ONLY",
                                    "canonical_schema_version": CANONICAL_CCTV_SCHEMA_VERSION,
                                    "canonical_coordinate_type": "mediapipe_world",
                                    "canonical_scale_mode": "none",
                                },
                            )
                            accepted += 1
                    except (TypeError, ValueError):
                        pass
            if not args.no_display:
                display, _ = resize_for_display(read.frame, max_width=1280, max_height=720)
                display = draw_multiperson_overlay(display, result)
                display = _status(display, selected=selected, label=label, expected=expected, detected=len(subjects), collecting=collecting)
                cv2.imshow("Controlled Activity Collection", display)
                key_code = cv2.waitKey(1) & 0xFF
                key = chr(key_code).upper() if key_code else ""
                if key in {"Q"} or key_code == 27:
                    break
                if key in {"S", "D"}:
                    label = "SITTING" if key == "S" else "STANDING"
                elif key_code in {ord("0"), ord("U")}:
                    label = None
                    collecting = False
                elif key_code == ord(" "):
                    collecting = bool(expected and selected and label in ACTIVITY_LABELS) and not collecting
                elif key_code == 9 or key == "N":
                    ids = [subject.subject_id for subject in subjects]
                    if ids:
                        selected = ids[0] if selected not in ids else ids[(ids.index(selected) + 1) % len(ids)]
                elif key == "[":
                    expected = max(1, (expected or 1) - 1)
                elif key == "]":
                    expected = min(4, (expected or 0) + 1)
            frame_index += 1
    finally:
        backend.close()
        latest.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    telemetry_path = Path(args.output_root) / "sessions" / f"{session_id}.jsonl"
    telemetry.write_jsonl(telemetry_path)
    metadata = {
        "schema_version": ACTIVITY_COLLECTION_SCHEMA_VERSION,
        "session_id": session_id,
        "participant_code": participant,
        "clip_id": clip_id,
        "accepted_samples_this_session": accepted,
        "frames_seen": frame_index,
        "expected_people_count_at_close": expected,
        "source_type": source.source_type,
        "camera_id": source.camera_id,
        "camera_profile": source.spec.profile,
        "label_source": "MANUAL_OPERATOR_ONLY",
        "prediction_used_as_label": False,
        "raw_video_recorded": False,
        "training_performed": False,
        "telemetry_path": str(telemetry_path),
        "events": events,
    }
    metadata_path = writer.write_session_metadata(metadata)
    print(json.dumps({"status": "CONTROLLED_ACTIVITY_COLLECTION_SESSION_CLOSED", "accepted_samples": accepted, "metadata": str(metadata_path), "samples_csv": str(writer.samples_path), "raw_video_recorded": False, "training_performed": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
