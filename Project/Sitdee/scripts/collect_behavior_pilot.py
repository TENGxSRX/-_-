"""One-pass, attended activity/posture collection for partial CCTV poses.

The operator provides every ground-truth label.  No raw video is saved and
V4 is stored only as an optional diagnostic for later evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.data.behavior_collection import (  # noqa: E402
    ACTIVITY_LABELS,
    BEHAVIOR_COLLECTION_SCHEMA_VERSION,
    BEHAVIOR_EXTRA_SAMPLE_COLUMNS,
    POSTURE_NONE,
    validate_behavior_labels,
)
from src.posture_ai.data.controlled_collection import ControlledSampleWriter, make_session_id, validate_safe_id, validate_subject_id  # noqa: E402
from src.posture_ai.ergonomics.geometry import calculate_geometry  # noqa: E402
from src.posture_ai.features.activity_partial_pose_v1 import (  # noqa: E402
    ACTIVITY_PARTIAL_POSE_V1_SCHEMA_VERSION,
    FULLER_AVAILABLE_POSE,
    LOWER_LANDMARKS,
    activity_feature_names,
    build_activity_partial_pose_features,
    partial_pose_collection_quality,
)
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput  # noqa: E402
from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner  # noqa: E402
from src.posture_ai.inference.stream_runtime import resize_for_display, resize_for_inference  # noqa: E402
from src.posture_ai.inference.video_source import SOURCE_CLOSED, SOURCE_FAILED, VideoSource  # noqa: E402
from src.posture_ai.multiperson.mediapipe_tasks import MultiPersonMediaPipeBackend  # noqa: E402
from src.posture_ai.multiperson.overlay import draw_multiperson_overlay  # noqa: E402
from src.posture_ai.multiperson.schema import MultiPersonFrameResult  # noqa: E402
from src.posture_ai.multiperson.telemetry import MultiPersonTelemetry  # noqa: E402
from src.posture_ai.multiperson.tracking import AnonymousPoseTracker  # noqa: E402


def _lower_visibility(subject) -> dict[str, float | None]:
    return {
        name: None if (point := subject.landmarks.by_name(name)) is None or point.visibility is None else float(point.visibility)
        for name in LOWER_LANDMARKS
    }


def _draw_status(frame, *, selected: str | None, activity: str | None, posture: str | None, collecting: bool, expected: int | None):
    output = frame.copy()
    height = output.shape[0]
    lines = (
        f"TARGET: {selected or 'NONE'} | EXPECTED: {expected or 'NOT SET'} | {'COLLECTING' if collecting else 'PAUSED'}",
        f"HUMAN LABEL: activity={activity or 'NONE'} posture={posture or 'NONE'}",
        "D=STANDING/NONE | 1=SITTING/TUP | 2=SITTING/TLB | 0=clear",
        "TAB/N=target | SPACE=start/pause | [/]=expected people | Q=finish | NO RAW VIDEO",
    )
    for index, line in enumerate(lines):
        y = height - 82 + index * 20
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _v4_diagnostic(runner: RGBNativeV4Runner, subject) -> tuple[dict | None, str]:
    """Keep frozen V4 separate from the manual collection target."""

    if not subject.core_valid or subject.world_landmarks is None:
        return None, "NOT_RUN_CORE_INVALID"
    try:
        return runner.predict_world_landmarks(subject.world_landmarks, image_visibility=subject.landmarks), "DIAGNOSTIC_ONLY"
    except (KeyError, TypeError, ValueError):
        return None, "NOT_RUN_V4_FEATURE_OR_VISIBILITY_REJECTED"


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual SITTING/STANDING + TUP/TLB pilot; no raw video is recorded")
    parser.add_argument("--source", default="rtsp")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default="HIKVISION_MAIN")
    parser.add_argument("--camera-id", default="CAM02")
    parser.add_argument("--subject-id", required=True, help="Anonymous participant code, e.g. P001")
    parser.add_argument("--clip-id", required=True, help="One safe pilot clip code, e.g. P001_PILOT")
    parser.add_argument("--camera-position-id", default="UNKNOWN")
    parser.add_argument("--distance-band", choices=("NEAR", "MID", "FAR", "UNKNOWN"), default="UNKNOWN")
    parser.add_argument("--expected-people-count", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--max-people", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--pose-model", default="metadata/pose_landmarker_heavy.task")
    parser.add_argument("--model", default="outputs/models/rgb_native_v4")
    parser.add_argument("--rtsp-config", default="configs/rtsp_profiles.yaml")
    parser.add_argument("--output-root", default="data/behavior_partial_pose_v1")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--no-display", action="store_true", help="Diagnostics only; no manual samples can be captured")
    args = parser.parse_args()
    participant = validate_subject_id(args.subject_id)
    clip_id = validate_safe_id(args.clip_id, field_name="clip_id")
    position_id = validate_safe_id(args.camera_position_id, field_name="camera_position_id")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    session_id = make_session_id(participant, clip_id)
    writer = ControlledSampleWriter(
        args.output_root,
        feature_names=activity_feature_names(FULLER_AVAILABLE_POSE),
        allowed_labels=ACTIVITY_LABELS,
        schema_version=BEHAVIOR_COLLECTION_SCHEMA_VERSION,
        extra_columns=BEHAVIOR_EXTRA_SAMPLE_COLUMNS,
    )
    source = VideoSource(args.source, camera_id=args.camera_id, profile=args.profile, config_path=args.rtsp_config, reconnect=True, low_latency=True)
    latest = LatestFrameInput(source, asynchronous=source.source_type == "RTSP")
    if not latest.open():
        print("Could not open explicit source", file=sys.stderr)
        return 2
    backend = MultiPersonMediaPipeBackend(args.pose_model, max_people=args.max_people, running_mode="video")
    tracker = AnonymousPoseTracker(max_people=args.max_people)
    runner = RGBNativeV4Runner(args.model)
    telemetry = MultiPersonTelemetry(max_events=2000)
    selected: str | None = None
    activity_label: str | None = None
    posture_label: str | None = None
    collecting = False
    expected = args.expected_people_count
    started = time.perf_counter()
    origin: float | None = None
    frame_index = 0
    accepted = 0
    labels: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    close_reason = "USER_EXIT"
    try:
        if not args.no_display:
            cv2.namedWindow("Behavior Pilot Collection", cv2.WINDOW_NORMAL)
        while args.max_seconds is None or time.perf_counter() - started < args.max_seconds:
            read = latest.read()
            if not read.ok or read.frame is None:
                if read.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                    close_reason = read.reason or read.state
                    break
                continue
            if origin is None:
                origin = float(read.timestamp_seconds)
            timestamp = max(0.0, float(read.timestamp_seconds) - origin)
            inference, _ = resize_for_inference(read.frame, max_width=960, max_height=540)
            detections = backend.detect(inference, timestamp_ms=int(round(timestamp * 1000.0)))
            subjects = tracker.update(detections, timestamp)
            result = MultiPersonFrameResult(timestamp, subjects=subjects, camera_id=args.camera_id or source.camera_id or "CAM01", tracking_status="BEHAVIOR_PILOT_COLLECTION", events=tuple(tracker.last_events))
            telemetry.add_frame(result, frame_index=frame_index, source_metadata=read.health)
            if selected is None and subjects:
                selected = subjects[0].subject_id
            by_id = {subject.subject_id: subject for subject in subjects}
            if collecting and activity_label and posture_label and selected:
                activity, posture = validate_behavior_labels(activity_label, posture_label)
                subject = by_id.get(selected)
                if subject is None or subject.tracking_state not in {"NEW", "ACTIVE"}:
                    rejections["TARGET_NOT_ACTIVELY_VISIBLE"] += 1
                elif subject.world_landmarks is None:
                    rejections["WORLD_LANDMARKS_UNAVAILABLE"] += 1
                else:
                    features = build_activity_partial_pose_features(subject.landmarks, subject.world_landmarks, bbox=subject.bbox, variant=FULLER_AVAILABLE_POSE)
                    quality = partial_pose_collection_quality(subject.landmarks, subject.world_landmarks, features)
                    if quality["accepted"]:
                        prediction, v4_status = _v4_diagnostic(runner, subject) if activity == "SITTING" else (None, "NOT_RUN_STANDING")
                        pose_index = next((index for index, item in enumerate(subjects) if item.subject_id == selected), -1)
                        writer.append_sample(
                            sample_id=validate_safe_id(f"{session_id}_{selected}_{frame_index:06d}", field_name="sample_id"),
                            session_id=session_id, subject_id=participant, clip_id=clip_id, human_label=activity,
                            timestamp_seconds=timestamp, frame_index=frame_index, source_type=source.source_type,
                            camera_id=source.camera_id, camera_profile=source.spec.profile,
                            frame_width=int(inference.shape[1]), frame_height=int(inference.shape[0]),
                            core_visibility=subject.core_visibility, quality_flag=quality["quality_flag"], quality_warnings=quality["reasons"],
                            v4_prediction=None if prediction is None else prediction.get("label"),
                            v4_confidence=None if prediction is None else prediction.get("confidence"),
                            v4_probabilities={} if prediction is None else prediction.get("probabilities", {}),
                            geometry=calculate_geometry(subject.world_landmarks), calibration_status="PARTIAL_POSE_ACTIVITY_COLLECTION",
                            feature_values=features.values, frame_age_ms=read.frame_age_ms,
                            source_timestamp_seconds=read.timestamp_seconds, mediapipe_timestamp_seconds=timestamp,
                            extra_values={
                                "human_activity_label": activity, "human_posture_label": posture, "label_source": "MANUAL_OPERATOR_ONLY",
                                "target_subject_id": selected, "track_id": selected, "pose_index": pose_index, "expected_people_count": expected,
                                "distance_band": args.distance_band, "camera_position_id": position_id,
                                "lower_body_visibility_json": json.dumps(_lower_visibility(subject), ensure_ascii=False),
                                "partial_quality_json": json.dumps(quality, ensure_ascii=False),
                                "activity_feature_schema_version": ACTIVITY_PARTIAL_POSE_V1_SCHEMA_VERSION,
                                "activity_feature_variant": FULLER_AVAILABLE_POSE, "v4_diagnostic_status": v4_status,
                            },
                        )
                        accepted += 1
                        labels[f"{activity}/{posture}"] += 1
                    else:
                        rejections[quality["quality_flag"]] += 1
                        writer.append_rejection({"session_id": session_id, "subject_id": participant, "track_id": selected, "human_activity_label": activity, "human_posture_label": posture, "reasons": quality["reasons"], "raw_video_recorded": False, "label_source": "MANUAL_OPERATOR_ONLY"})
            if not args.no_display:
                display, _ = resize_for_display(read.frame, max_width=1280, max_height=720)
                display = draw_multiperson_overlay(display, result)
                display = _draw_status(display, selected=selected, activity=activity_label, posture=posture_label, collecting=collecting, expected=expected)
                cv2.imshow("Behavior Pilot Collection", display)
                key_code = cv2.waitKey(1) & 0xFF
                key = chr(key_code).upper() if key_code else ""
                if key == "Q" or key_code == 27:
                    break
                if key == "D":
                    activity_label, posture_label, collecting = "STANDING", POSTURE_NONE, False
                elif key == "1":
                    activity_label, posture_label, collecting = "SITTING", "TUP", False
                elif key == "2":
                    activity_label, posture_label, collecting = "SITTING", "TLB", False
                elif key == "0":
                    activity_label = posture_label = None
                    collecting = False
                elif key_code == 32:
                    collecting = bool(selected and activity_label and posture_label) and not collecting
                elif key_code == 9 or key == "N":
                    ids = [subject.subject_id for subject in subjects]
                    if ids:
                        selected = ids[0] if selected not in ids else ids[(ids.index(selected) + 1) % len(ids)]
                elif key == "[":
                    expected = max(1, int(expected or 1) - 1)
                elif key == "]":
                    expected = min(args.max_people, int(expected or 1) + 1)
            frame_index += 1
    finally:
        backend.close()
        latest.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    telemetry_path = Path(args.output_root) / "sessions" / f"{session_id}.jsonl"
    telemetry.write_jsonl(telemetry_path)
    metadata_path = writer.write_session_metadata({
        "session_id": session_id, "participant_code": participant, "clip_id": clip_id, "accepted_samples_this_session": accepted,
        "label_counts": dict(labels), "rejection_counts": dict(rejections), "frames_seen": frame_index, "close_reason": close_reason,
        "source_type": source.source_type, "camera_id": source.camera_id, "camera_profile": source.spec.profile,
        "activity_feature_schema_version": ACTIVITY_PARTIAL_POSE_V1_SCHEMA_VERSION, "activity_feature_variant": FULLER_AVAILABLE_POSE,
        "human_labels_are_manual": True, "prediction_used_as_label": False, "raw_video_recorded": False,
        "training_performed": False, "telemetry_path": str(telemetry_path),
    })
    print(json.dumps({"status": "BEHAVIOR_PILOT_COLLECTION_CLOSED", "accepted_samples": accepted, "samples_csv": str(writer.samples_path), "metadata": str(metadata_path), "raw_video_recorded": False, "predictions_used_as_labels": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
