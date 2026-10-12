"""Collect manually labelled TUP/TLB V4 feature rows without raw video by default."""

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

from src.posture_ai.data.controlled_collection import (  # noqa: E402
    CONTROLLED_LABELS,
    ControlledSampleWriter,
    evaluate_collection_quality,
    make_session_id,
    validate_safe_id,
    validate_subject_id,
)
from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig  # noqa: E402
from src.posture_ai.inference.pipeline import create_pose_landmarker  # noqa: E402
from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner  # noqa: E402
from src.posture_ai.inference.rtsp_ai_lab import draw_ai_lab_overlay, merge_source_metadata, monotonic_pose_timestamp_seconds, profile_for_source  # noqa: E402
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput  # noqa: E402
from src.posture_ai.inference.stream_runtime import resize_for_inference, resolve_rtsp_preset  # noqa: E402
from src.posture_ai.inference.video_source import SOURCE_CLOSED, SOURCE_FAILED, SourceConfigurationError, VideoSource  # noqa: E402
from src.posture_ai.inference.webcam_v1 import LiveFrameProcessor, prepare_display_frame  # noqa: E402


def draw_collection_overlay(frame, record: dict, image_pose, *, selected_label: str | None, capturing: bool, mirror: bool) -> object:
    output = draw_ai_lab_overlay(frame, record, image_pose, mirror=mirror, calibration=True)
    height, width = output.shape[:2]
    panel_x = 10
    panel_y = max(210, height - 125)
    panel_w = min(520, width - 20)
    overlay = output.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, min(height - 1, panel_y + 112)), (30, 25, 65), -1)
    output[:] = cv2.addWeighted(overlay, 0.72, output, 0.28, 0.0)
    label = selected_label or "NOT SELECTED"
    state = "CAPTURING ACCEPTED FRAMES" if capturing else "PAUSED"
    lines = [
        f"HUMAN LABEL: {label}    {state}",
        "1=TUP Upright   2=TLB Lean backward   SPACE=Start/Pause",
        "Q/ESC=Finish   Labels are manual; V4 prediction is diagnostic only",
        "RAW VIDEO: OFF",
    ]
    for offset, line in enumerate(lines):
        yy = panel_y + 24 + offset * 23
        cv2.putText(output, line, (panel_x + 10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(output, line, (panel_x + 10, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect manually labelled controlled TUP/TLB samples for later V5 review")
    parser.add_argument("--source", default="rtsp", help="rtsp/profile, file, or explicit USB index")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default="HIKVISION_MAIN")
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--subject-id", required=True, help="Anonymous participant code, for example P001")
    parser.add_argument("--clip-id", required=True, help="Short clip code, for example P001_TUP_01")
    parser.add_argument("--model", default="outputs/models/rgb_native_v4")
    parser.add_argument("--pose-model", default="metadata/pose_landmarker_heavy.task")
    parser.add_argument("--config", default="configs/posture_engine.yaml")
    parser.add_argument("--rtsp-config", default="configs/rtsp_profiles.yaml")
    parser.add_argument("--preset", choices=("RTSP_DESKTOP", "RTSP_BALANCED", "RTSP_LOW_LATENCY"), default="RTSP_LOW_LATENCY")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-reconnects", type=int, default=5)
    parser.add_argument("--output-root", default="data/controlled_rgb_v5")
    parser.add_argument("--no-display", action="store_true", help="Do not open a preview; no manual samples can be captured")
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()
    try:
        subject_id = validate_subject_id(args.subject_id)
        clip_id = validate_safe_id(args.clip_id, field_name="clip_id")
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_reconnects < 0:
        parser.error("--max-reconnects must not be negative")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")

    runtime = resolve_rtsp_preset(args.preset, config_path=args.rtsp_config)
    config = PostureEngineConfig.from_yaml(args.config)
    runner = RGBNativeV4Runner(args.model)
    writer = ControlledSampleWriter(args.output_root, feature_names=runner.expected_names)
    session_id = make_session_id(subject_id, clip_id)
    video = VideoSource(
        args.source,
        camera_id=args.camera_id,
        profile=profile_for_source(args.source, args.profile),
        config_path=args.rtsp_config,
        reconnect=True,
        max_reconnects=args.max_reconnects,
        stale_after_seconds=runtime["stale_after_seconds"],
        low_latency=bool(runtime.get("low_latency_buffer_request", False)),
    )
    latest = LatestFrameInput(video, asynchronous=video.source_type == "RTSP")
    if not latest.open():
        print(f"Controlled collection could not open source: {video.spec.source_display}", file=sys.stderr)
        return 2

    landmarker = create_pose_landmarker(args.pose_model, running_mode="video")
    processor = LiveFrameProcessor(runner, landmarker, config, calibration=True)
    selected_label: str | None = None
    capturing = False
    accepted_count = 0
    rejection_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    frame_index = 0
    processed_frames = 0
    started = time.perf_counter()
    last_timestamp = 0.0
    closed_reason = "USER_EXIT"
    last_pose_timestamp: float | None = None
    source_time_origin: float | None = None
    try:
        if not args.no_display:
            cv2.namedWindow("Controlled Posture Collection", cv2.WINDOW_NORMAL)
        while args.max_frames is None or frame_index < args.max_frames:
            if args.max_seconds is not None and time.perf_counter() - started >= args.max_seconds:
                closed_reason = "MAX_SECONDS"
                break
            source_result = latest.read()
            if not source_result.ok or source_result.frame is None:
                closed_reason = source_result.reason or "SOURCE_UNAVAILABLE"
                if source_result.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                    break
                continue
            frame = source_result.frame
            inference_frame, scale = resize_for_inference(
                frame,
                max_width=runtime.get("inference_max_width"),
                max_height=runtime.get("inference_max_height"),
            )
            fps = float(source_result.health.get("effective_received_fps") or source_result.health.get("reported_fps") or 0.0)
            source_timestamp = float(source_result.timestamp_seconds)
            if source_time_origin is None:
                source_time_origin = source_timestamp
            session_source_timestamp = max(0.0, source_timestamp - source_time_origin)
            pose_timestamp = monotonic_pose_timestamp_seconds(session_source_timestamp, last_pose_timestamp)
            last_pose_timestamp = pose_timestamp
            inference_started = time.perf_counter()
            record, image_pose = processor.process(
                inference_frame,
                frame_index=frame_index,
                timestamp_seconds=pose_timestamp,
                fps=fps,
            )
            inference_time_ms = (time.perf_counter() - inference_started) * 1000.0
            record = merge_source_metadata(record, source_result.health, source_type=video.source_type, camera_id=video.camera_id, profile=video.spec.profile, frame_scale=scale)
            record["inference_skipped"] = False
            record["source_timestamp_seconds"] = source_timestamp
            record["mediapipe_timestamp_seconds"] = float(pose_timestamp)
            record["frame_age_ms"] = source_result.frame_age_ms
            record["inference_time_ms"] = inference_time_ms
            record["collection_label_selected"] = selected_label
            record["collection_capturing"] = capturing
            processed_frames += 1

            height, width = inference_frame.shape[:2]
            gate = evaluate_collection_quality(
                pose_detected=bool(record.get("pose_detected")),
                core_visibility=record.get("core_visibility"),
                feature_values=record.get("v4_feature_values"),
                feature_names=record.get("v4_feature_names"),
                frame_width=width,
                frame_height=height,
                min_visibility=config.minimum_pose_visibility,
            )
            if capturing and selected_label is not None:
                if gate["accepted"]:
                    sample_id = f"{session_id}_{frame_index:06d}"
                    writer.append_sample(
                        sample_id=sample_id,
                        session_id=session_id,
                        subject_id=subject_id,
                        clip_id=clip_id,
                        human_label=selected_label,
                        timestamp_seconds=source_result.timestamp_seconds,
                        frame_index=frame_index,
                        source_type=video.source_type,
                        camera_id=video.camera_id,
                        camera_profile=video.spec.profile,
                        frame_width=width,
                        frame_height=height,
                        core_visibility=record.get("core_visibility", {}),
                        quality_flag=gate["quality_flag"],
                        quality_warnings=gate["warnings"],
                        v4_prediction=record.get("ai_label"),
                        v4_confidence=record.get("ai_confidence"),
                        v4_probabilities=record.get("ai_probabilities", {}),
                        geometry=record.get("geometry", {}),
                        calibration_status="RUNTIME_GEOMETRY_ONLY",
                        feature_values=record["v4_feature_values"],
                        frame_age_ms=source_result.frame_age_ms,
                        inference_time_ms=inference_time_ms,
                        source_timestamp_seconds=source_timestamp,
                        mediapipe_timestamp_seconds=pose_timestamp,
                    )
                    accepted_count += 1
                    label_counts[selected_label] += 1
                else:
                    for reason in gate["reasons"]:
                        rejection_counts[reason.split(":", 1)[0]] += 1
                    writer.append_rejection({
                        "schema_version": "controlled_rgb_v5_v1",
                        "session_id": session_id,
                        "subject_id": subject_id,
                        "clip_id": clip_id,
                        "frame_index": frame_index,
                        "timestamp_seconds": float(source_result.timestamp_seconds),
                        "human_label_selected": selected_label,
                        "reasons": gate["reasons"],
                        "source_status": source_result.health.get("source_status"),
                        "raw_video_recorded": False,
                    })

            if not args.no_display:
                annotated = draw_collection_overlay(
                    prepare_display_frame(frame, mirror=not args.no_mirror),
                    record,
                    image_pose,
                    selected_label=selected_label,
                    capturing=capturing,
                    mirror=not args.no_mirror,
                )
                cv2.imshow("Controlled Posture Collection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    closed_reason = "USER_EXIT"
                    break
                if key == ord("1"):
                    selected_label = "TUP"
                elif key == ord("2"):
                    selected_label = "TLB"
                elif key == ord(" "):
                    if selected_label is not None:
                        capturing = not capturing
                elif key == ord("r"):
                    capturing = False
                    selected_label = None
            frame_index += 1
    except KeyboardInterrupt:
        closed_reason = "KEYBOARD_INTERRUPT"
    finally:
        landmarker.close()
        final_health = latest.health()
        latest.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    metadata = {
        "session_id": session_id,
        "subject_id": subject_id,
        "clip_id": clip_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_type": video.source_type,
        "source_display": video.spec.source_display,
        "camera_id": video.camera_id,
        "camera_profile": video.spec.profile,
        "preset": runtime["name"],
        "frames_seen": frame_index,
        "frames_processed": processed_frames,
        "accepted_samples_this_session": accepted_count,
        "label_counts_this_session": dict(label_counts),
        "rejection_counts": dict(rejection_counts),
        "last_source_health": final_health,
        "close_reason": closed_reason,
        "human_label_options": list(CONTROLLED_LABELS),
        "human_labels_are_manual": True,
        "v4_predictions_are_not_ground_truth": True,
        "raw_video_recorded": False,
        "training_performed": False,
        "data_is_for_controlled_robustness_review": True,
    }
    metadata_path = writer.write_session_metadata(metadata)
    print(json.dumps({
        "status": "CONTROLLED_COLLECTION_SESSION_CLOSED",
        "session_id": session_id,
        "accepted_samples_this_session": accepted_count,
        "metadata": str(metadata_path),
        "samples_csv": str(writer.samples_path),
        "rejections_jsonl": str(writer.rejections_path),
        "raw_video_recorded": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
