"""Blind, guided webcam Human-GT collection; stores no pixels or predictions."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".venv_yolo/Lib/site-packages"))

import cv2
import numpy as np

from src.posture_ai.detection.yolo_optional import UltralyticsPersonTracker
from src.posture_ai.features.upper_body_posture_v2 import CORE_NAMES, FEATURES, SCHEMA, build_features
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput
from src.posture_ai.inference.pose_v2_stage import MediaPipeCropPoseBackend, POSE_PENDING, PoseV2Stage
from src.posture_ai.inference.upper_body_posture_v1 import verify_v4
from src.posture_ai.inference.upper_body_posture_v2 import evidence_xyz
from src.posture_ai.inference.video_source import SOURCE_CLOSED, SOURCE_FAILED, VideoSource
from src.posture_ai.monitoring.long_session import atomic_write_json

LABEL_KEYS = {ord("1"): "TUP", ord("2"): "TLF", ord("3"): "TLB", ord("4"): "TLL", ord("5"): "TLR"}
LABEL_NUMBERS = {label: chr(key) for key, label in LABEL_KEYS.items()}
LABELS = tuple(LABEL_KEYS.values())
PARTICIPANT_RE = re.compile(r"^P[0-9]{3}$")
DATASET_SCHEMA = "webcam_human_gt_v1"
FEATURE_CONDITION = "NO_HIP"


def build_sequence(labels: list[str] | tuple[str, ...], rounds_per_class: int,
                   round_start: int = 1) -> list[tuple[str, int]]:
    if not labels or any(label not in LABELS for label in labels):
        raise ValueError("labels must contain supported posture labels")
    if rounds_per_class < 1 or round_start < 1:
        raise ValueError("round count and start must be positive")
    return [(label, round_index)
            for label in labels
            for round_index in range(round_start, round_start + rounds_per_class)]


def validate_participant(value: str) -> str:
    participant = str(value).strip().upper()
    if participant == "P000" or not PARTICIPANT_RE.fullmatch(participant):
        raise ValueError("participant must be an anonymous ID such as P001")
    return participant


def hip_visibility(evidence) -> str:
    values = [evidence.landmark_visibility.get(name) for name in ("left_hip", "right_hip")]
    visible = sum(value is not None and math.isfinite(float(value)) and float(value) >= 0.5 for value in values)
    return ("BOTH_HIDDEN", "PARTLY_HIDDEN", "VISIBLE")[visible]


def accepted_record(*, evidence, participant: str, session_id: str, segment_id: str,
                    round_index: int, human_label: str, timestamp_utc: str,
                    elapsed_seconds: float, frame_index: int, camera_index: int,
                    resolution: tuple[int, int], human_confirmed: bool) -> dict:
    participant = validate_participant(participant)
    if not human_confirmed or human_label not in LABELS:
        raise ValueError("accepted observations require an explicit human posture label")
    if evidence.pose_state != "POSE_OK" or not evidence.core_valid:
        raise ValueError("unsafe or invalid pose cannot enter accepted Human-GT data")
    xyz = evidence_xyz(evidence)
    features = build_features(xyz, FEATURE_CONDITION)[0]
    if not np.isfinite(features).all():
        raise ValueError("pose features must be finite")
    return {
        "schema_version": DATASET_SCHEMA,
        "record_type": "ACCEPTED_HUMAN_GT_OBSERVATION",
        "participant_id": participant,
        "session_id": session_id,
        "segment_id": segment_id,
        "round_index": int(round_index),
        "human_label": human_label,
        "label_status": "HUMAN_GROUND_TRUTH",
        "label_source": "HUMAN",
        "human_confirmation": "KEYBOARD_EXPLICIT",
        "timestamp_utc": timestamp_utc,
        "elapsed_seconds": float(elapsed_seconds),
        "frame_index": int(frame_index),
        "source_type": "USB",
        "camera_index": int(camera_index),
        "resolution": {"width": int(resolution[0]), "height": int(resolution[1])},
        "track_id": int(evidence.track_id),
        "pose_state": evidence.pose_state,
        "pose_valid": True,
        "core_valid": True,
        "crop_safety_rejected": False,
        "feature_schema_version": f"{SCHEMA}:{FEATURE_CONDITION}",
        "feature_names": list(FEATURES[FEATURE_CONDITION]),
        "features": features.tolist(),
        "world_landmarks_xyz": evidence.world_landmarks_xyz,
        "landmark_visibility": evidence.landmark_visibility,
        "hip_visibility_proxy": hip_visibility(evidence),
        "model_prediction": None,
        "v4_called": False,
        "raw_video_recorded": False,
        "raw_image_saved": False,
    }


def rejected_record(*, participant: str, session_id: str, segment_id: str,
                    round_index: int, human_label: str, reason: str,
                    elapsed_seconds: float, frame_index: int | None,
                    track_id: int | None, pose_state: str | None) -> dict:
    return {
        "schema_version": DATASET_SCHEMA,
        "record_type": "REJECTED_EVIDENCE",
        "participant_id": validate_participant(participant),
        "session_id": session_id,
        "segment_id": segment_id,
        "round_index": int(round_index),
        "human_label": human_label,
        "label_status": "REJECTED_NOT_TRAINING_DATA",
        "label_source": "HUMAN",
        "elapsed_seconds": float(elapsed_seconds),
        "frame_index": frame_index,
        "track_id": track_id,
        "pose_state": pose_state,
        "pose_valid": False,
        "crop_safety_rejected": pose_state == "POSE_AMBIGUOUS",
        "reason": reason,
        "model_prediction": None,
        "v4_called": False,
        "raw_video_recorded": False,
        "raw_image_saved": False,
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_dataset(session_dir: Path) -> dict:
    accepted = read_jsonl(session_dir / "accepted_observations.jsonl")
    rejected = read_jsonl(session_dir / "rejected_evidence.jsonl")
    sample_keys, dimensions = set(), set()
    for row in accepted:
        if row.get("label_status") != "HUMAN_GROUND_TRUTH" or row.get("label_source") != "HUMAN":
            raise ValueError("accepted row is not explicit Human Ground Truth")
        if row.get("human_label") not in LABELS or row.get("model_prediction") is not None:
            raise ValueError("label leakage or unsupported label")
        validate_participant(row.get("participant_id", ""))
        key = (row.get("segment_id"), row.get("frame_index"))
        if key in sample_keys:
            raise ValueError("duplicate accepted observation")
        sample_keys.add(key)
        dimensions.add(len(row.get("features", [])))
        forbidden = {"name", "face", "identity_embedding", "rtsp_url", "password", "credentials"} & set(row)
        if forbidden or row.get("raw_video_recorded") or row.get("raw_image_saved") or row.get("v4_called"):
            raise ValueError("privacy or frozen-model contract violated")
    if dimensions - {len(FEATURES[FEATURE_CONDITION])}:
        raise ValueError("feature dimensionality mismatch")
    segments = {row["segment_id"] for row in accepted} | {row["segment_id"] for row in rejected}
    return {"status": "PASS", "accepted_observations": len(accepted), "rejected_observations": len(rejected),
            "segment_count": len(segments), "feature_dimensions": sorted(dimensions)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="webcam", choices=("webcam",))
    parser.add_argument("--camera-index", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--participant", default="P001")
    parser.add_argument("--labels", nargs="+", choices=LABELS, default=list(LABELS))
    parser.add_argument("--rounds-per-class", type=int, default=5)
    parser.add_argument("--round-start", type=int, default=1)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--record-seconds", type=float, default=5.0)
    parser.add_argument("--max-pose-per-second", type=float, default=2.0)
    parser.add_argument("--max-session-minutes", type=float, default=20.0)
    parser.add_argument("--output-root", default="datasets/derived/webcam_human_gt_v1")
    args = parser.parse_args()
    participant = validate_participant(args.participant)
    if args.rounds_per_class < 1 or args.round_start < 1 or min(args.settle_seconds, args.record_seconds, args.max_pose_per_second, args.max_session_minutes) <= 0:
        parser.error("rounds and timing values must be positive")

    started_utc = datetime.now(timezone.utc)
    session_id = started_utc.strftime("SESSION_%Y%m%dT%H%M%SZ") + f"_{participant}"
    session_dir = Path(args.output_root) / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    accepted_path, rejected_path = session_dir / "accepted_observations.jsonl", session_dir / "rejected_evidence.jsonl"
    sequence = build_sequence(args.labels, args.rounds_per_class, args.round_start)
    source = LatestFrameInput(VideoSource(str(args.camera_index), reconnect=False,
        preferred_width=args.camera_width, preferred_height=args.camera_height, preferred_fps=args.camera_fps))
    tracker = UltralyticsPersonTracker("outputs/yolo_tracking/weights/yolo26n.pt", device="0", confidence=0.35, tracker="bytetrack.yaml")
    stage = PoseV2Stage(MediaPipeCropPoseBackend("metadata/pose_landmarker_heavy.task", core_landmark_names=CORE_NAMES),
                        max_pose_per_second=args.max_pose_per_second)
    before_v4 = verify_v4()
    state, target_id, segment_index = "SELECT_TARGET", None, 0
    state_started = time.monotonic()
    last_rejection = -1e12
    accepted_counts, rejected_counts, hips, tracks = Counter(), Counter(), Counter(), set()
    completed_segments, processed = [], 0
    current_boxes = []
    stopped_reason = "SESSION_TIMEOUT"
    window = "Webcam Human GT V1 - BLIND COLLECTION"

    def select_target(event, x, y, flags, param):
        nonlocal target_id, state
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        matches = [box for box in current_boxes if box.track_id is not None and box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2]
        chosen = max(matches, key=lambda box: box.confidence, default=None)
        if chosen is not None:
            target_id, state = int(chosen.track_id), "WAIT_CONFIRM"

    try:
        if not source.open():
            stopped_reason = "SOURCE_OPEN_FAILED"
            return 2
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, select_target)
        started = time.monotonic()
        with accepted_path.open("x", encoding="utf-8") as accepted_log, rejected_path.open("x", encoding="utf-8") as rejected_log:
            while time.monotonic() - started < args.max_session_minutes * 60:
                read = source.read()
                now = time.monotonic()
                if not read.ok or read.frame is None:
                    if read.state in (SOURCE_CLOSED, SOURCE_FAILED):
                        stopped_reason = "SOURCE_FAILED"
                        break
                    continue
                frame, elapsed = read.frame, now - started
                if frame.shape[1] > 960:
                    frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
                current_boxes = tracker.track(frame)
                tracks.update(int(box.track_id) for box in current_boxes if box.track_id is not None)
                processed += 1
                target = next((box for box in current_boxes if box.track_id == target_id), None)
                label, round_index = sequence[min(segment_index, len(sequence) - 1)]
                segment_id = f"{session_id}_{label}_R{round_index:02d}"

                if state == "SETTLING" and now - state_started >= args.settle_seconds:
                    state, state_started = "RECORDING", now
                if state == "RECORDING" and target is not None:
                    evidence_items = stage.process(frame, [target], elapsed, all_boxes=current_boxes)
                    for evidence in evidence_items:
                        if evidence.pose_state == POSE_PENDING:
                            continue
                        if evidence.pose_state == "POSE_OK":
                            try:
                                row = accepted_record(evidence=evidence, participant=participant, session_id=session_id,
                                    segment_id=segment_id, round_index=round_index, human_label=label,
                                    timestamp_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=elapsed,
                                    frame_index=read.frame_index, camera_index=args.camera_index,
                                    resolution=(frame.shape[1], frame.shape[0]), human_confirmed=True)
                            except ValueError as exc:
                                row = rejected_record(participant=participant, session_id=session_id, segment_id=segment_id,
                                    round_index=round_index, human_label=label, reason=str(exc), elapsed_seconds=elapsed,
                                    frame_index=read.frame_index, track_id=target_id, pose_state=evidence.pose_state)
                                rejected_log.write(json.dumps(row) + "\n"); rejected_log.flush(); rejected_counts[label] += 1
                            else:
                                accepted_log.write(json.dumps(row) + "\n"); accepted_log.flush()
                                accepted_counts[label] += 1; hips[row["hip_visibility_proxy"]] += 1
                        elif now - last_rejection >= 0.5:
                            row = rejected_record(participant=participant, session_id=session_id, segment_id=segment_id,
                                round_index=round_index, human_label=label, reason=evidence.detail or evidence.pose_state,
                                elapsed_seconds=elapsed, frame_index=read.frame_index, track_id=target_id,
                                pose_state=evidence.pose_state)
                            rejected_log.write(json.dumps(row) + "\n"); rejected_log.flush(); rejected_counts[label] += 1
                            last_rejection = now
                elif state == "RECORDING" and now - last_rejection >= 0.5:
                    row = rejected_record(participant=participant, session_id=session_id, segment_id=segment_id,
                        round_index=round_index, human_label=label, reason="TARGET_TRACK_UNAVAILABLE",
                        elapsed_seconds=elapsed, frame_index=read.frame_index, track_id=target_id, pose_state=None)
                    rejected_log.write(json.dumps(row) + "\n"); rejected_log.flush(); rejected_counts[label] += 1
                    last_rejection = now

                if state == "RECORDING" and now - state_started >= args.record_seconds:
                    completed_segments.append(segment_id)
                    segment_index += 1
                    if segment_index >= len(sequence):
                        state = "COMPLETE"
                        stopped_reason = "PROTOCOL_COMPLETED"
                    else:
                        state = "WAIT_CONFIRM"
                    state_started = now

                display = frame.copy()
                for box in current_boxes:
                    color = (0, 255, 0) if box.track_id == target_id else (0, 220, 220)
                    cv2.rectangle(display, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), color, 2)
                    cv2.putText(display, f"TRACK {box.track_id}", (round(box.x1), max(20, round(box.y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                remaining = max(0.0, (args.settle_seconds if state == "SETTLING" else args.record_seconds) - (now - state_started))
                lines = [f"{participant} | {min(segment_index + 1, len(sequence))}/{len(sequence)} | TARGET {target_id}",
                         f"REQUEST: {label} round {round_index}/{args.rounds_per_class}",
                         "Click target person" if state == "SELECT_TARGET" else
                         f"Press {LABEL_NUMBERS[label]} to explicitly confirm label" if state == "WAIT_CONFIRM" else
                         f"SETTLING {remaining:.1f}s" if state == "SETTLING" else
                         f"RECORDING {remaining:.1f}s | valid {sum(accepted_counts.values())} rejected {sum(rejected_counts.values())}" if state == "RECORDING" else
                         "COMPLETE - closing safely",
                         "BLIND GT: model predictions hidden | 0=uncertain/retry | Q/ESC=stop"]
                for index, text in enumerate(lines):
                    cv2.putText(display, text, (10, 24 + 24 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stopped_reason = "OPERATOR_STOP"
                    break
                if state == "WAIT_CONFIRM" and key in LABEL_KEYS and LABEL_KEYS[key] == label:
                    state, state_started = "SETTLING", now
                elif state in ("WAIT_CONFIRM", "SETTLING", "RECORDING") and key == ord("0"):
                    row = rejected_record(participant=participant, session_id=session_id, segment_id=segment_id,
                        round_index=round_index, human_label=label, reason="HUMAN_UNKNOWN_UNCERTAIN_RETRY",
                        elapsed_seconds=elapsed, frame_index=read.frame_index, track_id=target_id, pose_state=None)
                    rejected_log.write(json.dumps(row) + "\n"); rejected_log.flush(); rejected_counts[label] += 1
                    state, state_started = "WAIT_CONFIRM", now
                if state == "COMPLETE" and now - state_started >= 1:
                    break
    except (RuntimeError, ValueError, OSError) as exc:
        stopped_reason = f"ERROR_{type(exc).__name__}:{exc}"
        return 2
    finally:
        source.close(); stage.close(); cv2.destroyAllWindows()
        audit = validate_dataset(session_dir)
        if len(set(completed_segments)) != len(completed_segments):
            raise RuntimeError("DUPLICATE_COMPLETED_SEGMENT_ID")
        manifest = {
            "schema_version": DATASET_SCHEMA, "session_id": session_id, "participant_count": 1,
            "participants": [participant], "session_count": 1, "segment_count": len(completed_segments),
            "completed_segment_ids": completed_segments, "segments_per_label": dict(Counter(s.split("_")[-2] for s in completed_segments)),
            "label_counts": dict(accepted_counts), "accepted_observation_count": sum(accepted_counts.values()),
            "rejected_observation_count": sum(rejected_counts.values()), "rejected_by_label": dict(rejected_counts),
            "hip_visibility_distribution": dict(hips), "source_type": "USB", "camera_index": args.camera_index,
            "camera_preferences": {"width": args.camera_width, "height": args.camera_height, "fps": args.camera_fps},
            "source_health": source.health(), "feature_schema": f"{SCHEMA}:{FEATURE_CONDITION}",
            "collection_started_utc": started_utc.isoformat(), "collection_ended_utc": datetime.now(timezone.utc).isoformat(),
            "processed_frames": processed, "unique_track_ids": len(tracks), "track_ids_are_anonymous": True,
            "stopped_reason": stopped_reason, "human_labels_are_explicit": True, "prediction_used_as_label": False,
            "model_predictions_recorded": False, "raw_video_recorded": False, "raw_images_saved": False,
            "v4_called": False, "v4_sha256": verify_v4(), "audit": audit,
            "limitations": ["Single-participant calibration pilot", "Correlated observations are grouped by session and segment",
                            "No accuracy claim", "Tracker IDs are transient anonymous runtime IDs"],
        }
        if manifest["v4_sha256"] != before_v4:
            raise RuntimeError("FROZEN_V4_HASH_CHANGED")
        atomic_write_json(session_dir / "manifest.json", manifest)
        print(json.dumps({"dataset_path": str(session_dir), **manifest}, ensure_ascii=False, indent=2))
    return 0 if stopped_reason in ("PROTOCOL_COMPLETED", "OPERATOR_STOP") else 2


if __name__ == "__main__":
    raise SystemExit(main())
