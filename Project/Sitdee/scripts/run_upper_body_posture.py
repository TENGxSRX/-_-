"""RTSP diagnostic or blind human-GT collector; no raw media recording or V4 calls."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from src.posture_ai.detection.yolo_optional import UltralyticsPersonTracker
from src.posture_ai.features.upper_body_posture_v1 import SCHEMA, CORE_NAMES, CORE_INDICES, build_features
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput
from src.posture_ai.inference.pose_v2_stage import MediaPipeCropPoseBackend, PoseV2Stage
from src.posture_ai.inference.upper_body_posture_v1 import UpperBodyPostureRunner, verify_v4
from src.posture_ai.inference.video_source import VideoSource, SOURCE_CLOSED, SOURCE_FAILED
from src.posture_ai.monitoring.long_session import CandidateSelector, CandidateSelectorConfig, iou

HUMAN_KEYS = {ord("1"): "UPRIGHT", ord("2"): "FORWARD_LEAN", ord("3"): "BACKWARD_OR_RECLINED",
              ord("4"): "LATERAL_LEFT", ord("5"): "LATERAL_RIGHT", ord("0"): "UNKNOWN/SKIP"}


def human_record(participant, evidence, label, frame_index, captured_at):
    if not re.fullmatch(r"P[0-9]{3}", participant) or participant == "P000" or label not in HUMAN_KEYS.values():
        raise ValueError("Anonymous participant or human label invalid")
    core = [evidence.world_landmarks_xyz[i] for i in CORE_INDICES]
    return {"schema": SCHEMA, "record_type": "HUMAN_GROUND_TRUTH", "label_source": "HUMAN_GROUND_TRUTH",
            "participant_id": participant, "human_label": label, "track_id": evidence.track_id,
            "frame_index": frame_index, "captured_at": captured_at, "annotated_at": datetime.now(timezone.utc).isoformat(),
            "core_xyz": core, "core_visibility": {name: evidence.landmark_visibility[name] for name in CORE_NAMES},
            "features": build_features(core)[0].tolist(), "coordinate_system": "MEDIAPIPE_CROP_WORLD",
            "seated_context": "UNKNOWN" if label == "UNKNOWN/SKIP" else "HUMAN_CONFIRMED_AT_ANNOTATION", "model_prediction": None,
            "raw_video_recorded": False, "raw_image_saved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default="HIKVISION_MAIN")
    parser.add_argument("--max-seconds", type=float, default=30)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--collect-human-gt", action="store_true")
    parser.add_argument("--participant", default=None)
    parser.add_argument("--model-dir", default="outputs/models/upper_body_posture_v1_drivenet_exp")
    parser.add_argument("--output-root", default="outputs/upper_body_sessions")
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("max-seconds must be positive")
    if args.collect_human_gt and (args.no_display or not re.fullmatch(r"P[0-9]{3}", args.participant or "") or args.participant == "P000"):
        parser.error("Collector requires display and anonymous --participant P001, P002, ...")
    before = verify_v4()
    runner = None if args.collect_human_gt else UpperBodyPostureRunner(args.model_dir)
    session = Path(args.output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session.mkdir(parents=True, exist_ok=False)
    source = LatestFrameInput(VideoSource("rtsp", profile=args.profile, config_path="configs/rtsp_profiles.yaml",
                                         low_latency=True, reconnect=True, max_reconnects=0))
    tracker = UltralyticsPersonTracker("outputs/yolo_tracking/weights/yolo26n.pt", device="0", confidence=0.35, tracker="bytetrack.yaml")
    stage = PoseV2Stage(MediaPipeCropPoseBackend("metadata/pose_landmarker_heavy.task", core_landmark_names=CORE_NAMES), max_pose_per_second=2)
    selector = CandidateSelector(CandidateSelectorConfig(representative_interval_seconds=0.5, event_cooldown_seconds=0.5))
    current_boxes, clicked, known_seated = [], [None], set()
    pose_counts, outputs, ids = Counter(), Counter(), set()
    processed = labels_written = 0
    stopped = "DURATION_COMPLETED"
    snapshot = None
    cache = {}
    started = time.perf_counter()
    window = "Upper body human GT" if args.collect_human_gt else "Upper body diagnostic"

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            matches = [b for b in current_boxes if b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2 and b.track_id is not None]
            box = max(matches, key=lambda b: b.confidence, default=None)
            clicked[0] = None if box is None else int(box.track_id)
            if box is not None and not args.collect_human_gt:
                known_seated.symmetric_difference_update({int(box.track_id)})

    try:
        with (session / ("human_gt.jsonl" if args.collect_human_gt else "predictions.jsonl")).open("x", encoding="utf-8") as log:
            if not source.open():
                stopped = "SOURCE_OPEN_FAILED"
                return 2
            if not args.no_display:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(window, click)
            while time.perf_counter() - started < args.max_seconds:
                read = source.read()
                if not read.ok or read.frame is None:
                    if read.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                        stopped = "SOURCE_FAILED"
                        break
                    continue
                frame = read.frame
                if frame.shape[1] > 960:
                    frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
                current_boxes = tracker.track(frame)
                elapsed = time.perf_counter() - started
                now_ids = {int(b.track_id) for b in current_boxes if b.track_id is not None}
                selected = selector.inspect(current_boxes, timestamp_seconds=elapsed, frame_width=frame.shape[1], frame_height=frame.shape[0], new_ids=now_ids - ids, reacquired_ids=set())
                ids.update(now_ids)
                chosen_ids = {r["anonymous_track_id"] for r in selected}
                chosen = [b for b in current_boxes if b.track_id in chosen_ids]
                if args.collect_human_gt:
                    chosen = [b for b in chosen if b.track_id == clicked[0]]
                records = stage.process(frame, chosen, elapsed, all_boxes=current_boxes)
                captured_at = datetime.now(timezone.utc).isoformat()
                for evidence in records:
                    pose_counts[evidence.pose_state] += 1
                    if args.collect_human_gt:
                        if evidence.pose_state == "POSE_OK":
                            snapshot = (frame.copy(), evidence, read.frame_index, captured_at)
                    else:
                        prediction = runner.predict(evidence)
                        # Upper torso alone cannot prove sitting versus standing. Operator confirms context per ID.
                        if prediction["output"] != "UNKNOWN" and evidence.track_id not in known_seated:
                            prediction = {"output": "UNKNOWN", "subtype": None, "reason": "SEATED_CONTEXT_UNVERIFIED"}
                        cache[evidence.track_id] = (elapsed, prediction)
                        outputs[prediction["output"]] += 1
                        log.write(json.dumps({"record_type": "MODEL_PREDICTION", "label_source": "MODEL_PREDICTION",
                                              "schema": SCHEMA, "track_id": evidence.track_id, "frame_index": read.frame_index,
                                              "timestamp_utc": captured_at, "pose_state": evidence.pose_state,
                                              "core_visibility": {n: evidence.landmark_visibility.get(n) for n in CORE_NAMES},
                                              **prediction, "v4_called": False, "raw_video_recorded": False}) + "\n")
                        log.flush()
                processed += 1
                cache = {tid: value for tid, value in cache.items() if tid in now_ids}
                known_seated.intersection_update(now_ids)
                if args.no_display:
                    continue
                display = frame.copy()
                for box in current_boxes:
                    tid = box.track_id
                    text = "UNKNOWN"
                    if not args.collect_human_gt and tid in cache and elapsed - cache[tid][0] <= 1:
                        overlap = any(other.track_id != tid and iou((box.x1, box.y1, box.x2, box.y2), (other.x1, other.y1, other.x2, other.y2)) >= 0.25 for other in current_boxes)
                        if not overlap:
                            p = cache[tid][1]
                            text = p["output"].replace("_SITTING", "") + (" " + p["subtype"] if p["subtype"] else "")
                    if args.collect_human_gt:
                        text = "SELECTED" if tid == clicked[0] else "CLICK TARGET"
                    cv2.rectangle(display, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), (0, 220, 220), 2)
                    cv2.putText(display, f"TRACK {tid} {text}", (round(box.x1), max(20, round(box.y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)
                instruction = "Click verified SEATED person to toggle context | Q quit | EXPERIMENTAL, no measured live accuracy"
                if args.collect_human_gt:
                    instruction = "Click seated target | SPACE freeze exact pose frame | Q quit | model predictions hidden"
                cv2.putText(display, instruction, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stopped = "OPERATOR_STOP"
                    break
                if args.collect_human_gt and key == ord(" ") and snapshot and snapshot[1].track_id == clicked[0] and clicked[0] in now_ids:
                    frozen, evidence, frame_index, timestamp = snapshot
                    frozen = frozen.copy()
                    box = evidence.bbox
                    cv2.rectangle(frozen, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (0, 220, 220), 2)
                    cv2.putText(frozen, f"LABEL TRACK {evidence.track_id}: 1 upright 2 forward 3 reclined 4 subject-left 5 subject-right 0 skip", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.imshow(window, frozen)
                    label_key = cv2.waitKey(0) & 0xFF
                    if label_key in HUMAN_KEYS:
                        log.write(json.dumps(human_record(args.participant, evidence, HUMAN_KEYS[label_key], frame_index, timestamp)) + "\n")
                        log.flush()
                        labels_written += 1
                    snapshot = None
    except (RuntimeError, ValueError, OSError) as exc:
        stopped = f"ERROR_{type(exc).__name__}"
        return 2
    finally:
        source.close()
        stage.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        summary = {"schema": SCHEMA, "stopped_reason": stopped, "processed_frames": processed,
                   "track_ids_count": len(ids), "pose_counts": dict(pose_counts), "pose_stage": stage.summary(),
                   "outputs": dict(outputs), "human_gt_rows": labels_written, "model_loaded": runner is not None,
                   "source_closed": source.health().get("source_status") == SOURCE_CLOSED,
                   "seated_context": "OPERATOR_CONFIRMED_PER_TRACK_OR_UNKNOWN", "rejection_policy": "EXPERIMENTAL",
                   "v4_called": False, "v4_sha256": verify_v4(), "raw_video_recorded": False, "raw_images_saved": False}
        assert summary["v4_sha256"] == before
        (session / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({**summary, "session": str(session)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
