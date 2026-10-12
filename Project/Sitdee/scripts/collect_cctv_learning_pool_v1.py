"""Unattended upper-body CCTV pool; no labels, media recording or training."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / ".venv_yolo/Lib/site-packages"))
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import cv2
import numpy as np

from scripts.run_upper_body_posture_v2 import paired_predictions
from scripts.train_upper_body_posture_v2 import frozen_hashes
from src.posture_ai.detection.yolo_optional import UltralyticsPersonTracker
from src.posture_ai.features.upper_body_posture_v1 import build_features as v1_features
from src.posture_ai.features.upper_body_posture_v2 import CORE_NAMES, build_features
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput
from src.posture_ai.inference.pose_v2_stage import MediaPipeCropPoseBackend, PoseV2Stage
from src.posture_ai.inference.upper_body_posture_v1 import UpperBodyPostureRunner
from src.posture_ai.inference.upper_body_posture_v2 import UpperBodyPostureV2Runner, evidence_xyz
from src.posture_ai.inference.video_source import SOURCE_CLOSED, SOURCE_FAILED, VideoSource
from src.posture_ai.monitoring.cctv_learning_pool import LearningPool, PoolCapacityReached, observation
from src.posture_ai.monitoring.long_session import CandidateSelector, CandidateSelectorConfig

MAX_TRACK_IDS = 65536


def model_hashes():
    return frozen_hashes() | {"v2": hashlib.sha256((ROOT / "outputs/models/upper_body_posture_v2/candidate.joblib").read_bytes()).hexdigest()}


def diagnostic(prediction, bundle, features):
    score = None
    if features is not None and np.isfinite(features).all():
        lo, hi = bundle["minimum"], bundle["maximum"]
        score = float(np.maximum(np.maximum(lo - features[0], features[0] - hi), 0).max())
    return {**prediction, "confidence": None, "ood": "OOD" in prediction["reason"],
            "range_excess_max": score, "label_source": "MODEL_PREDICTION"}


def run(args):
    before = model_hashes()
    session_id = datetime.now(timezone.utc).strftime("SESSION_%Y%m%dT%H%M%S%fZ")
    pool = LearningPool(args.output_root, session_id, max_bytes=int(args.max_pool_mib * 1024**2),
                        max_files=args.max_files, min_free_bytes=int(args.min_free_mib * 1024**2), max_representatives=args.max_representatives)
    pool.manifest["limits"]["track_id_capacity"] = MAX_TRACK_IDS
    pool.manifest["limits"]["source_max_reconnects"] = args.max_reconnects
    source = stage = None
    runtime = {"processed_frames": 0, "person_detections": 0, "pose_attempts": 0, "valid_upper_body_poses": 0,
               "rejected_pose_attempts": 0, "maximum_people": 0, "overlap_observations": 0, "overlap_records": 0,
               "pose_pending": 0, "source_read_waits": 0, "elapsed_seconds": 0, "source_reconnects": 0}
    rejection_reasons, pose_states = Counter(), Counter()
    seen, active, missing = set(), set(), set()
    last_overlap = {}
    started, last_checkpoint = None, -1e12
    stopped = "DURATION_COMPLETED"
    try:
        v1, v2 = UpperBodyPostureRunner(), UpperBodyPostureV2Runner()
        if v2.bundle["condition"] != "NO_HIP":
            raise ValueError("COLLECTOR_REQUIRES_FROZEN_NO_HIP_SCHEMA")
        tracker = UltralyticsPersonTracker("outputs/yolo_tracking/weights/yolo26n.pt", device="0", confidence=0.35, tracker="bytetrack.yaml")
        stage = PoseV2Stage(MediaPipeCropPoseBackend("metadata/pose_landmarker_heavy.task", core_landmark_names=CORE_NAMES), max_pose_per_second=2)
        selector = CandidateSelector(CandidateSelectorConfig(representative_interval_seconds=0.5, event_cooldown_seconds=0.5))
        source = LatestFrameInput(VideoSource("rtsp", profile="HIKVISION_MAIN", config_path="configs/rtsp_profiles.yaml", low_latency=True, reconnect=True, max_reconnects=args.max_reconnects))
        pool.manifest["model_hashes"] = before
        pool.manifest["pipeline"] = "RTSP/YOLO26n/ByteTrack/CandidateSelector/MediaPipeHeavy/nose+shoulder-world-features/quality/dedup/diversity"
        if not source.open():
            stopped = "SOURCE_OPEN_FAILED"
        else:
            started = time.perf_counter()
            while time.perf_counter() - started < args.duration_minutes * 60:
                read = source.read()
                seconds = time.perf_counter() - started
                runtime["elapsed_seconds"] = seconds
                if not read.ok or read.frame is None:
                    runtime["source_read_waits"] += 1
                    if read.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                        stopped = "SOURCE_FAILED"
                        break
                else:
                    frame = read.frame
                    if frame.shape[1] > 960:
                        frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
                    boxes = tracker.track(frame)
                    current = {int(b.track_id) for b in boxes if b.track_id is not None}
                    new, lost, reacquired = current - seen, active - current, current & missing
                    if len(seen | current) > MAX_TRACK_IDS:
                        stopped = "TRACK_MEMORY_CAPACITY"
                        break
                    for state, ids in (("NEW", new), ("LOST", lost), ("REACQUIRED", reacquired)):
                        for track in sorted(ids):
                            pool.event(track, state, seconds)
                    chosen = selector.inspect(boxes, timestamp_seconds=seconds, frame_width=frame.shape[1], frame_height=frame.shape[0], new_ids=new, reacquired_ids=reacquired)
                    seen.update(current)
                    missing.update(lost)
                    missing.difference_update(reacquired)
                    active = current
                    runtime["processed_frames"] += 1
                    runtime["person_detections"] += len(boxes)
                    runtime["maximum_people"] = max(runtime["maximum_people"], len(boxes))
                    chosen_ids = {r["anonymous_track_id"] for r in chosen}
                    records = stage.process(frame, [b for b in boxes if b.track_id in chosen_ids], seconds, all_boxes=boxes)
                    for evidence in records:
                        pose_states[evidence.pose_state] += 1
                        if evidence.pose_state == "POSE_PENDING":
                            runtime["pose_pending"] += 1
                            continue
                        if evidence.pose_state == "POSE_AMBIGUOUS":
                            runtime["overlap_observations"] += 1
                            if seconds - last_overlap.get(evidence.track_id, -1e12) < 10:
                                continue
                            last_overlap[evidence.track_id] = seconds
                            runtime["overlap_records"] += 1
                        else:
                            runtime["pose_attempts"] += 1
                        _, p1, p2 = paired_predictions(evidence, v1, v2)
                        core = evidence_xyz(evidence)
                        try:
                            f1 = v1_features(core[1:])
                        except ValueError:
                            f1 = None
                        try:
                            f2 = build_features(core, "NO_HIP")
                        except ValueError:
                            f2 = None
                        predictions = {"v1": diagnostic(p1, v1.bundle, f1), "v2": diagnostic(p2, v2.bundle, f2)}
                        state = "NEW" if evidence.track_id in new else "REACQUIRED" if evidence.track_id in reacquired else "ACTIVE"
                        record = observation(evidence, session_id=session_id, seconds=seconds, frame_index=read.frame_index,
                                             width=frame.shape[1], height=frame.shape[0], predictions=predictions, tracking_state=state)
                        if evidence.pose_state != "POSE_AMBIGUOUS":
                            if record["quality_rejection"]:
                                runtime["rejected_pose_attempts"] += 1
                                rejection_reasons[record["quality_rejection"]] += 1
                            else:
                                runtime["valid_upper_body_poses"] += 1
                        pool.add(record)
                if seconds - last_checkpoint >= 30:
                    last_checkpoint = seconds
                    runtime.update(unique_tracks=len(seen), rejection_reasons=dict(rejection_reasons), pose_states=dict(pose_states))
                    pool.checkpoint(runtime)
                    print(json.dumps({"session_id": session_id, "seconds": round(seconds, 1), "frames": runtime["processed_frames"],
                                      "valid": runtime["valid_upper_body_poses"], "retained": pool.counts["retained"], "representatives": len(pool.representatives)}), flush=True)
    except KeyboardInterrupt:
        stopped = "OPERATOR_STOP"
    except PoolCapacityReached as exc:
        stopped = str(exc)
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        stopped = "ERROR_" + type(exc).__name__  # never log credential-bearing exception text
    finally:
        if started is not None:
            runtime["elapsed_seconds"] = time.perf_counter() - started
        if source is not None:
            source.close()
            runtime["source_reconnects"] = source.health().get("reconnect_count", 0)
        if stage is not None:
            stage.close()
        runtime.update(unique_tracks=len(seen), rejection_reasons=dict(rejection_reasons), pose_states=dict(pose_states), source_closed=source is None or source.health().get("source_status") == SOURCE_CLOSED)
        unchanged = before == model_hashes()
        runtime["frozen_models_unchanged"] = unchanged
        pool.close(runtime, stopped if unchanged else "FROZEN_MODEL_HASH_MISMATCH")
        print(json.dumps({"session_directory": str(pool.directory), "stopped_reason": pool.manifest["stopped_reason"], "runtime": runtime, "sample_counts": dict(pool.counts), "representative_samples": len(pool.representatives)}, indent=2), flush=True)
    return pool.manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-minutes", type=float, default=10)
    parser.add_argument("--output-root", default="outputs/learning_pools/CCTV_LEARNING_POOL_V1")
    parser.add_argument("--max-pool-mib", type=float, default=512)
    parser.add_argument("--max-files", type=int, default=64)
    parser.add_argument("--min-free-mib", type=float, default=512)
    parser.add_argument("--max-representatives", type=int, default=512)
    parser.add_argument("--max-reconnects", type=int, default=5)
    args = parser.parse_args()
    if not all(math.isfinite(x) for x in (args.duration_minutes, args.max_pool_mib, args.min_free_mib)) or not 0 < args.duration_minutes <= 1440 or args.max_reconnects < 0:
        parser.error("finite duration required: 0 < minutes <= 1440")
    try:
        manifest = run(args)
    except (RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"stopped_reason": "PREFLIGHT_" + type(exc).__name__}))
        return 2
    return 0 if manifest["stopped_reason"] in {"DURATION_COMPLETED", "OPERATOR_STOP"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
