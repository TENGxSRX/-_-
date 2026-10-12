"""Unified upper-body demo; the P001 five-class model is explicit opt-in."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import replace
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
# Reuse the already-installed YOLO environment; no dependency installation.
sys.path.append(str(ROOT / ".venv_yolo/Lib/site-packages"))
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import cv2
import numpy as np

from src.posture_ai.detection.yolo_optional import UltralyticsPersonTracker
from src.posture_ai.features.upper_body_posture_v2 import CORE_NAMES, SCHEMA
from src.posture_ai.inference.latest_frame_runtime import LatestFrameInput
from src.posture_ai.inference.pose_v2_stage import MediaPipeCropPoseBackend, PoseV2Stage
from src.posture_ai.inference.upper_body_posture_v1 import UpperBodyPostureRunner, verify_v4
from src.posture_ai.inference.upper_body_posture_v2 import UpperBodyPostureV2Runner, evidence_xyz
from src.posture_ai.inference.video_source import VideoSource, SOURCE_CLOSED, SOURCE_FAILED, SOURCE_FILE
from src.posture_ai.inference.webcam_fiveclass_exp import (
    LABELS_TH, MODEL_ID, WebcamFiveClassExperimentalRunner,
)
from src.posture_ai.monitoring.long_session import CandidateSelector, CandidateSelectorConfig

EXPERIMENTAL_MODEL = "webcam-p001-fiveclass-exp-v1"
UPPER_BODY_V2_MODEL = "upper-body-v2"
UPPER_BODY_CONNECTIONS = ((0, 11), (0, 12), (11, 12), (11, 13), (13, 15),
                          (12, 14), (14, 16), (11, 23), (12, 24), (23, 24))


def paired_predictions(evidence, v1, v2):
    xyz = evidence_xyz(evidence)
    hip_count = int(np.isfinite(xyz[3:]).all(axis=1).sum())
    hip_state = ("both_hidden", "partly_hidden", "visible")[hip_count]
    # Reconstruct V1's original four-point core contract on the SAME pose.
    v1_valid = np.isfinite(xyz[1:]).all()
    state = evidence.pose_state
    if evidence.world_landmarks_xyz is not None and state in ("POSE_OK", "POSE_CORE_INVALID"):
        state = "POSE_OK" if v1_valid else "POSE_CORE_INVALID"
    p1 = v1.predict(replace(evidence, core_valid=bool(v1_valid), pose_state=state))
    if state == "POSE_CORE_INVALID" and hip_count < 2 and np.isfinite(xyz[1:3]).all():
        p1 = {"output": "UNKNOWN", "subtype": None, "reason": "MISSING_HIPS"}
    return hip_state, p1, v2.predict(evidence)


def frozen_hashes():
    paths = sorted((ROOT / "outputs/models").glob("upper_body_posture_v1*/*"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()} | {"v4": verify_v4()}


def resolve_source(source: str, camera_index: int) -> str:
    """Resolve the friendly webcam alias without opening any hardware."""
    return str(camera_index) if source.strip().lower() == "webcam" else source


def profile_for_source(source: str, profile: str) -> str | None:
    value = source.strip()
    return profile if value.lower() in ("rtsp", "hikvision", "camera") or value.upper().startswith("HIKVISION_") else None


def hit_test_track(boxes, x: int, y: int, *, display_size: tuple[int, int],
                   inference_size: tuple[int, int]) -> int | None:
    display_width, display_height = display_size
    inference_width, inference_height = inference_size
    if min(display_width, display_height, inference_width, inference_height) <= 0:
        return None
    frame_x = x * inference_width / display_width
    frame_y = y * inference_height / display_height
    matched = [box for box in boxes if box.track_id is not None
               and box.x1 <= frame_x <= box.x2 and box.y1 <= frame_y <= box.y2]
    target = max(matched, key=lambda box: box.confidence, default=None)
    return None if target is None else int(target.track_id)


def select_track_at(current_track_id: int | None, boxes, x: int, y: int, *,
                    display_size: tuple[int, int], inference_size: tuple[int, int]) -> int | None:
    selected = hit_test_track(boxes, x, y, display_size=display_size, inference_size=inference_size)
    return current_track_id if selected is None else selected


def selection_status_text(selected_track_id: int | None, boxes) -> str:
    if selected_track_id is None:
        return "NO TARGET SELECTED - CLICK A PERSON"
    visible_ids = {int(box.track_id) for box in boxes if box.track_id is not None}
    return (f"SELECTED: TRACK {selected_track_id}" if selected_track_id in visible_ids
            else f"TARGET LOST - TRACK {selected_track_id}")


def pose_boxes_for_model(boxes, chosen_ids: set[int], selected_track_id: int | None, *, experimental: bool):
    eligible = [box for box in boxes if box.track_id in chosen_ids]
    return [box for box in eligible if box.track_id == selected_track_id] if experimental else eligible


def invalidate_lost_target(selected_track_id: int | None, visible_ids: set[int], *caches: dict) -> bool:
    lost = selected_track_id is not None and selected_track_id not in visible_ids
    if lost:
        for cache in caches:
            cache.pop(selected_track_id, None)
    return lost


def pose_points_on_frame(evidence, *, frame_size: tuple[int, int]) -> list[tuple[int, int] | None]:
    """Map normalized crop landmarks back to the resized inference/display frame."""
    width, height = frame_size
    if evidence.pose_state != "POSE_OK" or evidence.image_landmarks_xyz is None:
        return []
    left, top, right, bottom = evidence.bbox
    crop_width, crop_height = right - left, bottom - top
    if min(width, height, crop_width, crop_height) <= 0:
        return []
    points = []
    for point in evidence.image_landmarks_xyz:
        if len(point) < 2 or not np.isfinite(point[:2]).all():
            points.append(None)
            continue
        x, y = left + float(point[0]) * crop_width, top + float(point[1]) * crop_height
        points.append((round(x), round(y)) if 0 <= x < width and 0 <= y < height else None)
    return points


def draw_upper_body_skeleton(frame, evidence) -> None:
    points = pose_points_on_frame(evidence, frame_size=(frame.shape[1], frame.shape[0]))
    if not points:
        return
    for first, second in UPPER_BODY_CONNECTIONS:
        if first < len(points) and second < len(points) and points[first] is not None and points[second] is not None:
            cv2.line(frame, points[first], points[second], (255, 120, 0), 2, cv2.LINE_AA)
    for index in {value for pair in UPPER_BODY_CONNECTIONS for value in pair}:
        if index < len(points) and points[index] is not None:
            cv2.circle(frame, points[index], 4, (0, 255, 255), -1, cv2.LINE_AA)


def _display_prediction(prediction: dict | None) -> str:
    if not prediction:
        return "POSTURE: WAITING"
    label = str(prediction.get("output", "UNKNOWN"))
    confidence = prediction.get("confidence")
    confidence_text = "" if confidence is None else f" {float(confidence):.0%}"
    return f"{label} - {LABELS_TH.get(label, LABELS_TH['UNKNOWN'])}{confidence_text}"


def draw_unicode_lines(frame, lines: list[tuple[str, tuple[int, int], tuple[int, int, int], int]]) -> None:
    """Use already-installed Pillow because OpenCV's built-in font cannot render Thai."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    fonts = {}
    for text, position, bgr, size in lines:
        font = fonts.setdefault(size, ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", size))
        draw.text(position, text, font=font, fill=(bgr[2], bgr[1], bgr[0]), stroke_width=1, stroke_fill=(0, 0, 0))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="rtsp", help="webcam, local video path, rtsp alias, or HIKVISION profile")
    parser.add_argument("--camera-index", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--camera-width", type=int, default=1280, help="Best-effort USB capture width")
    parser.add_argument("--camera-height", type=int, default=720, help="Best-effort USB capture height")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="Best-effort USB capture FPS")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default="HIKVISION_MAIN")
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--posture-model", choices=(UPPER_BODY_V2_MODEL, EXPERIMENTAL_MODEL), default=UPPER_BODY_V2_MODEL)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 120:
        parser.error("duration must be 0 < seconds <= 120")
    before = frozen_hashes()
    experimental = args.posture_model == EXPERIMENTAL_MODEL
    model_dir = args.model_dir or ("outputs/models/webcam_p001_fiveclass_exp_v1" if experimental else "outputs/models/upper_body_posture_v2")
    if experimental:
        v1 = v2 = None
        fiveclass = WebcamFiveClassExperimentalRunner(model_dir)
    else:
        v1, v2 = UpperBodyPostureRunner(), UpperBodyPostureV2Runner(model_dir)
        fiveclass = None
    output_root = args.output_root or ("outputs/webcam_p001_fiveclass_demo_sessions" if experimental else "outputs/upper_body_v2_sessions")
    session = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    session.mkdir(parents=True, exist_ok=False)
    source_value = resolve_source(args.source, args.camera_index)
    video = VideoSource(
        source_value,
        profile=profile_for_source(source_value, args.profile),
        config_path="configs/rtsp_profiles.yaml",
        low_latency=True,
        reconnect=True,
        max_reconnects=5 if profile_for_source(source_value, args.profile) else 0,
        preferred_width=args.camera_width,
        preferred_height=args.camera_height,
        preferred_fps=args.camera_fps,
    )
    source = LatestFrameInput(video, asynchronous=video.source_type == "RTSP")
    tracker = stage = None
    totals = {version: Counter(dict.fromkeys(("CORRECT_SITTING", "INCORRECT_SITTING", "UNKNOWN", "OOD", "missing_hips", "other_missing_core"), 0)) for version in ("v1", "v2")}
    strata = defaultdict(lambda: {v: Counter() for v in totals})
    pose_counts, hip_counts, range_counts = Counter(), Counter(), Counter()
    ids, boxes, cache, pose_cache, pose_status_cache = set(), [], {}, {}, {}
    selected_track_id = None
    display_size = inference_size = (1, 1)
    frames = additional = sampled = valid = maximum_people = 0
    experimental_counts = Counter()
    frame_times = deque(maxlen=60)
    stopped, seconds = "DURATION_COMPLETED", 0.0
    window = "ANYA Live Demo | click person to select target"

    def click(event, x, y, flags, param):
        nonlocal selected_track_id
        if event == cv2.EVENT_LBUTTONDOWN:
            selected = select_track_at(selected_track_id, boxes, x, y,
                                       display_size=display_size, inference_size=inference_size)
            if experimental and selected != selected_track_id:
                cache.clear()
                pose_cache.clear()
                pose_status_cache.clear()
            selected_track_id = selected

    try:
        tracker = UltralyticsPersonTracker("outputs/yolo_tracking/weights/yolo26n.pt", device="0", confidence=0.35, tracker="bytetrack.yaml")
        stage = PoseV2Stage(MediaPipeCropPoseBackend("metadata/pose_landmarker_heavy.task", core_landmark_names=CORE_NAMES), max_pose_per_second=2)
        selector = CandidateSelector(CandidateSelectorConfig(representative_interval_seconds=0.5, event_cooldown_seconds=0.5))
        if not source.open():
            stopped = "SOURCE_OPEN_FAILED"
        else:
            if not args.no_display:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(window, click)
            started = time.perf_counter()
            log_name = "fiveclass_predictions.jsonl" if experimental else "paired_predictions.jsonl"
            with (session / log_name).open("x", encoding="utf-8") as log:
                while time.perf_counter() - started < args.max_seconds:
                    frame_started = time.perf_counter()
                    read = source.read()
                    if not read.ok or read.frame is None:
                        if read.state in (SOURCE_CLOSED, SOURCE_FAILED):
                            stopped = "END_OF_STREAM" if video.source_type == SOURCE_FILE and read.state == SOURCE_CLOSED else "SOURCE_FAILED"
                            break
                        continue
                    frame = read.frame
                    if frame.shape[1] > 960:
                        frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
                    inference_size = display_size = (frame.shape[1], frame.shape[0])
                    boxes = tracker.track(frame)
                    seconds = time.perf_counter() - started
                    now_ids = {int(b.track_id) for b in boxes if b.track_id is not None}
                    invalidate_lost_target(selected_track_id, now_ids, cache, pose_cache, pose_status_cache)
                    maximum_people = max(maximum_people, len(now_ids))
                    chosen = selector.inspect(boxes, timestamp_seconds=seconds, frame_width=frame.shape[1], frame_height=frame.shape[0], new_ids=now_ids - ids, reacquired_ids=set())
                    ids.update(now_ids)
                    chosen_ids = {r["anonymous_track_id"] for r in chosen}
                    pose_boxes = pose_boxes_for_model(boxes, chosen_ids, selected_track_id, experimental=experimental)
                    records = stage.process(frame, pose_boxes, seconds, all_boxes=boxes)
                    for evidence in records:
                        pose_counts[evidence.pose_state] += 1
                        pose_status_cache[evidence.track_id] = (seconds, evidence.pose_state, evidence.detail)
                        # Pending/overlap records are not pose attempts; do not inflate rates.
                        if evidence.sampling_reason != "SAMPLED":
                            cache.pop(evidence.track_id, None)
                            pose_cache.pop(evidence.track_id, None)
                            continue
                        sampled += 1
                        valid += int(evidence.pose_state == "POSE_OK")
                        if evidence.pose_state == "POSE_OK":
                            pose_cache[evidence.track_id] = (seconds, evidence)
                        else:
                            pose_cache.pop(evidence.track_id, None)
                        if experimental:
                            prediction = fiveclass.predict(evidence)
                            experimental_counts[prediction["output"]] += 1
                            experimental_counts["reason:" + prediction["reason"]] += 1
                            cache[evidence.track_id] = (seconds, prediction)
                            log.write(json.dumps({
                                "schema": "anya_live_demo_v1", "record_type": "MODEL_PREDICTION_DIAGNOSTIC",
                                "label_source": "MODEL_PREDICTION", "seconds": seconds,
                                "track_id": evidence.track_id, "pose_state": evidence.pose_state,
                                "prediction": prediction, "selection_status": "SELECTED",
                                "model_id": MODEL_ID, "experimental": True,
                                "raw_video_recorded": False, "raw_images_saved": False,
                            }, ensure_ascii=False) + "\n")
                            continue
                        hips, p1, p2 = paired_predictions(evidence, v1, v2)
                        if evidence.world_landmarks_xyz is None:
                            hips = "unobservable_no_pose"
                        hip_counts[hips] += 1
                        fraction = (evidence.bbox[3] - evidence.bbox[1]) / frame.shape[0]
                        distance = "near" if fraction >= 0.5 else "mid" if fraction >= 0.25 else "far"
                        range_counts[distance] += 1
                        additional += int(hips in ("partly_hidden", "both_hidden") and p1["output"] == "UNKNOWN" and p2["output"] != "UNKNOWN")
                        products = {}
                        for version, prediction in (("v1", p1), ("v2", p2)):
                            counts = totals[version]
                            counts[prediction["output"]] += 1
                            counts["OOD"] += int("OOD" in prediction["reason"])
                            counts["missing_hips"] += int(prediction["reason"] == "MISSING_HIPS")
                            counts["other_missing_core"] += int(prediction["reason"] in ("POSE_CORE_INVALID", "INVALID_NOSE_OR_SHOULDERS", "INVALID_SHOULDERS_OR_HIPS", "LOW_CORE_VISIBILITY"))
                            counts["reason:" + prediction["reason"]] += 1
                            for key in ("hips:" + hips, "range:" + distance):
                                strata[key][version][prediction["output"]] += 1
                                strata[key][version]["OOD"] += int("OOD" in prediction["reason"])
                            products[version] = prediction if evidence.track_id == selected_track_id else {"output": "UNKNOWN", "subtype": None, "reason": "TARGET_NOT_SELECTED"}
                            counts["product:" + products[version]["output"]] += 1
                        cache[evidence.track_id] = (seconds, p2)
                        log.write(json.dumps({"schema": SCHEMA, "record_type": "MODEL_PREDICTION_DIAGNOSTIC", "label_source": "MODEL_PREDICTION",
                                              "seconds": seconds, "track_id": evidence.track_id, "pose_state": evidence.pose_state,
                                              "hip_visibility_proxy": hips, "distance_bbox_proxy": distance,
                                              "v1_raw": p1, "v2_raw": p2, "product": products,
                                              "selection_status": "SELECTED" if evidence.track_id == selected_track_id else "NOT_SELECTED",
                                              "raw_video_recorded": False}) + "\n")
                    frames += 1
                    if not args.no_display:
                        processing_ms = (time.perf_counter() - frame_started) * 1000.0
                        frame_times.append(max(time.perf_counter() - frame_started, 1e-9))
                        fps = len(frame_times) / sum(frame_times)
                        unicode_lines = []
                        for box in boxes:
                            cached = cache.get(box.track_id)
                            selected = box.track_id == selected_track_id
                            fresh_prediction = cached[1] if cached and seconds - cached[0] <= 1 else None
                            status = pose_status_cache.get(box.track_id)
                            pose_text = "POSE: WAITING"
                            if status and seconds - status[0] <= 1:
                                pose_text = "POSE: VALID" if status[1] == "POSE_OK" else f"POSE: {status[1]}"
                            elif status:
                                pose_text = "POSE: STALE"
                            color, thickness = ((0, 255, 0), 3) if selected else ((0, 220, 220), 2)
                            cv2.rectangle(frame, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), color, thickness)
                            if selected:
                                pose_item = pose_cache.get(box.track_id)
                                if pose_item and seconds - pose_item[0] <= 1:
                                    draw_upper_body_skeleton(frame, pose_item[1])
                                unicode_lines.append((f"TRACK {box.track_id} | {pose_text}", (round(box.x1), max(52, round(box.y1) - 48)), color, 20))
                                unicode_lines.append((_display_prediction(fresh_prediction), (round(box.x1), max(76, round(box.y1) - 24)), color, 22))
                            else:
                                unicode_lines.append((f"TRACK {box.track_id} | NOT SELECTED", (round(box.x1), max(48, round(box.y1) - 24)), color, 18))
                        selection_text = selection_status_text(selected_track_id, boxes)
                        selection_color = ((0, 255, 0) if selection_text.startswith("SELECTED") else
                                           (0, 0, 255) if selection_text.startswith("TARGET LOST") else (255, 255, 255))
                        unicode_lines.extend([
                            ("ANYA LIVE DEMO | EXPERIMENTAL", (10, 8), (255, 255, 255), 24),
                            (f"MODEL: {MODEL_ID if experimental else UPPER_BODY_V2_MODEL}", (10, 38), (0, 220, 255), 20),
                            (f"PERSON DETECTED: {len(boxes)} | {selection_text}", (10, 64), selection_color, 22),
                            (f"FPS {fps:.1f} | PROCESSING {processing_ms:.1f} ms | Q/ESC EXIT", (10, 92), (255, 255, 255), 18),
                        ])
                        draw_unicode_lines(frame, unicode_lines)
                        cv2.imshow(window, frame)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            stopped = "OPERATOR_STOP"
                            break
                seconds = time.perf_counter() - started
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        stopped = "ERROR_" + type(exc).__name__
    finally:
        source.close()
        if stage is not None:
            stage.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        assert before == frozen_hashes()
        summary = {"schema": SCHEMA, "stopped_reason": stopped, "elapsed_seconds": seconds, "processed_frames": frames,
                   "source": video.spec.to_dict(), "source_health": video.health(),
                   "pose_attempts": sampled, "valid_pose_attempts": valid, "pose_counts": dict(pose_counts),
                   "unique_track_ids": len(ids), "maximum_simultaneous_people": maximum_people,
                   "hip_visibility_proxy": dict(hip_counts), "distance_bbox_proxy": dict(range_counts),
                   "models": ({MODEL_ID: dict(experimental_counts)} if experimental else totals),
                   "strata": dict(strata), "additional_hidden_hip_cases_handled": additional,
                   "rates": ({"unknown": experimental_counts["UNKNOWN"] / sampled if sampled else None}
                             if experimental else {v: {"unknown": c["UNKNOWN"] / sampled if sampled else None, "ood": c["OOD"] / sampled if sampled else None} for v, c in totals.items()}),
                   "seated_context": "OPERATOR_CONFIRMATION_OR_UNKNOWN", "rejection_policy": "EXPERIMENTAL_UNCHANGED_RANGE_AND_UNANIMITY",
                   "limitations": ["No human GT or accuracy measurement", "Hip visibility is a MediaPipe confidence proxy, not verified physical desk occlusion", "Distance is bbox fraction, not calibrated depth", "Counts are correlated attempts, not independent people"],
                   "posture_model": MODEL_ID if experimental else UPPER_BODY_V2_MODEL,
                   "experimental_fiveclass_counts": dict(experimental_counts),
                   "frozen_hashes": before, "v4_called": False, "raw_video_recorded": False, "raw_images_saved": False}
        (session / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"session": str(session), **summary}, indent=2))
    return 0 if stopped in ("DURATION_COMPLETED", "OPERATOR_STOP", "END_OF_STREAM") else 2


if __name__ == "__main__":
    raise SystemExit(main())
