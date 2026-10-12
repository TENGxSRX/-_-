"""Offline, stage-segmented YOLO/ByteTrack and MediaPipe compute benchmark.

No source camera, model classifier, ground truth, or raw frames are written.
Duplicated crops are explicitly synthetic compute load, not multi-person data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import psutil
import torch
import ultralytics
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.detection.yolo_optional import UltralyticsPersonTracker, crop_frame  # noqa: E402
from src.posture_ai.inference.live_posture_demo import padded_box  # noqa: E402


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "median": float(np.median(array)), "p95": float(np.percentile(array, 95))}


def load_frames(video: Path, count: int, width: int) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("local video could not be opened")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    try:
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok:
                break
            if width > 0 and frame.shape[1] > width:
                scale = width / frame.shape[1]
                frame = cv2.resize(frame, (width, max(1, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) < count:
        raise RuntimeError("local video has fewer decoded frames than requested")
    return frames, float(fps)


def synchronized_call(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def make_tracker(weights: Path) -> UltralyticsPersonTracker:
    return UltralyticsPersonTracker(weights, device="0", confidence=0.35, iou=0.70, tracker="bytetrack.yaml")


def make_pose(model: Path, *, gpu: bool = False):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model), delegate=BaseOptions.Delegate.GPU if gpu else BaseOptions.Delegate.CPU),
        running_mode=RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def pose_once(landmarker, crop: np.ndarray) -> bool:
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return bool(result.pose_landmarks and result.pose_world_landmarks)


def synthetic_crops(crop: np.ndarray, count: int) -> list[np.ndarray]:
    if count < 1:
        raise ValueError("crop count must be positive")
    return [crop] * count


def sample_due(frame_index: int, source_fps: float, last_sample_seconds: float, rate_hz: float) -> bool:
    """Offline sampling schedule uses video time, not faster/slower playback time."""
    if rate_hz <= 0 or source_fps <= 0:
        raise ValueError("rates must be positive")
    return frame_index / source_fps - last_sample_seconds >= 1.0 / rate_hz - 1e-9


def gpu_probe(model: Path) -> dict:
    """Isolate optional GPU delegate errors/crashes from the CPU benchmark."""
    probe = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--gpu-probe-child", "--pose-model", str(model)],
        capture_output=True, text=True, timeout=25, check=False,
    )
    return {"available": probe.returncode == 0, "reason": "GPU_PROBE_SUCCESS" if probe.returncode == 0 else "GPU_DELEGATE_UNAVAILABLE_OR_FAILED"}


def report_markdown(data: dict) -> str:
    baseline = data["yolo_bytetrack"]
    lines = ["# Offline stage-segmented pose benchmark", "", "MEASURED — local recorded video only; no camera/RTSP, no posture accuracy.",
             f"Video: `{data['video']}`; frames/run: {data['frames_per_run']}; runs: {data['repeats']}",
             f"Environment: `{data['environment']['python']}`; GPU: {data['environment']['gpu']}", "",
             "## YOLO + ByteTrack (measured)",
             f"Steady FPS: {baseline['fps']}; stage ms/frame: {baseline['latency_ms']}; peak allocated VRAM MiB: {baseline['peak_vram_mib']}",
             "", "## MediaPipe CPU (measured)",
             f"GPU delegate: {data['mediapipe_gpu_delegate']['reason']}",
             "All N>1 results duplicate one valid crop as SYNTHETIC_COMPUTE_LOAD, not multi-person accuracy."]
    for count, item in data["pose_event_scaling"].items():
        lines.append(f"N={count}: event ms {item['event_ms']}; ms/crop {item['per_crop_ms']}; pose successes {item['successes']}/{item['attempts']}")
    lines.extend(["", "## Combined offline replay (measured)", "Sampling is scheduled on video timestamps; throughput is measured wall-clock playback speed, not network latency."])
    lines.append("The default 90-frame window is about 3 seconds of video time: low-rate cases have only 2-3 sample events. Per-frame p95 can therefore hide rare long pose events; inspect pose-event p95 and total-frame maximum as well.")
    for key, item in data["combined"].items():
        lines.append(f"{key}: FPS {item['fps']}; YOLO ms {item['yolo_ms']}; Pose ms/event {item['pose_event_ms']}; total ms/frame {item['total_ms']}; total max {item['total_max_ms']:.1f} ms; attempts {item['pose_attempts']}")
    lines.extend(["", f"Initial measured-budget recommendation: at most {data['recommendation']['max_pose_per_second']} pose crops/second globally (experimental, not production default).",
                  f"Reason: {data['recommendation']['reason']}",
                  "No V4/posture classification or labels were used.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline YOLO/ByteTrack + MediaPipe stage benchmark")
    parser.add_argument("--video", default="datasets/raw/rgb_native/drivenet_v6/source_videos/SS_1.mp4")
    parser.add_argument("--weights", default="outputs/yolo_tracking/weights/yolo26n.pt")
    parser.add_argument("--pose-model", default="metadata/pose_landmarker_heavy.task")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--pose-events", type=int, default=8)
    parser.add_argument("--output", default="outputs/metrics/offline_pose_stage_benchmark.json")
    parser.add_argument("--report", default="reports/offline_pose_stage_benchmark.md")
    parser.add_argument("--gpu-probe-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gpu_probe_child:
        try:
            landmarker = make_pose(Path(args.pose_model), gpu=True)
            try:
                pose_once(landmarker, np.zeros((256, 256, 3), dtype=np.uint8))
            finally:
                landmarker.close()
            return 0
        except Exception:
            return 2
    if args.frames <= args.warmup or args.repeats < 1 or args.pose_events < 2:
        parser.error("frames must exceed warmup; repeats >= 1; pose-events >= 2")

    video, weights, pose_model = Path(args.video), Path(args.weights), Path(args.pose_model)
    frames, source_fps = load_frames(video, args.frames + args.warmup, args.width)
    process = psutil.Process()
    torch.cuda.reset_peak_memory_stats()
    runs = []
    all_yolo_latencies = []
    example_crop = None
    for _ in range(args.repeats):
        tracker = make_tracker(weights)
        for frame in frames[:args.warmup]:
            tracker.track(frame)
        latencies = []
        started = time.perf_counter()
        for frame in frames[args.warmup:]:
            boxes, ms = synchronized_call(lambda f=frame: tracker.track(f))
            latencies.append(ms)
            if example_crop is None and boxes:
                box = max(boxes, key=lambda item: item.confidence)
                example_crop, _ = crop_frame(frame, padded_box(box, 0.10))
        wall_seconds = time.perf_counter() - started
        all_yolo_latencies.extend(latencies)
        runs.append({"frames": args.frames, "fps": args.frames / wall_seconds, "latency_ms": stats(latencies)})
    if example_crop is None:
        raise RuntimeError("detector returned no person crop in recorded frames")
    peak_vram_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    pose = make_pose(pose_model)
    try:
        for _ in range(5):
            pose_once(pose, example_crop)
        scaling = {}
        for n in (1, 2, 4, 8):
            event_times, crop_times, successes = [], [], 0
            cpu_start = process.cpu_times()
            wall_start = time.perf_counter()
            for _ in range(args.pose_events):
                event_start = time.perf_counter()
                for crop in synthetic_crops(example_crop, n):
                    ok, ms = synchronized_call(lambda c=crop: pose_once(pose, c))
                    crop_times.append(ms)
                    successes += int(ok)
                event_times.append((time.perf_counter() - event_start) * 1000.0)
            cpu_end = process.cpu_times()
            wall = time.perf_counter() - wall_start
            scaling[str(n)] = {
                "load_type": "REAL_CROP" if n == 1 else "SYNTHETIC_COMPUTE_LOAD",
                "events": args.pose_events, "attempts": n * args.pose_events, "successes": successes,
                "per_crop_ms": stats(crop_times), "event_ms": stats(event_times),
                "process_cpu_core_equivalents": (cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system) / wall,
            }

        combined = {}
        for n in (1, 2, 4, 8):
            for rate in (0.5, 1.0, 2.0):
                tracker = make_tracker(weights)
                for frame in frames[:args.warmup]:
                    tracker.track(frame)
                stage_yolo, stage_pose, stage_total, stage_overhead = [], [], [], []
                attempts = successes = 0
                last_sample = -1e12
                cpu_start = process.cpu_times()
                started = time.perf_counter()
                for index, frame in enumerate(frames[args.warmup:]):
                    total_start = time.perf_counter()
                    boxes, yolo_ms = synchronized_call(lambda f=frame: tracker.track(f))
                    pose_ms = 0.0
                    if boxes and sample_due(index, source_fps, last_sample, rate):
                        last_sample = index / source_fps
                        box = max(boxes, key=lambda item: item.confidence)
                        crop, _ = crop_frame(frame, padded_box(box, 0.10))
                        if crop.size:
                            pose_start = time.perf_counter()
                            for selected in synthetic_crops(crop, n):
                                successes += int(pose_once(pose, selected))
                                attempts += 1
                            pose_ms = (time.perf_counter() - pose_start) * 1000.0
                            stage_pose.append(pose_ms)
                    total_ms = (time.perf_counter() - total_start) * 1000.0
                    stage_yolo.append(yolo_ms)
                    stage_total.append(total_ms)
                    stage_overhead.append(max(0.0, total_ms - yolo_ms - pose_ms))
                wall = time.perf_counter() - started
                cpu_end = process.cpu_times()
                combined[f"N{n}_rate{rate:g}"] = {
                    "load_type": "REAL_CROP" if n == 1 else "SYNTHETIC_COMPUTE_LOAD",
                    "frames": args.frames, "fps": args.frames / wall,
                    "yolo_ms": stats(stage_yolo), "pose_event_ms": stats(stage_pose),
                    "total_ms": stats(stage_total), "total_max_ms": max(stage_total), "overhead_ms": stats(stage_overhead),
                    "pose_attempts": attempts, "pose_successes": successes,
                    "process_cpu_core_equivalents": (cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system) / wall,
                    "peak_vram_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
                }
    finally:
        pose.close()

    probe = gpu_probe(pose_model)
    baseline_fps = np.median([item["fps"] for item in runs])
    eligible = [
        n * rate for n in (1, 2, 4) for rate in (0.5, 1.0, 2.0)
        if combined[f"N{n}_rate{rate:g}"]["fps"] >= 0.7 * baseline_fps
        and (combined[f"N{n}_rate{rate:g}"]["total_ms"]["p95"] or 0) <= 250.0
    ]
    recommended = min(2.0, max(eligible)) if eligible else 0.5
    data = {
        "schema_version": "offline_pose_stage_benchmark_v1", "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "video": str(video), "source_fps": source_fps, "frames_per_run": args.frames,
        "repeats": args.repeats, "warmup_frames": args.warmup, "width": args.width,
        "environment": {"python": sys.version.split()[0], "executable": sys.executable,
                        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                        "mediapipe": mp.__version__, "opencv": cv2.__version__,
                        "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "yolo_weights": weights.name, "tracker": "bytetrack.yaml", "pose_model": pose_model.name},
        "yolo_bytetrack": {"runs": runs, "fps": stats([item["fps"] for item in runs]),
                           "latency_ms": stats(all_yolo_latencies),
                           "peak_vram_mib": peak_vram_mib, "peak_vram_measure": "torch.cuda.max_memory_allocated"},
        "mediapipe_gpu_delegate": probe,
        "pose_event_scaling": scaling, "combined": combined,
        "recommendation": {"max_pose_per_second": recommended,
                           "reason": "Experimental global start: measured N<=4 cases retaining >=70% of tracking-only median FPS and <=250 ms p95 total frame time; capped at 2 crops/s. Revalidate on real CCTV before production."},
        "raw_video_recorded": False, "posture_accuracy_evaluated": False,
    }
    output, report = Path(args.output), Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    report.write_text(report_markdown(data), encoding="utf-8")
    print(json.dumps({"output": str(output), "report": str(report), "combined_cases": len(combined)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
