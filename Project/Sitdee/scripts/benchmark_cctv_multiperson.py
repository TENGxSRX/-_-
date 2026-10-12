"""Short read-only CCTV multi-person benchmark for the existing RTSP path.

The benchmark measures detector/runtime capacity only. It never records video,
collects labels, trains a model, or treats current V4 predictions as truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.multiperson.demo import run_multiperson_demo  # noqa: E402
from src.posture_ai.multiperson.runtime import PERFORMANCE_PRESETS  # noqa: E402
from src.posture_ai.inference.video_source import SourceConfigurationError  # noqa: E402


def _one_line(result: dict) -> dict:
    health = result.get("source_health") or {}
    return {
        "max_people": result.get("max_people"),
        "status": result.get("status"),
        "hardware_status": result.get("hardware_status"),
        "source_status": result.get("source_status"),
        "resolution": health.get("resolution"),
        "source_fps": result.get("source_fps"),
        "inference_fps": result.get("inference_fps"),
        "frame_age_ms": result.get("frame_age_ms"),
        "inference_time_ms": result.get("inference_time_ms"),
        "frames_captured": result.get("frames_captured"),
        "frames_inferred": result.get("frames_inferred"),
        "frames_intentionally_dropped": result.get("frames_intentionally_dropped"),
        "stale_events": result.get("stale_events"),
        "decode_failures": health.get("decode_failures"),
        "reconnects": health.get("reconnect_count"),
        "detected_people": result.get("detected_people"),
        "valid_core_poses": result.get("valid_core_poses"),
        "rejected_poses": result.get("rejected_poses"),
        "poses_detected_rate": result.get("poses_detected_rate"),
        "core_valid_rate": result.get("core_valid_rate"),
        "core_quality": result.get("core_quality"),
        "tracking_events": result.get("tracking_events"),
        "activity_events": result.get("activity_events"),
        "activity_state_distribution": result.get("activity_state_distribution"),
        "tracking_state_distribution": result.get("tracking_state_distribution"),
        "active_track_count": result.get("active_track_count"),
        "anonymous_person_ids_seen": result.get("anonymous_person_ids_seen"),
        "id_fragmentation_proxy": result.get("id_fragmentation_proxy"),
        "person_count_accuracy": "NOT_MEASURED_NO_MANUAL_GROUND_TRUTH",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Short read-only CCTV multi-person RTSP benchmark")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default="HIKVISION_MAIN")
    parser.add_argument("--people", nargs="+", type=int, choices=(1, 2, 3, 4), default=(1, 2, 3, 4), help="max_people values to test")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--preset", choices=sorted(PERFORMANCE_PRESETS), default="RTSP_CCTV")
    parser.add_argument("--max-reconnects", type=int, default=3)
    parser.add_argument("--output-root", default="outputs/cctv_multiperson_benchmark")
    parser.add_argument("--json", default="outputs/metrics/cctv_multiperson_benchmark.json")
    parser.add_argument("--report", default="reports/cctv_multiperson_benchmark_th.md")
    args = parser.parse_args()
    if args.seconds <= 0 or args.max_reconnects < 0:
        parser.error("--seconds must be positive and --max-reconnects must not be negative")

    process = psutil.Process(os.getpid())
    before_rss = int(process.memory_info().rss)
    results: list[dict] = []
    failures: list[dict] = []
    for max_people in args.people:
        try:
            result = run_multiperson_demo(
                "rtsp",
                max_people=max_people,
                performance_preset=args.preset,
                profile=args.profile,
                max_seconds=args.seconds,
                max_reconnects=args.max_reconnects,
                display=False,
                telemetry_path=None,
                latest_frame_only=True,
            )
            results.append(_one_line(result))
        except (SourceConfigurationError, FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            failures.append({"max_people": max_people, "status": "FAILED", "detail": str(exc)})
    after_rss = int(process.memory_info().rss)

    observed_people = max((int((item.get("detected_people") or {}).get("max") or 0) for item in results), default=0)
    if not results:
        status = "NOT_RUN"
    elif observed_people == 0:
        status = "REAL_MULTIPERSON_DATA_NOT_AVAILABLE"
    else:
        status = "REAL_MULTIPERSON_SOURCE_RUN_COMPLETED_BUT_NOT_GROUND_TRUTH_VALIDATED"
    payload = {
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "preset": args.preset,
        "seconds_per_setting": args.seconds,
        "results": results,
        "failures": failures,
        "observed_max_people": observed_people,
        "person_count_accuracy": "NOT_MEASURED_NO_MANUAL_GROUND_TRUTH",
        "cpu_ram_measurement": {
            "process_rss_before_bytes": before_rss,
            "process_rss_after_bytes": after_rss,
            "system_cpu_percent_at_end": psutil.cpu_percent(interval=0.1),
            "system_available_ram_bytes_at_end": int(psutil.virtual_memory().available),
        },
        "raw_video_recorded": False,
        "labels_collected": False,
        "training_performed": False,
        "interpretation": "Engineering capacity telemetry only. Detection, core quality, posture classification and track stability are separate; no accuracy, room or medical claim.",
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    lines = [
        "# CCTV multi-person benchmark",
        "",
        f"สถานะ: **{status}**",
        f"Profile: `{args.profile}` | Preset: `{args.preset}` | ประมาณ {args.seconds:.1f} วินาทีต่อค่า max_people",
        "",
        "การทดสอบนี้ใช้ RTSP latest-frame แบบ read-only ไม่บันทึกวิดีโอ ไม่เก็บ label และไม่ train โมเดล",
        "",
        "| max_people | Source FPS | AI FPS | Frame age median | Inference median | Captured | Inferred | Dropped old | Max detected | Max core valid | Core OK | Reconnects |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        age = (item.get("frame_age_ms") or {}).get("median_ms")
        infer = (item.get("inference_time_ms") or {}).get("median_ms")
        detected = item.get("detected_people") or {}
        valid = item.get("valid_core_poses") or {}
        quality = item.get("core_quality") or {}
        resolution = item.get("resolution") or {}
        lines.append(
            f"| {item.get('max_people')} | {float(item.get('source_fps') or 0):.2f} | {float(item.get('inference_fps') or 0):.2f} | "
            f"{('--' if age is None else f'{float(age):.1f} ms')} | {('--' if infer is None else f'{float(infer):.1f} ms')} | "
            f"{int(item.get('frames_captured') or 0)} | {int(item.get('frames_inferred') or 0)} | {int(item.get('frames_intentionally_dropped') or 0)} | "
            f"{int(detected.get('max') or 0)} | {int(valid.get('max') or 0)} | {float(quality.get('core_ok_rate') or 0) * 100:.1f}% | {int(item.get('reconnects') or 0)} |"
        )
    lines.extend(["", "## Tracking และ activity diagnostics", ""])
    for item in results:
        lines.extend([
            f"### max_people={item.get('max_people')}",
            f"- active tracks: `{item.get('active_track_count')}`",
            f"- anonymous IDs seen: `{item.get('anonymous_person_ids_seen')}`",
            f"- lifecycle events: `{item.get('tracking_events')}`",
            f"- activity events: `{item.get('activity_events')}`",
            f"- activity state distribution: `{item.get('activity_state_distribution')}`",
            f"- tracking state distribution: `{item.get('tracking_state_distribution')}`",
            f"- fragmentation proxy: `{item.get('id_fragmentation_proxy')}`",
            "",
        ])
    if failures:
        lines.extend(["", "## ทดสอบที่ไม่สำเร็จ", ""])
        lines.extend(f"- max_people={item['max_people']}: {item['detail']}" for item in failures)
    lines.extend([
        "",
        "## ข้อจำกัด",
        "",
        "ยังไม่มี manual ground truth สำหรับ expected_people_count จึงไม่คำนวณ person-count accuracy, missed-person rate หรือ IDF1. หากไม่มีคนนั่งในภาพ สถานะจะเป็น `REAL_MULTIPERSON_DATA_NOT_AVAILABLE`.",
        "การเพิ่ม `max_people` เป็นการตั้งเพดานของ MediaPipe ไม่ใช่หลักฐานว่ากล้องตรวจพบคนจำนวนดังกล่าวจริง.",
        "",
        f"JSON: `{json_path}`",
    ])
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(main())
