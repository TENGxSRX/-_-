"""Short, no-recording MAIN/SUB RTSP AI Lab benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.inference.rtsp_ai_lab import run_rtsp_ai_lab_session  # noqa: E402
from src.posture_ai.inference.video_source import SourceConfigurationError  # noqa: E402


def _one_line(result: dict) -> dict:
    summary = result["summary"]
    timing = summary.get("timing_metrics") or {}
    source_health = summary.get("source_health") or {}
    return {
        "profile": summary.get("source_profile"),
        "status": "PASS" if summary.get("processed_frames", 0) else "NO_INFERENCE",
        "source_status": summary.get("source_status"),
        "resolution": source_health.get("resolution"),
        "source_fps": timing.get("source_fps"),
        "display_fps": timing.get("display_fps"),
        "inference_fps": timing.get("inference_fps"),
        "capture_age_ms": timing.get("capture_age_ms"),
        "inference_time_ms": timing.get("inference_time_ms"),
        "render_time_ms": timing.get("render_time_ms"),
        "frames_captured": timing.get("frames_captured"),
        "frames_inferred": timing.get("frames_inferred"),
        "frames_intentionally_dropped": timing.get("frames_intentionally_dropped"),
        "stale_events": timing.get("stale_events"),
        "decode_failures": source_health.get("decode_failures"),
        "reconnects": source_health.get("reconnect_count"),
        "pose_detection_percentage": summary.get("pose_detection_percentage"),
        "core_landmark_acceptance_percentage": summary.get("core_landmark_acceptance_percentage"),
        "raw_video_recorded": summary.get("raw_video_recorded"),
        "training_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Hikvision MAIN and SUB with frozen V4; no recording or collection")
    parser.add_argument("--profiles", nargs="+", choices=("HIKVISION_MAIN", "HIKVISION_SUB"), default=("HIKVISION_MAIN", "HIKVISION_SUB"))
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--preset", choices=("RTSP_DESKTOP", "RTSP_BALANCED", "RTSP_LOW_LATENCY"), default="RTSP_LOW_LATENCY")
    parser.add_argument("--inference-width", type=int)
    parser.add_argument("--inference-height", type=int)
    parser.add_argument("--max-reconnects", type=int, default=3)
    parser.add_argument("--output-root", default="outputs/rtsp_ai_benchmark")
    parser.add_argument("--report", default="reports/rtsp_ai_lab_main_sub_benchmark_th.md")
    parser.add_argument("--json", default="outputs/metrics/rtsp_ai_lab_main_sub_benchmark.json")
    args = parser.parse_args()
    if args.seconds <= 0 or args.max_reconnects < 0:
        parser.error("--seconds must be positive and --max-reconnects must not be negative")

    results: list[dict] = []
    failures: list[dict] = []
    for profile in args.profiles:
        try:
            result = run_rtsp_ai_lab_session(
                "rtsp",
                profile=profile,
                preset=args.preset,
                max_seconds=args.seconds,
                max_reconnects=args.max_reconnects,
                output_root=Path(args.output_root) / profile,
                display=False,
                inference_width=args.inference_width,
                inference_height=args.inference_height,
            )
            results.append(_one_line(result))
        except (SourceConfigurationError, FileNotFoundError, RuntimeError, ValueError) as exc:
            failures.append({"profile": profile, "status": "FAILED", "detail": str(exc)})

    payload = {
        "status": "PASS" if results and not failures else ("PARTIAL" if results else "NOT_RUN"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "duration_seconds_per_profile": args.seconds,
        "results": results,
        "failures": failures,
        "no_recording": True,
        "no_collection": True,
        "no_training": True,
        "interpretation": "Short engineering benchmark only; not a posture accuracy or medical validation study.",
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    lines = [
        "# RTSP AI Lab MAIN/SUB latency benchmark",
        "",
        f"สถานะ: **{payload['status']}**",
        f"Preset: `{args.preset}` | ระยะเวลาประมาณ {args.seconds:.1f} วินาทีต่อ profile",
        "",
        "การทดสอบนี้รับเฟรมแบบ latest-frame สำหรับ RTSP, ไม่บันทึกวิดีโอ, ไม่เก็บ human-labelled data และไม่ train model",
        "",
        "| Profile | Result | Resolution | Source FPS | AI FPS | Median frame age | Median inference | Pose detected | Core accepted | Dropped old | Reconnects |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        resolution = item.get("resolution") or {}
        age = (item.get("capture_age_ms") or {}).get("median_ms")
        infer = (item.get("inference_time_ms") or {}).get("median_ms")
        lines.append(
            f"| {item.get('profile')} | {item.get('status')} | {resolution.get('width', 0)}x{resolution.get('height', 0)} | "
            f"{float(item.get('source_fps') or 0):.2f} | {float(item.get('inference_fps') or 0):.2f} | "
            f"{('--' if age is None else f'{float(age):.1f} ms')} | {('--' if infer is None else f'{float(infer):.1f} ms')} | "
            f"{float(item.get('pose_detection_percentage') or 0):.1f}% | {float(item.get('core_landmark_acceptance_percentage') or 0):.1f}% | "
            f"{int(item.get('frames_intentionally_dropped') or 0)} | {int(item.get('reconnects') or 0)} |"
        )
    if failures:
        lines.extend(["", "## Profiles ที่วัดไม่ได้", ""])
        for failure in failures:
            lines.append(f"- `{failure['profile']}`: {failure['detail']}")
    lines.extend([
        "",
        "## การตีความ",
        "",
        "ตัวเลข frame age เป็นเวลาที่วัดได้จากการ publish ในช่อง latest-frame ถึงเวลาที่ AI loop อ่าน ไม่ใช่ timestamp จากตัวกล้องโดยตรง เพราะ OpenCV ไม่เปิดเผย timestamp นั้น",
        "การเลือก MAIN หรือ SUB ต้องพิจารณาความสดของเฟรมควบคู่กับ resolution และ core landmark acceptance ไม่เลือกจาก FPS อย่างเดียว",
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
