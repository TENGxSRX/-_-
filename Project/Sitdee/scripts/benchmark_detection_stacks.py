"""Preflight/benchmark gate for MediaPipe and optional YOLO stacks.

No YOLO installation or weight download is performed by this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.detection.yolo_optional import inspect_yolo_dependency


def preflight() -> dict:
    environment = inspect_yolo_dependency()
    return {
        "stack_a_mediapipe": "AVAILABLE_EXISTING_RUNTIME",
        "stack_b_yolo_crop_mediapipe": "BLOCKED_OPTIONAL_DEPENDENCY" if not environment.available else "DEPENDENCY_AVAILABLE_WEIGHTS_REQUIRED",
        "stack_c_yolo_pose": "BLOCKED_OPTIONAL_DEPENDENCY" if not environment.available else "DEPENDENCY_AVAILABLE_WEIGHTS_REQUIRED",
        "yolo_environment": environment.__dict__,
        "weights_downloaded": False,
        "real_camera_benchmark_run": False,
        "decision": "Keep MediaPipe primary until a controlled YOLO benchmark with local weights and ground truth is available.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="outputs/metrics/detection_stack_preflight.json")
    parser.add_argument("--report", default="reports/yolo_mediapipe_comparison_th.md")
    args = parser.parse_args()
    result = preflight()
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join([
        "# MediaPipe / YOLO stack comparison",
        "",
        "สถานะนี้เป็น preflight ไม่ใช่ผล benchmark YOLO",
        "",
        f"- Stack A MediaPipe: **{result['stack_a_mediapipe']}**",
        f"- Stack B YOLO person crop + MediaPipe: **{result['stack_b_yolo_crop_mediapipe']}**",
        f"- Stack C YOLO Pose: **{result['stack_c_yolo_pose']}**",
        f"- YOLO environment: `{result['yolo_environment']}`",
        "- ไม่ดาวน์โหลด weight อัตโนมัติ และไม่ติดตั้ง package ในขั้นตอนนี้.",
        "- COCO-17 keypoints ของ YOLO Pose ยังไม่ใช่ 33-landmark/world-landmark schema ของ V4 โดยอัตโนมัติ.",
        "- การเลือก stack ต้องรอ benchmark ที่มี local weights, source เดียวกัน และ human ground truth.",
        "",
    ]), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
