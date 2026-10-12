"""Finite Windows camera discovery for the office-posture project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.inference.webcam_v1 import SUPPORTED_CAMERA_INDICES, probe_camera  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check camera indices 0, 1, and 2 without probing indefinitely.")
    parser.add_argument("--indices", nargs="+", type=int, default=list(SUPPORTED_CAMERA_INDICES))
    args = parser.parse_args()
    results = []
    for index in args.indices:
        try:
            results.append(probe_camera(index))
        except ValueError as exc:
            results.append({"index": index, "opened": False, "error": str(exc)})
    for result in results:
        state = "READY" if result.get("opened") and result.get("readable_frame") else "NOT AVAILABLE"
        size = f"{result.get('width', 0)}x{result.get('height', 0)}"
        fps = result.get("reported_fps", 0.0) or 0.0
        print(f"Camera {result.get('index')}: {state} | resolution {size} | reported FPS {float(fps):.1f}")
    print(json.dumps({"cameras": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

