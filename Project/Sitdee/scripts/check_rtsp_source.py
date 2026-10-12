"""Bounded, no-recording RTSP/source health diagnostic.

This tool reads local settings, opens an explicitly requested source for a
small frame limit, reports redacted health metadata, and closes cleanly. It
does not run MediaPipe, save screenshots, record video, or start a listener.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.inference.video_source import (  # noqa: E402
    SOURCE_CLOSED,
    SOURCE_FAILED,
    SourceConfigurationError,
    VideoSource,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a prerecorded/USB/RTSP source without AI or recording")
    parser.add_argument("--source", default="rtsp", help="rtsp alias, HIKVISION profile, redacted RTSP URL, file path, or explicit USB index")
    parser.add_argument("--profile", choices=("HIKVISION_MAIN", "HIKVISION_SUB"))
    parser.add_argument("--frames", type=int, default=30, help="Maximum successful frames to read (1-300)")
    parser.add_argument("--seconds", type=float, default=10.0, help="Maximum diagnostic duration (0.1-30 seconds)")
    parser.add_argument("--output", help="Optional JSON report path; no video or image is written")
    args = parser.parse_args()
    if not 1 <= args.frames <= 300:
        parser.error("--frames must be between 1 and 300")
    if not 0.1 <= args.seconds <= 30.0:
        parser.error("--seconds must be between 0.1 and 30")

    try:
        source = VideoSource(
            args.source,
            profile=args.profile,
            config_path=ROOT / "configs" / "rtsp_profiles.yaml",
            max_reconnects=3,
            stale_after_seconds=2.0,
            low_latency=True,
        )
    except SourceConfigurationError as exc:
        payload = {"status": "CONFIG_MISSING_OR_INVALID", "detail": str(exc), "network_accessed": False, "ai_inference": False}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    opened = source.open()
    if not opened:
        health = source.health()
        source.close()
        payload = {"status": "SOURCE_OPEN_FAILED", "health": health, "network_accessed": source.source_type == "RTSP", "ai_inference": False, "recording": False}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    started = time.monotonic()
    frames = 0
    read_failures = 0
    while frames < args.frames and time.monotonic() - started < args.seconds:
        result = source.read()
        if result.ok:
            frames += 1
        else:
            read_failures += 1
            if result.state in {SOURCE_CLOSED, SOURCE_FAILED}:
                break
    health = source.health()
    source.close()
    payload = {
        "status": "PASS" if frames else "NO_FRAME_READ",
        "source": source.spec.source_display,
        "source_type": source.source_type,
        "camera_id": source.camera_id,
        "frames_read": frames,
        "read_failures_observed": read_failures,
        "health_before_close": health,
        "diagnostic_seconds": time.monotonic() - started,
        "network_accessed": source.source_type == "RTSP",
        "ai_inference": False,
        "recording": False,
        "screenshots": False,
        "closed_cleanly": True,
        "interpretation": "Source health smoke test only; no posture result or camera accuracy claim.",
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
