"""Estimate aggregate camera bandwidth and optional recording storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate_bandwidth(bitrate_mbps: float, camera_count: int, hours: float = 1.0, overhead_percent: float = 0.0) -> dict[str, float | str | int]:
    bitrate = float(bitrate_mbps)
    cameras = int(camera_count)
    duration_hours = float(hours)
    overhead = float(overhead_percent)
    if bitrate <= 0 or cameras < 1 or duration_hours <= 0 or overhead < 0:
        raise ValueError("bitrate, camera count and hours must be positive; overhead must not be negative")
    raw_mbps = bitrate * cameras
    planned_mbps = raw_mbps * (1.0 + overhead / 100.0)
    raw_gib_per_hour = bitrate * 3600.0 / 8.0 / 1024.0
    planned_gib_per_hour = raw_gib_per_hour * cameras * (1.0 + overhead / 100.0)
    return {
        "estimate_label": "ESTIMATE_ONLY",
        "bitrate_per_camera_mbps": bitrate,
        "camera_count": cameras,
        "overhead_percent": overhead,
        "aggregate_raw_mbps": raw_mbps,
        "aggregate_planned_mbps": planned_mbps,
        "storage_raw_gib_per_hour": raw_gib_per_hour * cameras,
        "storage_planned_gib_per_hour": planned_gib_per_hour,
        "hours_requested": duration_hours,
        "storage_planned_gib_for_requested_hours": planned_gib_per_hour * duration_hours,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate camera network throughput; actual Hikvision bitrate is not assumed")
    parser.add_argument("--bitrate-mbps", type=float, required=True, help="Measured/configured bitrate per camera")
    parser.add_argument("--cameras", type=int, required=True)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--overhead-percent", type=float, default=0.0, help="Optional planning overhead, explicitly supplied by the user")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        payload = estimate_bandwidth(args.bitrate_mbps, args.cameras, args.hours, args.overhead_percent)
    except ValueError as exc:
        parser.error(str(exc))
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
