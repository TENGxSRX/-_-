"""Structured event logging and neutral offline-session summaries."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class SessionAccumulator:
    def __init__(self, *, frame_interval_seconds: float = 1.0 / 30.0):
        if frame_interval_seconds <= 0:
            raise ValueError("frame_interval_seconds must be positive")
        self.frame_interval_seconds = float(frame_interval_seconds)
        self.records: list[dict] = []

    def add(self, record: dict) -> None:
        self.records.append(record)

    def summary(self, *, total_duration_seconds: float | None = None) -> dict:
        if not self.records:
            return {
                "records": 0,
                "total_analyzed_time_seconds": 0.0,
                "valid_pose_time_seconds": 0.0,
                "direction_time_seconds": {},
                "direction_time_percent": {},
                "risk_time_seconds": {},
                "warning_count": 0,
                "longest_continuous_poor_posture_seconds": 0.0,
                "warning_time_percent": 0.0,
                "interpretation": "Descriptive session telemetry only; not a medical health score.",
            }
        timestamps = [float(item.get("timestamp_seconds", 0.0)) for item in self.records]
        inferred_total = max(0.0, timestamps[-1] - timestamps[0]) + self.frame_interval_seconds
        total = float(total_duration_seconds) if total_duration_seconds is not None else inferred_total
        total = max(total, 0.0)
        valid_time = 0.0
        direction_time: defaultdict[str, float] = defaultdict(float)
        risk_time: defaultdict[str, float] = defaultdict(float)
        for index, record in enumerate(self.records):
            if index + 1 < len(self.records):
                delta = max(0.0, timestamps[index + 1] - timestamps[index])
            else:
                delta = self.frame_interval_seconds
            if record.get("pose_visible"):
                valid_time += delta
            direction = record.get("direction")
            if direction:
                direction_time[str(direction)] += delta
            risk = record.get("risk_level")
            if risk:
                risk_time[str(risk)] += delta
        poor_durations = [float(item.get("continuous_duration", 0.0) or 0.0) for item in self.records]
        warning_time = risk_time.get("WARNING", 0.0) + risk_time.get("HIGH_DURATION_RISK", 0.0)
        return {
            "records": len(self.records),
            "total_analyzed_time_seconds": total,
            "valid_pose_time_seconds": valid_time,
            "valid_pose_percent": 100.0 * valid_time / total if total else 0.0,
            "direction_time_seconds": dict(direction_time),
            "direction_time_percent": {key: 100.0 * value / total if total else 0.0 for key, value in direction_time.items()},
            "risk_time_seconds": dict(risk_time),
            "warning_count": int(sum(bool(item.get("warning_event")) for item in self.records)),
            "longest_continuous_poor_posture_seconds": max(poor_durations),
            "warning_time_seconds": warning_time,
            "warning_time_percent": 100.0 * warning_time / total if total else 0.0,
            "interpretation": "Descriptive session telemetry only; not a medical health score.",
        }


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def write_summary(path: str | Path, summary: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

