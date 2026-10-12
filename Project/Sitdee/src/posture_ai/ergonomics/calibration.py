"""Runtime-only neutral-pose calibration for live geometry indicators.

This module never trains, relabels, or writes dataset rows.  It estimates a
short-lived session reference so camera placement and a person's natural
neutral pose do not automatically look like forward or lateral lean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


_SCALAR_FIELDS = (
    "trunk_inclination_deg",
    "torso_lateral_displacement",
    "head_forward_proxy",
    "shoulder_tilt_deg",
)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _angle_delta(value: float, baseline: float) -> float:
    """Return the shortest signed degree difference in [-180, 180)."""

    return float((value - baseline + 180.0) % 360.0 - 180.0)


def relative_geometry(geometry: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    """Subtract a frozen session baseline from supported geometry fields.

    The raw geometry dictionary is copied and never mutated.  A missing or
    incomplete baseline leaves the values unchanged and marks the result as
    not calibrated.
    """

    result = dict(geometry or {})
    result["neutral_baseline_applied"] = False
    if not baseline or not result.get("valid"):
        return result
    if any(_finite(baseline.get(field)) is None for field in _SCALAR_FIELDS):
        return result

    for field in _SCALAR_FIELDS:
        value = _finite(result.get(field))
        reference = _finite(baseline.get(field))
        if value is None or reference is None:
            continue
        if field == "shoulder_tilt_deg":
            result[field] = _angle_delta(value, reference)
        else:
            result[field] = float(value - reference)

    raw_vector = result.get("trunk_vector")
    reference_vector = baseline.get("trunk_vector")
    if raw_vector is not None and reference_vector is not None:
        raw = np.asarray(raw_vector, dtype=float)
        reference = np.asarray(reference_vector, dtype=float)
        if raw.shape == (3,) and reference.shape == (3,) and np.isfinite(raw).all() and np.isfinite(reference).all():
            result["trunk_vector"] = [float(value) for value in raw - reference]
    result["neutral_baseline_applied"] = True
    result["neutral_reference"] = {field: baseline[field] for field in _SCALAR_FIELDS}
    return result


@dataclass
class NeutralBaselineCalibrator:
    """Collect a contiguous stable neutral interval in runtime memory only."""

    required_seconds: float = 4.0
    max_gap_seconds: float = 0.75
    _active: bool = field(default=False, init=False)
    _started_at: float | None = field(default=None, init=False)
    _last_good_at: float | None = field(default=None, init=False)
    _observations: list[dict[str, Any]] = field(default_factory=list, init=False)
    baseline: dict[str, Any] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.required_seconds <= 0 or self.max_gap_seconds <= 0:
            raise ValueError("neutral calibration durations must be positive")

    @property
    def active(self) -> bool:
        return self._active

    @property
    def completed(self) -> bool:
        return self.baseline is not None and not self._active

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    def reset(self) -> None:
        self._active = False
        self._started_at = None
        self._last_good_at = None
        self._observations.clear()
        self.baseline = None

    def start(self, timestamp_seconds: float) -> None:
        self._active = True
        self._started_at = float(timestamp_seconds)
        self._last_good_at = None
        self._observations.clear()
        self.baseline = None

    def observe(self, geometry: dict[str, Any] | None, *, accepted: bool, timestamp_seconds: float) -> dict[str, Any]:
        """Add one good frame or reset the contiguous stable interval."""

        timestamp = float(timestamp_seconds)
        if not self._active:
            return self.status(timestamp)
        if not accepted or not geometry or not geometry.get("valid"):
            self._started_at = None
            self._last_good_at = None
            self._observations.clear()
            return self.status(timestamp)
        if self._last_good_at is not None and timestamp - self._last_good_at > self.max_gap_seconds:
            self._started_at = timestamp
            self._observations.clear()
        if self._started_at is None:
            self._started_at = timestamp
        self._last_good_at = timestamp
        self._observations.append(dict(geometry))
        stable_seconds = max(0.0, timestamp - self._started_at)
        if stable_seconds >= self.required_seconds:
            self.baseline = self._estimate_baseline()
            self._active = False
        return self.status(timestamp)

    def _estimate_baseline(self) -> dict[str, Any]:
        baseline: dict[str, Any] = {}
        for field in _SCALAR_FIELDS:
            values = [_finite(item.get(field)) for item in self._observations]
            numeric = [value for value in values if value is not None]
            if numeric:
                baseline[field] = float(np.mean(numeric))
        vectors = []
        for item in self._observations:
            vector = np.asarray(item.get("trunk_vector"), dtype=float)
            if vector.shape == (3,) and np.isfinite(vector).all():
                vectors.append(vector)
        if vectors:
            baseline["trunk_vector"] = [float(value) for value in np.mean(np.vstack(vectors), axis=0)]
        baseline["sample_count"] = len(self._observations)
        baseline["required_seconds"] = self.required_seconds
        baseline["status"] = "RUNTIME_SESSION_BASELINE_ONLY"
        return baseline

    def status(self, timestamp_seconds: float) -> dict[str, Any]:
        timestamp = float(timestamp_seconds)
        stable_seconds = 0.0 if self._started_at is None else max(0.0, timestamp - self._started_at)
        return {
            "active": self._active,
            "completed": self.completed,
            "stable_seconds": stable_seconds,
            "required_seconds": self.required_seconds,
            "observation_count": len(self._observations),
            "ready": self.completed,
            "message": "NEUTRAL BASELINE READY" if self.completed else "นั่งตัวตรงตามธรรมชาติ แล้วค้างไว้",
        }
