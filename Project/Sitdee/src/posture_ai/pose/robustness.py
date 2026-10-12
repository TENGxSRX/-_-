"""Visibility policies for the experimental upper-body V2 adapter."""

from __future__ import annotations

import numpy as np

from src.posture_ai.features.upperbody_v2 import CORE_REQUIRED_LANDMARKS, OPTIONAL_LANDMARKS
from src.posture_ai.pose.landmarks import LANDMARK_NAMES, PoseLandmarks


def _visibility_by_name(values) -> dict[str, float]:
    pose = PoseLandmarks.from_any(values)
    return {
        name: float(point.visibility)
        for name, point in zip(LANDMARK_NAMES, pose.points)
        if point is not None and point.visibility is not None and np.isfinite(point.visibility)
    }


def pose_quality_score(values, *, names: tuple[str, ...] = CORE_REQUIRED_LANDMARKS, coverage_threshold: float = 0.3) -> dict:
    visibility = _visibility_by_name(values)
    raw = np.asarray([visibility.get(name, 0.0) for name in names], dtype=float)
    clipped = np.clip(raw, 0.0, 1.0)
    score = 0.7 * float(np.mean(clipped)) + 0.3 * float(np.mean(clipped >= coverage_threshold))
    return {
        "score": score,
        "mean_visibility": float(np.mean(clipped)),
        "coverage_fraction": float(np.mean(clipped >= coverage_threshold)),
        "missing_names": [name for name in names if name not in visibility],
        "low_names": [name for name in names if visibility.get(name, 0.0) < coverage_threshold],
    }


def evaluate_visibility_policy(
    values,
    policy: str,
    *,
    strict_threshold: float = 0.5,
    optional_threshold: float = 0.2,
    quality_threshold: float = 0.5,
) -> dict:
    """Return acceptance and transparent reasons for one pose.

    ``tiered`` gates only the core and records optional failures as warnings;
    V2 does not silently impute optional landmarks. ``quality`` has no
    per-landmark hard gate and accepts when the aggregate core score passes.
    """
    visibility = _visibility_by_name(values)
    if policy == "strict":
        required = LANDMARK_NAMES
        low = [name for name in required if visibility.get(name, 0.0) < strict_threshold]
        missing = [name for name in required if name not in visibility]
        return {
            "policy": policy,
            "accepted": not low and not missing,
            "required_names": list(required),
            "low_names": low,
            "missing_names": missing,
            "warnings": [],
            "quality": pose_quality_score(values, names=required, coverage_threshold=strict_threshold),
        }
    if policy == "core-only":
        required = CORE_REQUIRED_LANDMARKS
        low = [name for name in required if visibility.get(name, 0.0) < strict_threshold]
        missing = [name for name in required if name not in visibility]
        return {
            "policy": policy,
            "accepted": not low and not missing,
            "required_names": list(required),
            "low_names": low,
            "missing_names": missing,
            "warnings": [],
            "quality": pose_quality_score(values, names=required, coverage_threshold=strict_threshold),
        }
    if policy == "tiered":
        core = evaluate_visibility_policy(values, "core-only", strict_threshold=strict_threshold)
        optional_low = [name for name in OPTIONAL_LANDMARKS if visibility.get(name, 0.0) < optional_threshold]
        optional_missing = [name for name in OPTIONAL_LANDMARKS if name not in visibility]
        return {
            "policy": policy,
            "accepted": core["accepted"],
            "required_names": list(CORE_REQUIRED_LANDMARKS),
            "low_names": core["low_names"],
            "missing_names": core["missing_names"],
            "warnings": [f"optional_low_visibility:{name}" for name in optional_low]
            + [f"optional_missing:{name}" for name in optional_missing],
            "optional_low_names": optional_low,
            "optional_missing_names": optional_missing,
            "quality": core["quality"],
        }
    if policy == "quality":
        quality = pose_quality_score(values, names=CORE_REQUIRED_LANDMARKS, coverage_threshold=0.3)
        return {
            "policy": policy,
            "accepted": quality["score"] >= quality_threshold,
            "required_names": list(CORE_REQUIRED_LANDMARKS),
            "low_names": quality["low_names"],
            "missing_names": quality["missing_names"],
            "warnings": [],
            "quality": quality,
        }
    raise ValueError(f"Unknown visibility policy: {policy}")
