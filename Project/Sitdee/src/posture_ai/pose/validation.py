"""Validation for the canonical pose representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .landmarks import LANDMARK_NAMES, PoseLandmarks


class PoseValidationError(ValueError):
    """Raised when a pose cannot safely enter a feature adapter."""


@dataclass(frozen=True)
class PoseValidationResult:
    valid: bool
    missing_landmarks: tuple[str, ...]
    nonfinite_landmarks: tuple[str, ...]
    low_visibility_landmarks: tuple[str, ...]
    visibility_threshold: float | None

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "missing_landmarks": list(self.missing_landmarks),
            "nonfinite_landmarks": list(self.nonfinite_landmarks),
            "low_visibility_landmarks": list(self.low_visibility_landmarks),
            "visibility_threshold": self.visibility_threshold,
        }


def validate_pose_landmarks(
    values: PoseLandmarks,
    *,
    visibility_threshold: float | None = None,
    required_names: tuple[str, ...] | None = None,
    raise_on_error: bool = False,
) -> PoseValidationResult:
    pose = PoseLandmarks.from_any(values)
    names = required_names or LANDMARK_NAMES
    missing: list[str] = []
    nonfinite: list[str] = []
    low_visibility: list[str] = []
    for name in names:
        point = pose.by_name(name)
        if point is None or not point.present:
            missing.append(name)
            continue
        coords = np.asarray([point.x, point.y, point.z], dtype=float)
        if not np.isfinite(coords).all():
            nonfinite.append(name)
        if visibility_threshold is not None and point.visibility is not None and point.visibility < visibility_threshold:
            low_visibility.append(name)
    result = PoseValidationResult(
        valid=not missing and not nonfinite and not low_visibility,
        missing_landmarks=tuple(missing),
        nonfinite_landmarks=tuple(nonfinite),
        low_visibility_landmarks=tuple(low_visibility),
        visibility_threshold=visibility_threshold,
    )
    if raise_on_error and not result.valid:
        problems = []
        if missing:
            problems.append("missing=" + ",".join(missing))
        if nonfinite:
            problems.append("nonfinite=" + ",".join(nonfinite))
        if low_visibility:
            problems.append("low_visibility=" + ",".join(low_visibility))
        raise PoseValidationError("Invalid pose landmarks: " + "; ".join(problems))
    return result

