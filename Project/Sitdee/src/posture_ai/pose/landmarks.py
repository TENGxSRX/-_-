"""Canonical 33-landmark representation used by the posture engine.

The order follows MediaPipe Pose's landmark index order.  Keeping the order in
one module prevents a backend-specific positional guess from silently reaching
the trained model.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
AXES = ("x", "y", "z")
LANDMARK_INDEX = {name: index for index, name in enumerate(LANDMARK_NAMES)}


@dataclass(frozen=True)
class PoseLandmark:
    """One canonical point; ``None`` coordinates represent a missing point."""

    x: float | None
    y: float | None
    z: float | None
    visibility: float | None = None

    @property
    def present(self) -> bool:
        return self.x is not None and self.y is not None and self.z is not None

    @classmethod
    def from_object(cls, value: Any) -> "PoseLandmark | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            get = value.get
        else:
            get = lambda name, default=None: getattr(value, name, default)
        x, y, z = get("x"), get("y"), get("z")
        visibility = get("visibility", None)
        if x is None or y is None or z is None:
            return cls(None, None, None, None if visibility is None else float(visibility))
        return cls(float(x), float(y), float(z), None if visibility is None else float(visibility))


@dataclass(frozen=True)
class PoseLandmarks:
    """Fixed-length MediaPipe-compatible landmark container."""

    points: tuple[PoseLandmark | None, ...]

    def __post_init__(self) -> None:
        if len(self.points) != len(LANDMARK_NAMES):
            raise ValueError(f"Expected exactly 33 landmarks, got {len(self.points)}")
        object.__setattr__(self, "points", tuple(PoseLandmark.from_object(point) for point in self.points))

    @classmethod
    def from_sequence(cls, values: Sequence[Any]) -> "PoseLandmarks":
        return cls(tuple(PoseLandmark.from_object(value) for value in values))

    @classmethod
    def from_mapping(cls, values: Mapping[int | str, Any]) -> "PoseLandmarks":
        points: list[PoseLandmark | None] = []
        for index, name in enumerate(LANDMARK_NAMES):
            value = values.get(index, values.get(name))
            points.append(PoseLandmark.from_object(value))
        return cls(tuple(points))

    @classmethod
    def from_any(cls, values: "PoseLandmarks | Mapping[int | str, Any] | Sequence[Any]") -> "PoseLandmarks":
        if isinstance(values, cls):
            return values
        if isinstance(values, Mapping):
            return cls.from_mapping(values)
        return cls.from_sequence(values)

    @classmethod
    def from_xyz_array(cls, values: np.ndarray, visibility: Sequence[float | None] | None = None) -> "PoseLandmarks":
        array = np.asarray(values, dtype=float)
        if array.shape == (99,):
            array = array.reshape(33, 3)
        if array.shape != (33, 3):
            raise ValueError(f"Expected shape (33, 3) or (99,), got {array.shape}")
        vis = list(visibility) if visibility is not None else [None] * 33
        if len(vis) != 33:
            raise ValueError("visibility must contain 33 values")
        return cls(tuple(PoseLandmark(float(row[0]), float(row[1]), float(row[2]), vis[index]) for index, row in enumerate(array)))

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: int) -> PoseLandmark | None:
        return self.points[index]

    def by_name(self, name: str) -> PoseLandmark | None:
        return self.points[LANDMARK_INDEX[name]]

    def xyz_array(self) -> np.ndarray:
        result = np.full((33, 3), np.nan, dtype=float)
        for index, point in enumerate(self.points):
            if point is not None and point.present:
                result[index] = [point.x, point.y, point.z]
        return result

    def visibility_array(self) -> np.ndarray:
        result = np.full(33, np.nan, dtype=float)
        for index, point in enumerate(self.points):
            if point is not None and point.visibility is not None:
                result[index] = float(point.visibility)
        return result

    def present_names(self) -> list[str]:
        return [name for name, point in zip(LANDMARK_NAMES, self.points) if point is not None and point.present]

    def missing_names(self) -> list[str]:
        return [name for name, point in zip(LANDMARK_NAMES, self.points) if point is None or not point.present]
