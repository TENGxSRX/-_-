"""Explicit canonical CCTV pose representation for future hierarchical models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.posture_ai.pose.landmarks import LANDMARK_INDEX, LANDMARK_NAMES, PoseLandmarks


CANONICAL_CCTV_SCHEMA_VERSION = "cctv_canonical_v1"
CANONICAL_COORDINATE_TYPES = frozenset({"mediapipe_world", "mediapipe_image_normalized"})
CANONICAL_SCALE_MODES = frozenset({"none", "torso_length"})
CANONICAL_ROTATION_MODES = frozenset({"none"})
CANONICAL_LANDMARKS = (
    "nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


@dataclass(frozen=True)
class CanonicalCCTVFeatures:
    values: np.ndarray
    names: tuple[str, ...]
    metadata: dict[str, Any]


def canonical_feature_names() -> tuple[str, ...]:
    return tuple(f"{landmark}_{axis}" for landmark in CANONICAL_LANDMARKS for axis in ("x", "y", "z"))


def _array(values: PoseLandmarks | np.ndarray) -> np.ndarray:
    if isinstance(values, PoseLandmarks):
        array = values.xyz_array()
    else:
        array = np.asarray(values, dtype=float)
        if array.shape == (99,):
            array = array.reshape(33, 3)
    if array.shape != (33, 3) or not np.isfinite(array).all():
        raise ValueError("canonical CCTV features require finite 33-landmark XYZ values")
    return array.copy()


def build_canonical_cctv_features(
    values: PoseLandmarks | np.ndarray,
    *,
    coordinate_type: str,
    scale_mode: str = "torso_length",
    rotation_mode: str = "none",
) -> CanonicalCCTVFeatures:
    """Build a versioned representation; no silent coordinate conversion."""

    coordinate = str(coordinate_type)
    scale = str(scale_mode)
    rotation = str(rotation_mode)
    if coordinate not in CANONICAL_COORDINATE_TYPES:
        raise ValueError(f"unsupported canonical coordinate_type: {coordinate}")
    if scale not in CANONICAL_SCALE_MODES or rotation not in CANONICAL_ROTATION_MODES:
        raise ValueError("unsupported canonical normalization/rotation mode")
    array = _array(values)
    hips = (array[LANDMARK_INDEX["left_hip"]] + array[LANDMARK_INDEX["right_hip"]]) / 2.0
    transformed = array - hips
    torso = ((array[LANDMARK_INDEX["left_shoulder"]] + array[LANDMARK_INDEX["right_shoulder"]]) / 2.0) - hips
    torso_length = float(np.linalg.norm(torso))
    if scale == "torso_length":
        if torso_length <= 1e-8:
            raise ValueError("cannot torso-normalize a zero-length torso")
        transformed = transformed / torso_length
    indices = [LANDMARK_INDEX[name] for name in CANONICAL_LANDMARKS]
    matrix = transformed[indices].reshape(1, -1)
    names = canonical_feature_names()
    return CanonicalCCTVFeatures(
        values=matrix,
        names=names,
        metadata={
            "schema_version": CANONICAL_CCTV_SCHEMA_VERSION,
            "coordinate_type": coordinate,
            "landmarks": list(CANONICAL_LANDMARKS),
            "normalization": "hip_midpoint_translation" + (" + torso_length_scale" if scale == "torso_length" else ""),
            "rotation": rotation,
            "v4_compatible": False,
            "viewpoint_invariance": "NOT_PROVEN",
        },
    )


__all__ = [
    "CANONICAL_CCTV_SCHEMA_VERSION",
    "CANONICAL_COORDINATE_TYPES",
    "CANONICAL_LANDMARKS",
    "CanonicalCCTVFeatures",
    "build_canonical_cctv_features",
    "canonical_feature_names",
]
