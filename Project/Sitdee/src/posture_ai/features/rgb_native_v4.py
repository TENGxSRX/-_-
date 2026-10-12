"""Authoritative RGB-native V4 feature contract.

V4 deliberately starts from MediaPipe Pose *world* landmarks and performs
only an explicit hip-midpoint translation before building features.  This is
the same transformation used for training and offline RGB inference.  The
module is independent from the released MultiPosture V1/V2/V3 artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.posture_ai.pose.landmarks import LANDMARK_INDEX, LANDMARK_NAMES, PoseLandmarks


V4_CORE_LANDMARKS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)
V4_CORE_INDICES = tuple(LANDMARK_INDEX[name] for name in V4_CORE_LANDMARKS)
V4_FEATURE_FAMILIES = ("world", "normalized_world", "geometry", "hybrid")

V4_GEOMETRY_FEATURE_NAMES = (
    "trunk_inclination_deg",
    "trunk_vector_x",
    "trunk_vector_y",
    "trunk_vector_z",
    "shoulder_line_angle_deg",
    "hip_line_angle_deg",
    "shoulder_asymmetry_y",
    "hip_asymmetry_y",
    "left_shoulder_to_hip",
    "right_shoulder_to_hip",
    "head_to_shoulder_x",
    "head_to_shoulder_y",
    "head_to_shoulder_z",
    "neck_inclination_deg",
    "torso_length",
)


def v4_world_feature_names() -> list[str]:
    return [f"{name}_{axis}_world_hip_centered" for name in V4_CORE_LANDMARKS for axis in ("x", "y", "z")]


def v4_normalized_world_feature_names() -> list[str]:
    return [f"{name}_{axis}_world_hip_centered_scaled" for name in V4_CORE_LANDMARKS for axis in ("x", "y", "z")]


def v4_feature_names(family: str) -> list[str]:
    if family == "world":
        return v4_world_feature_names()
    if family == "normalized_world":
        return v4_normalized_world_feature_names()
    if family == "geometry":
        return list(V4_GEOMETRY_FEATURE_NAMES)
    if family == "hybrid":
        return v4_world_feature_names() + list(V4_GEOMETRY_FEATURE_NAMES)
    raise ValueError(f"Unknown RGB-native V4 feature family: {family}")


def _as_full_xyz(values: PoseLandmarks | Mapping | Sequence | np.ndarray) -> np.ndarray:
    if isinstance(values, PoseLandmarks):
        array = values.xyz_array()
    elif isinstance(values, Mapping):
        array = PoseLandmarks.from_any(values).xyz_array()
    else:
        array = np.asarray(values, dtype=np.float64)
        if array.shape == (99,):
            array = array.reshape(33, 3)
        elif array.ndim == 2 and array.shape == (1, 99):
            array = array.reshape(33, 3)
    if array.shape == (33, 3):
        array = array[None, :, :]
    elif array.ndim == 2 and array.shape[1] == 99:
        array = array.reshape(-1, 33, 3)
    elif array.ndim != 3 or array.shape[1:] != (33, 3):
        raise ValueError(f"Expected (33, 3), (99,), (n, 99), or (n, 33, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("RGB-native V4 requires finite World Landmark coordinates")
    return array


def hip_center_world(values: PoseLandmarks | Mapping | Sequence | np.ndarray) -> np.ndarray:
    """Return a copied full pose translated by the midpoint of both hips."""
    full = _as_full_xyz(values).copy()
    hip_midpoint = (full[:, LANDMARK_INDEX["left_hip"]] + full[:, LANDMARK_INDEX["right_hip"]]) / 2.0
    return full - hip_midpoint[:, None, :]


def _core_centered(values: PoseLandmarks | Mapping | Sequence | np.ndarray) -> np.ndarray:
    return hip_center_world(values)[:, V4_CORE_INDICES, :]


def _geometry(core: np.ndarray) -> np.ndarray:
    nose, left_shoulder, right_shoulder, left_hip, right_hip = [core[:, i] for i in range(5)]
    shoulders = (left_shoulder + right_shoulder) / 2.0
    hips = (left_hip + right_hip) / 2.0
    torso = shoulders - hips
    torso_length = np.maximum(np.linalg.norm(torso, axis=1), 1e-8)
    values = {
        "trunk_inclination_deg": np.degrees(np.arctan2(torso[:, 0], np.maximum(np.abs(torso[:, 1]), 1e-8))),
        "trunk_vector_x": torso[:, 0] / torso_length,
        "trunk_vector_y": torso[:, 1] / torso_length,
        "trunk_vector_z": torso[:, 2] / torso_length,
        "shoulder_line_angle_deg": np.degrees(np.arctan2(right_shoulder[:, 1] - left_shoulder[:, 1], right_shoulder[:, 0] - left_shoulder[:, 0] + 1e-8)),
        "hip_line_angle_deg": np.degrees(np.arctan2(right_hip[:, 1] - left_hip[:, 1], right_hip[:, 0] - left_hip[:, 0] + 1e-8)),
        "shoulder_asymmetry_y": (left_shoulder[:, 1] - right_shoulder[:, 1]) / torso_length,
        "hip_asymmetry_y": (left_hip[:, 1] - right_hip[:, 1]) / torso_length,
        "left_shoulder_to_hip": np.linalg.norm(left_shoulder - hips, axis=1) / torso_length,
        "right_shoulder_to_hip": np.linalg.norm(right_shoulder - hips, axis=1) / torso_length,
        "head_to_shoulder_x": (nose[:, 0] - shoulders[:, 0]) / torso_length,
        "head_to_shoulder_y": (nose[:, 1] - shoulders[:, 1]) / torso_length,
        "head_to_shoulder_z": (nose[:, 2] - shoulders[:, 2]) / torso_length,
        "neck_inclination_deg": np.degrees(np.arctan2(nose[:, 0] - shoulders[:, 0], np.maximum(np.abs(nose[:, 1] - shoulders[:, 1]), 1e-8))),
        "torso_length": torso_length,
    }
    result = np.column_stack([values[name] for name in V4_GEOMETRY_FEATURE_NAMES])
    if not np.isfinite(result).all():
        raise ValueError("RGB-native V4 geometry produced non-finite values")
    return result


def build_rgb_native_v4_features(values: PoseLandmarks | Mapping | Sequence | np.ndarray, family: str) -> tuple[np.ndarray, list[str]]:
    """Build V4 features from raw World Landmarks using the frozen contract."""
    if family not in V4_FEATURE_FAMILIES:
        raise ValueError(f"Unknown RGB-native V4 feature family: {family}")
    core = _core_centered(values)
    raw = core.reshape(len(core), -1)
    if family == "world":
        matrix = raw
    elif family == "normalized_world":
        scale = np.maximum(np.linalg.norm((core[:, 1] + core[:, 2]) / 2.0, axis=1), 1e-8)
        matrix = (core / scale[:, None, None]).reshape(len(core), -1)
    elif family == "geometry":
        matrix = _geometry(core)
    else:
        matrix = np.column_stack([raw, _geometry(core)])
    names = v4_feature_names(family)
    if matrix.shape[1] != len(names):
        raise AssertionError(f"V4 schema mismatch: {matrix.shape} vs {len(names)}")
    return matrix, names


def validate_v4_feature_schema(values: np.ndarray, names: Sequence[str], family: str) -> None:
    expected = v4_feature_names(family)
    if list(names) != expected:
        raise ValueError("RGB-native V4 feature names/order do not match the frozen schema")
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(expected):
        raise ValueError(f"Expected V4 matrix with {len(expected)} columns, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("RGB-native V4 features must be finite")
