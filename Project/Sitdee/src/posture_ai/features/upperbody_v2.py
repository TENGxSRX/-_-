"""Experimental upper-body-only MultiPosture V2 feature schema.

V2 deliberately uses only the five landmarks needed to describe head
position, shoulder alignment, trunk inclination, and hip alignment. It is a
separate schema and model family; V1's 126-feature contract is untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.posture_ai.features.pose_features import expected_feature_columns
from src.posture_ai.pose.landmarks import LANDMARK_INDEX, LANDMARK_NAMES, PoseLandmarks


CORE_REQUIRED_LANDMARKS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)
OPTIONAL_LANDMARKS = (
    "left_ear",
    "right_ear",
    "left_elbow",
    "right_elbow",
)
LOW_VALUE_FOR_OFFICE_POSTURE = tuple(
    name for name in LANDMARK_NAMES if name not in CORE_REQUIRED_LANDMARKS and name not in OPTIONAL_LANDMARKS
)
CORE_INDICES = tuple(LANDMARK_INDEX[name] for name in CORE_REQUIRED_LANDMARKS)

GEOMETRY_FEATURE_NAMES = (
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


def upperbody_raw_feature_names() -> list[str]:
    return [f"{landmark}_{axis}" for landmark in CORE_REQUIRED_LANDMARKS for axis in ("x", "y", "z")]


def upperbody_normalized_feature_names() -> list[str]:
    return [f"{landmark}_{axis}_norm" for landmark in CORE_REQUIRED_LANDMARKS for axis in ("x", "y", "z")]


def upperbody_feature_names(config: str) -> list[str]:
    if config == "raw":
        return upperbody_raw_feature_names()
    if config == "normalized":
        return upperbody_normalized_feature_names()
    if config == "geometry":
        return list(GEOMETRY_FEATURE_NAMES)
    if config == "hybrid":
        return upperbody_raw_feature_names() + list(GEOMETRY_FEATURE_NAMES)
    raise ValueError(f"Unknown upper-body V2 feature configuration: {config}")


def _coerce_core_xyz(values, *, already_hip_centered: bool = True) -> np.ndarray:
    if isinstance(values, PoseLandmarks):
        array = values.xyz_array()[list(CORE_INDICES)][None, :, :]
    elif isinstance(values, pd.DataFrame):
        full = values[expected_feature_columns()].to_numpy(dtype=np.float64).reshape(-1, len(LANDMARK_NAMES), 3)
        array = full[:, list(CORE_INDICES), :]
    else:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            if array.size == 99:
                array = array.reshape(33, 3)[list(CORE_INDICES)][None, :, :]
            elif array.size == len(CORE_REQUIRED_LANDMARKS) * 3:
                array = array.reshape(1, len(CORE_REQUIRED_LANDMARKS), 3)
            else:
                raise ValueError(f"Expected 99 full XYZ values or {len(CORE_REQUIRED_LANDMARKS) * 3} core values, got {array.size}")
        elif array.ndim == 2:
            if array.shape[1] == 99:
                array = array.reshape(-1, 33, 3)[:, list(CORE_INDICES), :]
            elif array.shape[1] == len(CORE_REQUIRED_LANDMARKS) * 3:
                array = array.reshape(-1, len(CORE_REQUIRED_LANDMARKS), 3)
            else:
                raise ValueError(f"Expected 99 or {len(CORE_REQUIRED_LANDMARKS) * 3} columns, got {array.shape}")
        elif array.ndim == 3 and array.shape[1:] == (33, 3):
            array = array[:, list(CORE_INDICES), :]
        elif array.ndim != 3 or array.shape[1:] != (len(CORE_REQUIRED_LANDMARKS), 3):
            raise ValueError(f"Expected full or core XYZ array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Upper-body V2 features require finite core XYZ coordinates")
    if not already_hip_centered:
        hip_center = (array[:, 3] + array[:, 4]) / 2.0
        array = array - hip_center[:, None, :]
    return array


def _geometry_features(core: np.ndarray) -> np.ndarray:
    nose = core[:, 0]
    left_shoulder = core[:, 1]
    right_shoulder = core[:, 2]
    left_hip = core[:, 3]
    right_hip = core[:, 4]
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
    matrix = np.column_stack([values[name] for name in GEOMETRY_FEATURE_NAMES])
    if not np.isfinite(matrix).all():
        raise ValueError("Upper-body V2 geometry produced non-finite values")
    return matrix


def build_upperbody_features(values, config: str, *, already_hip_centered: bool = True) -> tuple[np.ndarray, list[str]]:
    core = _coerce_core_xyz(values, already_hip_centered=already_hip_centered)
    if config == "raw":
        matrix = core.reshape(len(core), -1)
    elif config == "normalized":
        centered = core if already_hip_centered else _coerce_core_xyz(values, already_hip_centered=False)
        shoulder_center = (centered[:, 1] + centered[:, 2]) / 2.0
        torso_length = np.maximum(np.linalg.norm(shoulder_center, axis=1), 1e-8)
        matrix = (centered / torso_length[:, None, None]).reshape(len(centered), -1)
    elif config == "geometry":
        matrix = _geometry_features(core)
    elif config == "hybrid":
        matrix = np.column_stack([core.reshape(len(core), -1), _geometry_features(core)])
    else:
        raise ValueError(f"Unknown upper-body V2 feature configuration: {config}")
    names = upperbody_feature_names(config)
    if matrix.shape[1] != len(names):
        raise AssertionError(f"V2 feature schema mismatch: {matrix.shape} vs {len(names)}")
    return matrix, names


def adapt_mediapipe_to_upperbody_v2(landmarks, config: str, *, visibility_threshold: float = 0.5) -> tuple[np.ndarray, list[str]]:
    """Validate core landmarks and return one V2 row without optional imputation."""
    from src.posture_ai.pose.validation import validate_pose_landmarks

    pose = PoseLandmarks.from_any(landmarks)
    validation = validate_pose_landmarks(
        pose,
        required_names=CORE_REQUIRED_LANDMARKS,
        visibility_threshold=visibility_threshold,
        raise_on_error=True,
    )
    del validation
    return build_upperbody_features(pose, config, already_hip_centered=False)
