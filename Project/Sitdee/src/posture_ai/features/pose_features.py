"""MultiPosture-compatible keypoint and engineered geometry features.

The source CSV uses the 33 MediaPipe landmark names in this module, each with
XYZ columns. The source record says the coordinates are hip-centred. We keep
raw source columns unchanged and expose explicit transforms for offline
MediaPipe data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.posture_ai.pose.landmarks import AXES, LANDMARK_NAMES


def expected_feature_columns() -> list[str]:
    return [f"{landmark}_{axis}" for landmark in LANDMARK_NAMES for axis in AXES]


def _as_landmark_array(values: pd.DataFrame | np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        values = values[expected_feature_columns()].to_numpy(dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 99:
        raise ValueError(f"Expected a 2D array with 99 XYZ features, got {array.shape}")
    return array.reshape(array.shape[0], len(LANDMARK_NAMES), 3)


def _flatten(array: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    return array.reshape(array.shape[0], -1), list(names)


def raw_keypoint_features(values: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return source-order XYZ values exactly as a 99-column matrix."""
    array = _as_landmark_array(values)
    return _flatten(array, expected_feature_columns())


def normalize_keypoints(
    values: pd.DataFrame | np.ndarray,
    *,
    already_hip_centered: bool = False,
    torso_scale: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Hip-centre and optionally torso-scale keypoints.

    MultiPosture is documented as hip-centred, so training uses
    ``already_hip_centered=True`` to avoid subtracting a second time. The
    live adapter starts from MediaPipe image-normalized coordinates and uses
    ``False``. Torso length is the Euclidean distance between shoulder and hip
    centres, with an epsilon fallback for degenerate poses.
    """
    landmarks = _as_landmark_array(values).copy()
    hip_center = (landmarks[:, 23] + landmarks[:, 24]) / 2.0
    if not already_hip_centered:
        landmarks = landmarks - hip_center[:, None, :]
    if torso_scale:
        shoulder_center = (landmarks[:, 11] + landmarks[:, 12]) / 2.0
        torso_length = np.linalg.norm(shoulder_center - (np.zeros_like(shoulder_center) if already_hip_centered else hip_center), axis=1)
        torso_length = np.maximum(torso_length, 1e-8)
        landmarks = landmarks / torso_length[:, None, None]
    names = [f"{landmark}_{axis}_norm" for landmark in LANDMARK_NAMES for axis in AXES]
    return _flatten(landmarks, names)


def _distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    first = a - b
    second = c - b
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    cosine = np.sum(first * second, axis=1) / np.maximum(denominator, 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def engineered_geometry_features(values: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Calculate only geometry supported by the available 33 landmarks."""
    p = _as_landmark_array(values)
    out: dict[str, np.ndarray] = {}
    nose = p[:, 0]
    shoulders = (p[:, 11] + p[:, 12]) / 2.0
    hips = (p[:, 23] + p[:, 24]) / 2.0
    torso = shoulders - hips
    torso_length = np.maximum(np.linalg.norm(torso, axis=1), 1e-8)
    out["trunk_inclination_deg"] = np.degrees(np.arctan2(torso[:, 0], np.maximum(np.abs(torso[:, 1]), 1e-8)))
    out["trunk_vector_x"] = torso[:, 0] / torso_length
    out["trunk_vector_y"] = torso[:, 1] / torso_length
    out["trunk_vector_z"] = torso[:, 2] / torso_length
    out["shoulder_line_angle_deg"] = np.degrees(np.arctan2(p[:, 12, 1] - p[:, 11, 1], p[:, 12, 0] - p[:, 11, 0] + 1e-8))
    out["hip_line_angle_deg"] = np.degrees(np.arctan2(p[:, 24, 1] - p[:, 23, 1], p[:, 24, 0] - p[:, 23, 0] + 1e-8))
    out["shoulder_asymmetry_y"] = (p[:, 11, 1] - p[:, 12, 1]) / torso_length
    out["hip_asymmetry_y"] = (p[:, 23, 1] - p[:, 24, 1]) / torso_length
    out["left_shoulder_to_hip"] = _distance(p[:, 11], hips) / torso_length
    out["right_shoulder_to_hip"] = _distance(p[:, 12], hips) / torso_length
    out["head_to_shoulder_x"] = (nose[:, 0] - shoulders[:, 0]) / torso_length
    out["head_to_shoulder_y"] = (nose[:, 1] - shoulders[:, 1]) / torso_length
    out["head_to_shoulder_z"] = (nose[:, 2] - shoulders[:, 2]) / torso_length
    out["neck_inclination_deg"] = np.degrees(np.arctan2(nose[:, 0] - shoulders[:, 0], np.maximum(np.abs(nose[:, 1] - shoulders[:, 1]), 1e-8)))
    out["torso_length"] = torso_length
    for side, shoulder, elbow, wrist, hip, knee, ankle in (
        ("left", 11, 13, 15, 23, 25, 27),
        ("right", 12, 14, 16, 24, 26, 28),
    ):
        out[f"{side}_arm_upper_length"] = _distance(p[:, shoulder], p[:, elbow]) / torso_length
        out[f"{side}_arm_lower_length"] = _distance(p[:, elbow], p[:, wrist]) / torso_length
        out[f"{side}_leg_upper_length"] = _distance(p[:, hip], p[:, knee]) / torso_length
        out[f"{side}_leg_lower_length"] = _distance(p[:, knee], p[:, ankle]) / torso_length
        out[f"{side}_elbow_angle_deg"] = _angle(p[:, shoulder], p[:, elbow], p[:, wrist])
        out[f"{side}_knee_angle_deg"] = _angle(p[:, hip], p[:, knee], p[:, ankle])
    matrix = np.column_stack([out[name] for name in out])
    return matrix, list(out)


def build_features(values: pd.DataFrame | np.ndarray, config: str) -> tuple[np.ndarray, list[str]]:
    """Build one of raw, normalized, engineered, or hybrid configurations."""
    if config == "raw":
        return raw_keypoint_features(values)
    if config == "normalized":
        return normalize_keypoints(values, already_hip_centered=True, torso_scale=True)
    engineered, engineered_names = engineered_geometry_features(values)
    if config == "engineered":
        return engineered, engineered_names
    if config == "hybrid":
        raw, raw_names = raw_keypoint_features(values)
        return np.column_stack([raw, engineered]), raw_names + engineered_names
    raise ValueError(f"Unknown feature configuration: {config}")


def adapt_mediapipe_landmarks(landmarks: Sequence[object] | Mapping[int, object], *, visibility_threshold: float = 0.5) -> np.ndarray:
    """Compatibility wrapper for the authoritative canonical pose adapter."""
    from src.posture_ai.pose.adapter import mediapipe_to_hip_centered_xyz

    return mediapipe_to_hip_centered_xyz(landmarks, visibility_threshold=visibility_threshold)
