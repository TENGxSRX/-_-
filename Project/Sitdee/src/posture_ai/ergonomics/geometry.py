"""Deterministic geometry from canonical pose landmarks.

These quantities are geometric indicators in image/model coordinates. They are
not clinically validated posture measurements.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.posture_ai.pose.landmarks import PoseLandmarks


CORE_NAMES = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
GEOMETRY_FEATURE_NAMES = (
    "trunk_inclination_deg",
    "shoulder_tilt_deg",
    "hip_tilt_deg",
    "torso_lateral_displacement",
    "torso_length",
    "head_to_shoulder_x",
    "head_to_shoulder_y",
    "head_to_shoulder_z",
    "head_forward_proxy",
    "shoulder_height_asymmetry",
    "hip_height_asymmetry",
    "shoulder_to_hip_length_asymmetry",
)


def _coerce_pose(values) -> PoseLandmarks:
    if isinstance(values, np.ndarray):
        return PoseLandmarks.from_xyz_array(values)
    return PoseLandmarks.from_any(values)


def _point(pose: PoseLandmarks, name: str) -> np.ndarray | None:
    point = pose.by_name(name)
    if point is None or not point.present:
        return None
    value = np.asarray([point.x, point.y, point.z], dtype=float)
    return value if np.isfinite(value).all() else None


def _safe_float(value: float | np.floating | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _line_tilt_deg(left: np.ndarray, right: np.ndarray) -> float:
    delta = right - left
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def calculate_geometry(values) -> dict:
    """Calculate supported geometry, returning explicit validity metadata.

    Angles are in degrees. Distances and offsets divided by torso length are
    dimensionless ratios in the coordinate convention supplied by the pose
    backend. The x axis is assumed horizontal and y vertical for tilt/lean
    indicators; camera orientation is a deployment constraint.
    """

    pose = _coerce_pose(values)
    points = {name: _point(pose, name) for name in (
        "nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    )}
    missing = [name for name in CORE_NAMES if points[name] is None]
    result: dict[str, float | bool | list[str] | None] = {
        "valid": not missing,
        "missing_landmarks": missing,
        "coordinate_units": "backend coordinates; normalized ratios are dimensionless",
    }
    if missing:
        for name in (
            "trunk_inclination_deg", "shoulder_tilt_deg", "hip_tilt_deg",
            "torso_lateral_displacement", "torso_length", "head_to_shoulder_x",
            "head_to_shoulder_y", "head_to_shoulder_z", "head_forward_proxy",
            "shoulder_height_asymmetry", "hip_height_asymmetry",
            "shoulder_to_hip_length_asymmetry",
        ):
            result[name] = None
        return result

    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    left_hip = points["left_hip"]
    right_hip = points["right_hip"]
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
    hip_midpoint = (left_hip + right_hip) / 2.0
    torso_vector = shoulder_midpoint - hip_midpoint
    torso_length = float(np.linalg.norm(torso_vector))
    if torso_length <= 1e-8 or not np.isfinite(torso_length):
        result.update({"valid": False, "geometry_error": "degenerate torso length"})
        return result

    result.update({
        "trunk_inclination_deg": _safe_float(np.degrees(np.arctan2(torso_vector[0], max(abs(torso_vector[1]), 1e-8)))),
        "shoulder_tilt_deg": _line_tilt_deg(left_shoulder, right_shoulder),
        "hip_tilt_deg": _line_tilt_deg(left_hip, right_hip),
        "torso_lateral_displacement": _safe_float(torso_vector[0] / torso_length),
        "torso_length": torso_length,
        "shoulder_midpoint_x": _safe_float(shoulder_midpoint[0]),
        "shoulder_midpoint_y": _safe_float(shoulder_midpoint[1]),
        "hip_midpoint_x": _safe_float(hip_midpoint[0]),
        "hip_midpoint_y": _safe_float(hip_midpoint[1]),
        "trunk_vector": [float(value) for value in torso_vector],
        "shoulder_height_asymmetry": _safe_float((left_shoulder[1] - right_shoulder[1]) / torso_length),
        "hip_height_asymmetry": _safe_float((left_hip[1] - right_hip[1]) / torso_length),
        "shoulder_to_hip_length_asymmetry": _safe_float(
            (np.linalg.norm(left_shoulder - hip_midpoint) - np.linalg.norm(right_shoulder - hip_midpoint)) / torso_length
        ),
    })

    nose = points["nose"]
    if nose is None:
        result.update({
            "head_to_shoulder_x": None,
            "head_to_shoulder_y": None,
            "head_to_shoulder_z": None,
            "head_forward_proxy": None,
        })
    else:
        head_offset = (nose - shoulder_midpoint) / torso_length
        result.update({
            "head_to_shoulder_x": _safe_float(head_offset[0]),
            "head_to_shoulder_y": _safe_float(head_offset[1]),
            "head_to_shoulder_z": _safe_float(head_offset[2]),
            "head_forward_proxy": _safe_float(head_offset[2]),
        })
    return result


def geometry_feature_vector(values) -> tuple[np.ndarray, list[str]]:
    """Return the stable numeric geometry order for downstream experiments."""

    snapshot = calculate_geometry(values)
    if not snapshot.get("valid"):
        raise ValueError(f"Geometry is not valid: {snapshot.get('missing_landmarks') or snapshot.get('geometry_error')}")
    vector = np.asarray([snapshot[name] for name in GEOMETRY_FEATURE_NAMES], dtype=float).reshape(1, -1)
    if not np.isfinite(vector).all():
        raise ValueError("Geometry feature vector contains non-finite values")
    return vector, list(GEOMETRY_FEATURE_NAMES)
