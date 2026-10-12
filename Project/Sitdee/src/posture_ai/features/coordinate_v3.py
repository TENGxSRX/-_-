"""Explicit coordinate-convention candidates for the V3 forensic study.

The candidates are intentionally small and versioned.  They do not learn a
mapping from external labels and they do not alter the V1 or V2 production
artifacts.  ``world_hip_centered`` is the source-supported candidate because
the MultiPosture paper states that MediaPipe World Landmarks were used with
the hip midpoint as origin.  Image-landmark and sign-flip variants are kept as
diagnostic controls only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.posture_ai.pose.landmarks import LANDMARK_INDEX, LANDMARK_NAMES, PoseLandmarks


ADAPTER_V3_CANDIDATES = (
    "adapter_v3_a_world_hip_centered",
    "adapter_v3_b_world_hip_centered_zflip",
    "adapter_v3_c_image_hip_centered",
    "adapter_v3_d_image_hip_centered_zflip",
    "adapter_v3_e_world_hip_centered_mirror",
)

ADAPTER_JUSTIFICATIONS = {
    "adapter_v3_a_world_hip_centered": "Paper-supported World Landmarks in meters, re-centered by the observed hip midpoint residual; no geometric scale or sign change.",
    "adapter_v3_b_world_hip_centered_zflip": "Diagnostic sign control only; MediaPipe and the paper both describe world z, so a flip is not source-supported.",
    "adapter_v3_c_image_hip_centered": "Current image-normalized x/y/z representation after hip translation; retained as the existing V2 baseline.",
    "adapter_v3_d_image_hip_centered_zflip": "Diagnostic image-z sign control only; no source evidence supports changing the documented MediaPipe image-z convention.",
    "adapter_v3_e_world_hip_centered_mirror": "Diagnostic horizontal-mirror control only; no source evidence indicates that the MultiPosture source was mirrored.",
}


def _array(values) -> np.ndarray:
    if isinstance(values, PoseLandmarks):
        array = values.xyz_array()
    elif isinstance(values, Mapping):
        array = PoseLandmarks.from_any(values).xyz_array()
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim == 1 and array.size == 99:
            array = array.reshape(33, 3)
        if array.ndim == 2 and array.shape == (33, 3):
            pass
        elif array.ndim == 3 and array.shape[1:] == (33, 3):
            pass
        else:
            raise ValueError(f"Expected (33, 3), (99,), or (n, 33, 3), got {array.shape}")
    if array.ndim == 2:
        array = array[None, :, :]
    if array.shape[1:] != (33, 3) or not np.isfinite(array).all():
        raise ValueError("Coordinate adapter requires finite 33-landmark XYZ values")
    return array.copy()


def hip_midpoints(values) -> np.ndarray:
    array = _array(values)
    return (array[:, LANDMARK_INDEX["left_hip"]] + array[:, LANDMARK_INDEX["right_hip"]]) / 2.0


def hip_center(values) -> np.ndarray:
    array = _array(values)
    return array - hip_midpoints(array)[:, None, :]


def transform_coordinates(values, adapter_id: str) -> np.ndarray:
    """Apply one explicit V3 candidate and return ``(n, 33, 3)`` coordinates."""
    if adapter_id not in ADAPTER_V3_CANDIDATES:
        raise ValueError(f"Unknown V3 adapter: {adapter_id}")
    transformed = _array(values)
    if "world" in adapter_id or "image" in adapter_id:
        transformed = transformed - hip_midpoints(transformed)[:, None, :]
    if adapter_id.endswith("_zflip"):
        transformed[:, :, 2] *= -1.0
    if adapter_id.endswith("_mirror"):
        transformed[:, :, 0] *= -1.0
    return transformed


def apply_adapter_to_single(values, adapter_id: str) -> PoseLandmarks:
    transformed = transform_coordinates(values, adapter_id)[0]
    return PoseLandmarks.from_xyz_array(transformed)


def candidate_representation(adapter_id: str) -> str:
    if "world" in adapter_id:
        return "world_landmarks"
    if "image" in adapter_id:
        return "normalized_image_landmarks"
    raise ValueError(adapter_id)

