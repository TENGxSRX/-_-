"""Eight camera-axis features from shoulders and hips only; see reports schema."""
from __future__ import annotations

import numpy as np

SCHEMA = "upper_body_posture_v1"
CORE_NAMES = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
CORE_INDICES = (11, 12, 23, 24)
FEATURE_NAMES = (
    "trunk_lateral_rad", "trunk_depth_rad", "shoulder_tilt_rad",
    "shoulder_yaw_cos", "shoulder_yaw_sin", "shoulder_width_torso",
    "hip_width_torso", "side_length_asymmetry",
)
SUBTYPES = {"TUP": "UPRIGHT", "TLF": "FORWARD_LEAN", "TLB": "BACKWARD_OR_RECLINED",
            "TLL": "LATERAL_LEFT", "TLR": "LATERAL_RIGHT"}


def build_features(xyz) -> np.ndarray:
    """Accept (4,3), (33,3), or batches; ignore all non-core coordinates."""
    a = np.asarray(xyz, dtype=float)
    if a.ndim == 2:
        a = a[None]
    if a.ndim == 3 and a.shape[1:] == (33, 3):
        a = a[:, CORE_INDICES]
    if a.ndim != 3 or a.shape[1:] != (4, 3) or not np.isfinite(a).all():
        raise ValueError("INVALID_SHOULDERS_OR_HIPS")
    ls, rs, lh, rh = (a[:, i] for i in range(4))
    trunk = (ls + rs - lh - rh) / 2
    shoulder, hip = rs - ls, rh - lh
    length = np.linalg.norm(trunk, axis=1)
    horizontal = np.hypot(shoulder[:, 0], shoulder[:, 2])
    if np.any(length <= 1e-8) or np.any(horizontal <= 1e-8) or np.any(np.linalg.norm(hip, axis=1) <= 1e-8):
        raise ValueError("DEGENERATE_TORSO")
    return np.column_stack((
        np.arctan2(trunk[:, 0], -trunk[:, 1]),
        np.arctan2(trunk[:, 2], -trunk[:, 1]),
        np.arctan2(shoulder[:, 1], horizontal),
        shoulder[:, 0] / horizontal, shoulder[:, 2] / horizontal,
        np.linalg.norm(shoulder, axis=1) / length,
        np.linalg.norm(hip, axis=1) / length,
        (np.linalg.norm(ls - lh, axis=1) - np.linalg.norm(rs - rh, axis=1)) / length,
    ))
