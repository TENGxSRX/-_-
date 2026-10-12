"""Experimental shared anatomical schema; sources are benchmarked separately."""
from __future__ import annotations

import numpy as np

from src.posture_ai.features.upper_body_posture_v1 import FEATURE_NAMES as BASELINE_NAMES, build_features as baseline_features

SCHEMA = "upper_body_posture_v2"
CORE_NAMES = ("nose", "left_shoulder", "right_shoulder")
INDICES = (0, 11, 12, 23, 24)
CONDITIONS = ("CURRENT_BASELINE", "NO_HIP", "HIP_DROPOUT_ROBUST")
NO_HIP_NAMES = ("shoulder_tilt_rad", "shoulder_yaw_cos", "shoulder_yaw_sin",
                "nose_shoulder_x_width", "nose_shoulder_y_width", "nose_shoulder_z_width")
HIP_FEATURE_INDICES = (0, 1, 5, 6, 7)
FEATURES = dict(zip(CONDITIONS, (BASELINE_NAMES, NO_HIP_NAMES,
                               NO_HIP_NAMES + tuple(BASELINE_NAMES[i] for i in HIP_FEATURE_INDICES) + ("hips_available",))))


def build_features(xyz, condition="HIP_DROPOUT_ROBUST"):
    a = np.asarray(xyz, dtype=float)
    if a.ndim == 2:
        a = a[None]
    if a.ndim == 3 and a.shape[1:] == (33, 3):
        a = a[:, INDICES]
    if a.ndim != 3 or a.shape[1:] != (5, 3):
        raise ValueError("INVALID_UPPER_BODY_SHAPE")
    if condition == "CURRENT_BASELINE":
        return baseline_features(a[:, 1:])
    if condition not in FEATURES:
        raise ValueError("UNKNOWN_FEATURE_CONDITION")
    if not np.isfinite(a[:, :3]).all():
        raise ValueError("INVALID_NOSE_OR_SHOULDERS")
    nose, ls, rs = (a[:, i] for i in range(3))
    shoulder = rs - ls
    width = np.linalg.norm(shoulder, axis=1)
    horizontal = np.hypot(shoulder[:, 0], shoulder[:, 2])
    if np.any(width <= 1e-8) or np.any(horizontal <= 1e-8):
        raise ValueError("DEGENERATE_SHOULDERS")
    head = (nose - (ls + rs) / 2) / width[:, None]
    if np.any(np.linalg.norm(head, axis=1) <= 1e-8):
        raise ValueError("DEGENERATE_HEAD_ANCHOR")
    upper = np.column_stack((np.arctan2(shoulder[:, 1], horizontal),
                             shoulder[:, 0] / horizontal, shoulder[:, 2] / horizontal, head))
    if condition == "NO_HIP":
        return upper
    optional = np.zeros((len(a), len(HIP_FEATURE_INDICES)))
    available = np.isfinite(a[:, 3:]).all(axis=(1, 2))
    # A single hip cannot define a defensible hip midpoint: omit the pair.
    for i in np.flatnonzero(available):
        try:
            optional[i] = baseline_features(a[i, 1:])[0, list(HIP_FEATURE_INDICES)]
        except ValueError:
            available[i] = False
    return np.column_stack((upper, optional, available.astype(float)))


def training_hip_dropout(xyz, labels):
    """Called only AFTER slicing a training fold; no input or label mutation."""
    a = np.asarray(xyz, dtype=float)
    missing = a.copy()
    missing[:, 3:] = np.nan
    # Partial hip loss also omits the pair at inference; one masked view suffices.
    return np.concatenate((a, missing)), np.tile(labels, 2)
