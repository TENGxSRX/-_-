"""RGB-native V5 feature contract.

V5 intentionally reuses the frozen V4 coordinate convention and feature
math.  The module has its own public names so that a future V5 artifact can
be versioned and tested without changing V4.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.posture_ai.features.rgb_native_v4 import (
    V4_CORE_INDICES,
    V4_CORE_LANDMARKS,
    V4_FEATURE_FAMILIES,
    V4_GEOMETRY_FEATURE_NAMES,
    build_rgb_native_v4_features,
    hip_center_world,
    v4_feature_names,
    validate_v4_feature_schema,
)

V5_CORE_LANDMARKS = V4_CORE_LANDMARKS
V5_CORE_INDICES = V4_CORE_INDICES
V5_FEATURE_FAMILIES = ("normalized_world", "geometry", "hybrid")
V5_GEOMETRY_FEATURE_NAMES = V4_GEOMETRY_FEATURE_NAMES


def v5_feature_names(family: str) -> list[str]:
    if family not in V5_FEATURE_FAMILIES:
        raise ValueError(f"Unknown RGB-native V5 feature family: {family}")
    return v4_feature_names(family)


def build_rgb_native_v5_features(
    values: Mapping | Sequence | np.ndarray,
    family: str,
) -> tuple[np.ndarray, list[str]]:
    """Build V5 features with exactly the frozen V4 world-landmark pipeline."""
    if family not in V5_FEATURE_FAMILIES:
        raise ValueError(f"Unknown RGB-native V5 feature family: {family}")
    features, names = build_rgb_native_v4_features(values, family)
    if names != v5_feature_names(family):
        raise AssertionError("V5 feature schema diverged from the frozen V4 contract")
    return features, names


def validate_v5_feature_schema(values: np.ndarray, names: Sequence[str], family: str) -> None:
    if family not in V5_FEATURE_FAMILIES:
        raise ValueError(f"Unknown RGB-native V5 feature family: {family}")
    validate_v4_feature_schema(values, names, family)


__all__ = [
    "V5_CORE_LANDMARKS",
    "V5_CORE_INDICES",
    "V5_FEATURE_FAMILIES",
    "V5_GEOMETRY_FEATURE_NAMES",
    "build_rgb_native_v5_features",
    "hip_center_world",
    "v5_feature_names",
    "validate_v5_feature_schema",
]
