"""Explicit MediaPipe-to-MultiPosture feature adapter.

This adapter proves schema identity and the implemented hip-centering step. It
does not claim that a camera's coordinate scale or MediaPipe model version is
identical to the stored MultiPosture CSV convention; that semantic status stays
unvalidated until an appropriate paired source experiment is available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.posture_ai.features.pose_features import build_features, expected_feature_columns

from .landmarks import PoseLandmarks
from .validation import validate_pose_landmarks


@dataclass(frozen=True)
class AdaptedFeatures:
    values: np.ndarray
    names: tuple[str, ...]
    source_convention: str
    schema_identity_verified: bool
    semantic_compatibility_validated: bool


def mediapipe_to_hip_centered_xyz(landmarks, *, visibility_threshold: float = 0.5) -> np.ndarray:
    """Return one 99-value row in canonical landmark order after hip centering."""

    pose = PoseLandmarks.from_any(landmarks)
    validate_pose_landmarks(pose, visibility_threshold=visibility_threshold, raise_on_error=True)
    xyz = pose.xyz_array()
    hip_center = (xyz[23] + xyz[24]) / 2.0
    return (xyz - hip_center[None, :]).reshape(1, 99)


def multiposture_feature_names() -> list[str]:
    raw_names = expected_feature_columns()
    _, engineered_names = build_features(np.zeros((1, 99), dtype=float), "engineered")
    return raw_names + engineered_names


def validate_model_feature_schema(model_dir: str | Path) -> dict:
    """Check exact exported name/order without guessing or reordering columns."""

    config = json.loads((Path(model_dir) / "feature_config.json").read_text(encoding="utf-8"))
    expected = multiposture_feature_names()
    actual = list(config.get("columns", []))
    return {
        "model_dir": str(model_dir),
        "expected_count": len(expected),
        "actual_count": len(actual),
        "exact_name_order_match": actual == expected,
        "semantic_compatibility_validated": False,
        "reason": "Stored model schema is reproducible; MediaPipe source scale/convention equivalence is not proven.",
    }


def adapt_mediapipe_to_multiposture(landmarks, *, visibility_threshold: float = 0.5, model_dir: str | Path | None = None) -> AdaptedFeatures:
    hip_centered = mediapipe_to_hip_centered_xyz(landmarks, visibility_threshold=visibility_threshold)
    values, names = build_features(hip_centered, "hybrid")
    expected = multiposture_feature_names()
    if names != expected:
        raise ValueError("Internal MultiPosture feature ordering is inconsistent")
    schema_identity_verified = True
    if model_dir is not None:
        schema_identity_verified = bool(validate_model_feature_schema(model_dir)["exact_name_order_match"])
        if not schema_identity_verified:
            raise ValueError("Feature adapter output order does not match the exported model schema")
    return AdaptedFeatures(
        values=values,
        names=tuple(names),
        source_convention="MediaPipe x/y/z after hip-centering; no torso rescaling; visibility checked",
        schema_identity_verified=schema_identity_verified,
        semantic_compatibility_validated=False,
    )
