"""Explicit synthetic augmentation for future CCTV pose experiments.

This module only transforms an in-memory canonical skeleton.  It never writes
raw datasets and it does not create labels.  Augmented samples must retain the
source group and may only be used inside a training fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CCTV_AUGMENTATION_SCHEMA_VERSION = "cctv_aug_v1"


@dataclass(frozen=True)
class SkeletonAugmentationConfig:
    """Small, explicit perturbations suitable for a future training fold."""

    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    scale: float = 1.0
    translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gaussian_noise_std: float = 0.0
    visibility_dropout_probability: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite([self.yaw_degrees, self.pitch_degrees, self.scale, *self.translation_xyz, self.gaussian_noise_std]).all():
            raise ValueError("augmentation parameters must be finite")
        if self.scale <= 0.0 or self.gaussian_noise_std < 0.0:
            raise ValueError("scale must be positive and noise must be non-negative")
        if not 0.0 <= self.visibility_dropout_probability <= 1.0:
            raise ValueError("visibility_dropout_probability must be in [0, 1]")


def augment_world_skeleton(values: np.ndarray, config: SkeletonAugmentationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return a transformed copy and a per-landmark visibility mask.

    The input is expected to be a hip-centered or otherwise consistent
    ``(N, 3)`` skeleton.  No coordinate convention conversion is performed.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("values must be a finite (N, 3) array")
    yaw = np.deg2rad(float(config.yaw_degrees))
    pitch = np.deg2rad(float(config.pitch_degrees))
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    yaw_matrix = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    pitch_matrix = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    transformed = (array @ (yaw_matrix @ pitch_matrix).T) * float(config.scale)
    transformed += np.asarray(config.translation_xyz, dtype=float)
    rng = np.random.default_rng(config.seed)
    if config.gaussian_noise_std > 0.0:
        transformed += rng.normal(0.0, config.gaussian_noise_std, size=transformed.shape)
    visibility = np.ones(len(array), dtype=bool)
    if config.visibility_dropout_probability > 0.0:
        visibility = rng.random(len(array)) >= config.visibility_dropout_probability
    if not np.isfinite(transformed).all():
        raise ValueError("augmentation generated non-finite values")
    return transformed, visibility


__all__ = ["CCTV_AUGMENTATION_SCHEMA_VERSION", "SkeletonAugmentationConfig", "augment_world_skeleton"]
