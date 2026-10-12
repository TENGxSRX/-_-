"""Versioned partial-pose features for a future CCTV Activity AI.

The contract intentionally does not require knees, ankles, a face, a person
identifier, or a fixed image location.  Missing landmarks are represented by
an explicit availability mask and zeroed coordinate value together; no frame
is silently discarded simply because lower body landmarks are occluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.posture_ai.pose.landmarks import PoseLandmarks


ACTIVITY_PARTIAL_POSE_V1_SCHEMA_VERSION = "activity_partial_pose_v1"
PARTIAL_CCTV_POSE = "PARTIAL_CCTV_POSE"
FULLER_AVAILABLE_POSE = "FULLER_AVAILABLE_POSE"
FEATURE_VARIANTS = (PARTIAL_CCTV_POSE, FULLER_AVAILABLE_POSE)
CORE_LANDMARKS = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
LOWER_LANDMARKS = ("left_knee", "right_knee", "left_ankle", "right_ankle")


@dataclass(frozen=True)
class ActivityPartialPoseFeatures:
    values: np.ndarray
    names: tuple[str, ...]
    metadata: dict[str, Any]


def activity_feature_names(variant: str = FULLER_AVAILABLE_POSE) -> tuple[str, ...]:
    """Return exact feature order for a named, versioned feature variant."""

    selected = str(variant).upper()
    if selected not in FEATURE_VARIANTS:
        raise ValueError(f"variant must be one of {FEATURE_VARIANTS}")
    landmarks = CORE_LANDMARKS if selected == PARTIAL_CCTV_POSE else CORE_LANDMARKS + LOWER_LANDMARKS
    names = [
        "normalization_valid",
        "bbox_width_normalized",
        "bbox_height_normalized",
        "bbox_aspect_ratio",
        "shoulder_tilt_y_normalized",
        "torso_length_over_shoulder_width",
        "bilateral_hips_available",
    ]
    for landmark in landmarks:
        names.extend((
            f"{landmark}_x_rel_shoulder_normalized",
            f"{landmark}_y_rel_shoulder_normalized",
            f"{landmark}_z_rel_shoulder_normalized",
            f"{landmark}_available",
            f"{landmark}_visibility",
        ))
    return tuple(names)


def _visibility(image_pose: PoseLandmarks, name: str) -> float:
    point = image_pose.by_name(name)
    if point is None or point.visibility is None:
        return 0.0
    value = float(point.visibility)
    return value if np.isfinite(value) else 0.0


def _available(image_pose: PoseLandmarks, world_pose: PoseLandmarks, name: str, threshold: float) -> bool:
    image_point = image_pose.by_name(name)
    world_point = world_pose.by_name(name)
    return bool(
        image_point is not None and image_point.present
        and world_point is not None and world_point.present
        and _visibility(image_pose, name) >= float(threshold)
    )


def _bbox_features(bbox: tuple[float, float, float, float] | None) -> tuple[float, float, float]:
    if bbox is None:
        return 0.0, 0.0, 0.0
    left, top, right, bottom = (float(value) for value in bbox)
    width, height = max(0.0, right - left), max(0.0, bottom - top)
    return width, height, 0.0 if height <= 1e-8 else width / height


def build_activity_partial_pose_features(
    image_landmarks: PoseLandmarks,
    world_landmarks: PoseLandmarks,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    variant: str = FULLER_AVAILABLE_POSE,
    visibility_threshold: float = 0.5,
) -> ActivityPartialPoseFeatures:
    """Build finite static CCTV activity features with explicit missingness.

    World coordinates are translated to shoulder midpoint and scaled by
    shoulder width.  Image-space x/y positions are never features; only bbox
    size/aspect are retained as a documented camera-distance proxy.
    """

    image_pose = PoseLandmarks.from_any(image_landmarks)
    world_pose = PoseLandmarks.from_any(world_landmarks)
    names = activity_feature_names(variant)
    left_shoulder = world_pose.by_name("left_shoulder")
    right_shoulder = world_pose.by_name("right_shoulder")
    shoulders_available = _available(image_pose, world_pose, "left_shoulder", visibility_threshold) and _available(image_pose, world_pose, "right_shoulder", visibility_threshold)
    if shoulders_available and left_shoulder is not None and right_shoulder is not None:
        midpoint = np.asarray([(left_shoulder.x + right_shoulder.x) / 2.0, (left_shoulder.y + right_shoulder.y) / 2.0, (left_shoulder.z + right_shoulder.z) / 2.0], dtype=float)
        shoulder_width = float(np.linalg.norm(np.asarray([left_shoulder.x, left_shoulder.y, left_shoulder.z]) - np.asarray([right_shoulder.x, right_shoulder.y, right_shoulder.z])))
        normalization_valid = bool(np.isfinite(midpoint).all() and shoulder_width > 1e-8)
    else:
        midpoint = np.zeros(3, dtype=float)
        shoulder_width = 0.0
        normalization_valid = False
    scale = shoulder_width if normalization_valid else 1.0
    left_hip_available = _available(image_pose, world_pose, "left_hip", visibility_threshold)
    right_hip_available = _available(image_pose, world_pose, "right_hip", visibility_threshold)
    hips_available = left_hip_available and right_hip_available
    torso_ratio = 0.0
    if normalization_valid and hips_available:
        left_hip = world_pose.by_name("left_hip")
        right_hip = world_pose.by_name("right_hip")
        assert left_hip is not None and right_hip is not None
        hip_midpoint = np.asarray([(left_hip.x + right_hip.x) / 2.0, (left_hip.y + right_hip.y) / 2.0, (left_hip.z + right_hip.z) / 2.0], dtype=float)
        torso_ratio = float(np.linalg.norm(midpoint - hip_midpoint) / scale)
    shoulder_tilt = 0.0
    if normalization_valid and left_shoulder is not None and right_shoulder is not None:
        shoulder_tilt = float((left_shoulder.y - right_shoulder.y) / scale)
    width, height, aspect = _bbox_features(bbox)
    values: list[float] = [float(normalization_valid), width, height, aspect, shoulder_tilt, torso_ratio, float(hips_available)]
    selected = CORE_LANDMARKS if str(variant).upper() == PARTIAL_CCTV_POSE else CORE_LANDMARKS + LOWER_LANDMARKS
    for name in selected:
        available = bool(normalization_valid and _available(image_pose, world_pose, name, visibility_threshold))
        point = world_pose.by_name(name)
        if available and point is not None:
            vector = (np.asarray([point.x, point.y, point.z], dtype=float) - midpoint) / scale
            coordinates = [float(value) for value in vector]
        else:
            coordinates = [0.0, 0.0, 0.0]
        values.extend((*coordinates, float(available), _visibility(image_pose, name)))
    array = np.asarray(values, dtype=float)
    if array.shape != (len(names),) or not np.isfinite(array).all():
        raise ValueError("activity_partial_pose_v1 produced non-finite or misordered features")
    return ActivityPartialPoseFeatures(
        values=array,
        names=names,
        metadata={
            "schema_version": ACTIVITY_PARTIAL_POSE_V1_SCHEMA_VERSION,
            "variant": str(variant).upper(),
            "coordinate_source": "mediapipe_world_landmarks",
            "normalization": "shoulder_midpoint_translation + shoulder_width_scale when bilateral shoulders are available",
            "camera_relative_features": ["bbox_width_normalized", "bbox_height_normalized", "bbox_aspect_ratio"],
            "camera_relative_feature_policy": "Versioned distance/scale proxy only; no image x/y location is used.",
            "missingness_policy": "Unavailable landmarks are zeroed only with paired available=0 and visibility fields.",
            "required_for_collection": ["nose", "left_shoulder", "right_shoulder"],
            "optional": ["left_hip", "right_hip", *LOWER_LANDMARKS],
            "person_identifier_feature": False,
        },
    )


def partial_pose_collection_quality(
    image_landmarks: PoseLandmarks,
    world_landmarks: PoseLandmarks | None,
    features: ActivityPartialPoseFeatures | None,
    *,
    visibility_threshold: float = 0.5,
) -> dict[str, Any]:
    """Use a partial-pose gate: nose+shoulders required; hips/legs optional."""

    reasons: list[str] = []
    if world_landmarks is None:
        reasons.append("WORLD_LANDMARKS_UNAVAILABLE")
    else:
        for name in ("nose", "left_shoulder", "right_shoulder"):
            if not _available(image_landmarks, world_landmarks, name, visibility_threshold):
                reasons.append(f"REQUIRED_UPPER_BODY_UNAVAILABLE:{name}")
    if features is None or not np.isfinite(features.values).all():
        reasons.append("PARTIAL_POSE_FEATURES_UNAVAILABLE")
    return {
        "accepted": not reasons,
        "quality_flag": "ACCEPTED_PARTIAL_POSE" if not reasons else "REJECTED_" + reasons[0].split(":", 1)[0],
        "reasons": reasons,
        "required_landmarks": ["nose", "left_shoulder", "right_shoulder"],
        "optional_landmarks": ["left_hip", "right_hip", *LOWER_LANDMARKS],
        "lower_body_required": False,
    }


__all__ = [
    "ACTIVITY_PARTIAL_POSE_V1_SCHEMA_VERSION", "ActivityPartialPoseFeatures", "CORE_LANDMARKS", "FEATURE_VARIANTS",
    "FULLER_AVAILABLE_POSE", "LOWER_LANDMARKS", "PARTIAL_CCTV_POSE", "activity_feature_names",
    "build_activity_partial_pose_features", "partial_pose_collection_quality",
]
