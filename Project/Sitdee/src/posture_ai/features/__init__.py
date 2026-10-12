"""Pose feature definitions and MediaPipe adapters."""

from .canonical_cctv import (
    CANONICAL_CCTV_SCHEMA_VERSION,
    CANONICAL_COORDINATE_TYPES,
    CANONICAL_LANDMARKS,
    CanonicalCCTVFeatures,
    build_canonical_cctv_features,
    canonical_feature_names,
)
from .activity_partial_pose_v1 import (
    ACTIVITY_PARTIAL_POSE_V1_SCHEMA_VERSION,
    FEATURE_VARIANTS,
    FULLER_AVAILABLE_POSE,
    PARTIAL_CCTV_POSE,
    ActivityPartialPoseFeatures,
    activity_feature_names,
    build_activity_partial_pose_features,
    partial_pose_collection_quality,
)

__all__ = [
    "CANONICAL_CCTV_SCHEMA_VERSION",
    "CANONICAL_COORDINATE_TYPES",
    "CANONICAL_LANDMARKS",
    "CanonicalCCTVFeatures",
    "build_canonical_cctv_features",
    "canonical_feature_names",
    "ACTIVITY_PARTIAL_POSE_V1_SCHEMA_VERSION",
    "FEATURE_VARIANTS",
    "FULLER_AVAILABLE_POSE",
    "PARTIAL_CCTV_POSE",
    "ActivityPartialPoseFeatures",
    "activity_feature_names",
    "build_activity_partial_pose_features",
    "partial_pose_collection_quality",
]
