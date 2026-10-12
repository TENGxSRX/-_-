from __future__ import annotations

import numpy as np
import pytest

from src.posture_ai.data.activity_partial_training import activity_training_readiness, behavior_stability_metrics, evaluate_grouped_activity_models, v4_cctv_baseline_readiness
from src.posture_ai.data.behavior_collection import validate_behavior_labels
from src.posture_ai.features.activity_partial_pose_v1 import (
    FULLER_AVAILABLE_POSE,
    PARTIAL_CCTV_POSE,
    activity_feature_names,
    build_activity_partial_pose_features,
    partial_pose_collection_quality,
)
from src.posture_ai.pose.landmarks import PoseLandmarks


def _poses(*, lower_visibility: float = 0.05, hip_visibility: float = 0.95) -> tuple[PoseLandmarks, PoseLandmarks]:
    xyz = np.zeros((33, 3), dtype=float)
    xyz[0] = [0.0, 0.30, 0.05]
    xyz[11] = [-0.20, 0.50, 0.0]
    xyz[12] = [0.20, 0.50, 0.0]
    xyz[23] = [-0.15, 0.95, 0.02]
    xyz[24] = [0.15, 0.95, 0.02]
    xyz[25] = [-0.20, 1.10, 0.0]
    xyz[26] = [0.20, 1.10, 0.0]
    xyz[27] = [-0.20, 1.25, 0.0]
    xyz[28] = [0.20, 1.25, 0.0]
    visibility = [0.95] * 33
    for index in (23, 24):
        visibility[index] = hip_visibility
    for index in (25, 26, 27, 28):
        visibility[index] = lower_visibility
    return PoseLandmarks.from_xyz_array(xyz, visibility=visibility), PoseLandmarks.from_xyz_array(xyz)


def test_partial_pose_schema_accepts_missing_legs_with_explicit_masks():
    image, world = _poses(lower_visibility=0.05)
    features = build_activity_partial_pose_features(image, world, bbox=(0.2, 0.2, 0.8, 0.9), variant=FULLER_AVAILABLE_POSE)
    names = features.names
    assert features.metadata["schema_version"] == "activity_partial_pose_v1"
    assert features.values.shape == (len(names),)
    assert features.values[names.index("left_knee_available")] == 0.0
    assert features.values[names.index("left_knee_x_rel_shoulder_normalized")] == 0.0
    quality = partial_pose_collection_quality(image, world, features)
    assert quality["accepted"] is True
    assert quality["lower_body_required"] is False


def test_partial_variant_excludes_lower_body_and_person_id_features():
    names = activity_feature_names(PARTIAL_CCTV_POSE)
    assert not any("knee" in name or "ankle" in name for name in names)
    assert not any("person" in name or "track" in name for name in names)
    assert "bbox_width_normalized" in names
    assert "normalization_valid" in names


def test_partial_collection_still_accepts_missing_hips_when_upper_body_is_available():
    image, world = _poses(hip_visibility=0.05)
    features = build_activity_partial_pose_features(image, world, variant=FULLER_AVAILABLE_POSE)
    quality = partial_pose_collection_quality(image, world, features)
    assert quality["accepted"] is True
    assert features.values[features.names.index("bilateral_hips_available")] == 0.0


def test_human_behavior_label_contract_prevents_invalid_combinations():
    assert validate_behavior_labels("STANDING", "NONE") == ("STANDING", "NONE")
    assert validate_behavior_labels("SITTING", "TUP") == ("SITTING", "TUP")
    assert validate_behavior_labels("SITTING", "TLB") == ("SITTING", "TLB")
    with pytest.raises(ValueError):
        validate_behavior_labels("STANDING", "TUP")
    with pytest.raises(ValueError):
        validate_behavior_labels("SITTING", "NONE")


def test_activity_trainer_refuses_to_evaluate_without_verified_participant_data():
    readiness = activity_training_readiness([])
    assert readiness["status"] == "READY_FOR_BEHAVIOR_DATA_COLLECTION"
    assert v4_cctv_baseline_readiness([])["status"] == "NOT_EVALUATED_NO_VERIFIED_SITTING_V4_ROWS"
    with pytest.raises(ValueError, match="READY_FOR_BEHAVIOR_DATA_COLLECTION"):
        evaluate_grouped_activity_models([])
    assert behavior_stability_metrics([])["activity"]["status"] == "NOT_EVALUATED_NO_ACTIVITY_AI_PREDICTIONS"
