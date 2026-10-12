from __future__ import annotations

import numpy as np

from src.posture_ai.detection.yolo_optional import (
    PersonBox,
    crop_frame,
    inspect_yolo_dependency,
    map_crop_landmarks_to_full_frame,
)
from src.posture_ai.diagnostics.failure_routing import (
    ACTIVITY_CLASSIFIER_FAILURE,
    CORE_FAILURE,
    PERSON_DETECTION_FAILURE,
    classify_frame_failure,
)
from src.posture_ai.ergonomics.hierarchical import (
    HierarchicalEvidence,
    HierarchicalTemporalEngine,
    decide_hierarchical_posture,
)
from src.posture_ai.features.canonical_cctv import build_canonical_cctv_features, canonical_feature_names
from src.posture_ai.features.cctv_augmentation import SkeletonAugmentationConfig, augment_world_skeleton
from src.posture_ai.pose.landmarks import PoseLandmarks


def _pose() -> PoseLandmarks:
    values = np.zeros((33, 3), dtype=float)
    values[:, 0] = np.linspace(0.1, 0.9, 33)
    values[:, 1] = np.linspace(0.2, 0.8, 33)
    values[:, 2] = np.linspace(-0.1, 0.1, 33)
    return PoseLandmarks.from_xyz_array(values, visibility=[0.9] * 33)


def test_yolo_optional_geometry_helpers_do_not_require_backend():
    env = inspect_yolo_dependency()
    assert isinstance(env.available, bool)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop, clipped = crop_frame(frame, PersonBox(50, 20, 150, 80, 0.9))
    assert crop.shape == (60, 100, 3)
    mapped = map_crop_landmarks_to_full_frame(_pose(), clipped, frame_width=200, frame_height=100)
    assert mapped[0] is not None
    assert mapped[0].x == 0.3
    assert mapped[0].z == _pose()[0].z


def test_hierarchical_decision_is_evidence_gated():
    assert decide_hierarchical_posture(HierarchicalEvidence(True)).state == "UNCERTAIN"
    assert decide_hierarchical_posture(HierarchicalEvidence(True, "STANDING", 0.9)).state == "STANDING"
    assert decide_hierarchical_posture(HierarchicalEvidence(True, "SITTING", 0.9, "INCORRECT", 0.9, "FORWARD", 0.9)).state == "SITTING_INCORRECT_FORWARD"


def test_hierarchical_temporal_state_is_per_engine_and_raw_is_exposed():
    engine = HierarchicalTemporalEngine(min_consecutive=2)
    first = engine.update(0.0, HierarchicalEvidence(True))
    second = engine.update(0.1, HierarchicalEvidence(True, "STANDING", 0.9))
    third = engine.update(0.2, HierarchicalEvidence(True, "STANDING", 0.9))
    assert first["committed_decision"]["state"] == "UNCERTAIN"
    assert second["candidate_state"] == "STANDING"
    assert third["committed_decision"]["state"] == "STANDING"
    assert third["raw_outputs_exposed"] is True


def test_failure_routing_separates_detection_core_and_activity():
    assert classify_frame_failure(expected_count=2, detected_count=0, core_valid=False) == PERSON_DETECTION_FAILURE
    assert classify_frame_failure(expected_count=1, detected_count=1, core_valid=False) == CORE_FAILURE
    assert classify_frame_failure(expected_count=1, detected_count=1, core_valid=True, hierarchy={"state": "UNCERTAIN", "reason": "ACTIVITY_EVIDENCE_INSUFFICIENT"}) == ACTIVITY_CLASSIFIER_FAILURE


def test_canonical_features_are_versioned_and_hip_centered():
    pose = _pose()
    result = build_canonical_cctv_features(pose, coordinate_type="mediapipe_world", scale_mode="none")
    assert result.values.shape == (1, 27)
    assert result.names == canonical_feature_names()
    assert result.metadata["v4_compatible"] is False
    assert np.isclose(result.values[0, 3 * 3] + result.values[0, 4 * 3], 0.0)  # left/right hip midpoint is the origin


def test_augmentation_is_explicit_and_does_not_mutate_input():
    source = np.zeros((5, 3), dtype=float)
    augmented, visibility = augment_world_skeleton(source, SkeletonAugmentationConfig(yaw_degrees=20, translation_xyz=(1, 0, 0), seed=3))
    assert np.array_equal(source, np.zeros((5, 3)))
    assert np.allclose(augmented[:, 0], 1.0)
    assert visibility.shape == (5,)
