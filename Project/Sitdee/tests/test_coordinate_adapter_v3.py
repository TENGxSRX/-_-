from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.posture_ai.features.coordinate_v3 import (
    ADAPTER_V3_CANDIDATES,
    candidate_representation,
    hip_center,
    hip_midpoints,
    transform_coordinates,
)
from src.posture_ai.features.upperbody_v2 import build_upperbody_features, upperbody_feature_names
from src.posture_ai.pose.landmarks import LANDMARK_INDEX


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw" / "multiposture" / "data.csv"
V1_DIR = ROOT / "outputs" / "models" / "multiposture_v1"
V2_DIR = ROOT / "outputs" / "models" / "multiposture_upperbody_v2"


def _pose_array() -> np.ndarray:
    values = np.zeros((33, 3), dtype=float)
    values[LANDMARK_INDEX["nose"]] = [0.1, -0.9, 0.2]
    values[LANDMARK_INDEX["left_shoulder"]] = [-0.4, 0.0, 0.1]
    values[LANDMARK_INDEX["right_shoulder"]] = [0.5, 0.1, -0.2]
    values[LANDMARK_INDEX["left_hip"]] = [-0.3, 1.0, 0.05]
    values[LANDMARK_INDEX["right_hip"]] = [0.3, 1.2, -0.05]
    return values


def test_hip_centering_uses_both_hips_and_does_not_mutate_input():
    original = _pose_array()
    centered = hip_center(original)

    np.testing.assert_allclose(hip_midpoints(centered), 0.0, atol=1e-12)
    np.testing.assert_allclose(original[LANDMARK_INDEX["nose"]], [0.1, -0.9, 0.2])
    assert not np.shares_memory(original, centered)


def test_axis_controls_are_explicit_and_independent():
    original = _pose_array()
    centered = transform_coordinates(original, "adapter_v3_a_world_hip_centered")
    zflipped = transform_coordinates(original, "adapter_v3_b_world_hip_centered_zflip")
    mirrored = transform_coordinates(original, "adapter_v3_e_world_hip_centered_mirror")

    np.testing.assert_allclose(zflipped[:, :, :2], centered[:, :, :2])
    np.testing.assert_allclose(zflipped[:, :, 2], -centered[:, :, 2])
    np.testing.assert_allclose(mirrored[:, :, 0], -centered[:, :, 0])
    np.testing.assert_allclose(mirrored[:, :, 1:], centered[:, :, 1:])


def test_adapter_candidates_are_versioned_and_representation_specific():
    assert len(ADAPTER_V3_CANDIDATES) == 5
    assert candidate_representation("adapter_v3_a_world_hip_centered") == "world_landmarks"
    assert candidate_representation("adapter_v3_c_image_hip_centered") == "normalized_image_landmarks"
    try:
        transform_coordinates(_pose_array(), "unapproved_transform")
    except ValueError as error:
        assert "Unknown V3 adapter" in str(error)
    else:
        raise AssertionError("Unknown adapter was accepted")


def test_geometry_and_normalized_features_are_translation_invariant():
    pose = _pose_array()
    translated = pose + np.array([2.0, -3.0, 4.0])
    first, names = build_upperbody_features(pose[None, :, :], "normalized", already_hip_centered=False)
    second, translated_names = build_upperbody_features(translated[None, :, :], "normalized", already_hip_centered=False)

    assert names == upperbody_feature_names("normalized") == translated_names
    np.testing.assert_allclose(first, second, atol=1e-10)


def test_v1_v2_artifacts_and_raw_data_remain_isolated():
    v1 = json.loads((V1_DIR / "metadata.json").read_text(encoding="utf-8"))
    v2 = json.loads((V2_DIR / "metadata.json").read_text(encoding="utf-8"))
    raw_hash = hashlib.sha256(RAW.read_bytes()).hexdigest()

    assert v1["feature_count"] == 126
    assert v2["feature_count"] == 30
    assert v1["dataset_sha256"] == raw_hash
    assert V1_DIR != V2_DIR
    assert not (V2_DIR / "adapter_v3_selected.marker").exists()
