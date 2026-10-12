from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import joblib
from sklearn.linear_model import LogisticRegression

from src.posture_ai.features.rgb_native_v4 import (
    V4_CORE_LANDMARKS,
    build_rgb_native_v4_features,
    hip_center_world,
    validate_v4_feature_schema,
    v4_feature_names,
)
from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw" / "multiposture" / "data.csv"


def _pose() -> np.ndarray:
    values = np.zeros((33, 3), dtype=float)
    values[23] = [-0.2, 1.0, 0.1]
    values[24] = [0.2, 1.0, -0.1]
    values[11] = [-0.4, 0.0, 0.1]
    values[12] = [0.4, 0.1, -0.1]
    values[0] = [0.0, -0.5, 0.0]
    return values


def test_v4_hip_centering_is_explicit_and_non_mutating():
    original = _pose()
    centered = hip_center_world(original)
    assert centered.shape == (1, 33, 3)
    np.testing.assert_allclose((centered[0, 23] + centered[0, 24]) / 2.0, 0.0)
    np.testing.assert_allclose(original[23], [-0.2, 1.0, 0.1])


@pytest.mark.parametrize("family,expected", [("world", 15), ("normalized_world", 15), ("geometry", 15), ("hybrid", 30)])
def test_v4_feature_families_have_exact_order(family, expected):
    values, names = build_rgb_native_v4_features(_pose(), family)
    assert values.shape == (1, expected)
    assert names == v4_feature_names(family)
    validate_v4_feature_schema(values, names, family)


def test_v4_geometry_is_translation_invariant():
    first, names = build_rgb_native_v4_features(_pose(), "geometry")
    shifted, shifted_names = build_rgb_native_v4_features(_pose() + np.array([2.0, -3.0, 5.0]), "geometry")
    assert names == shifted_names
    np.testing.assert_allclose(first, shifted, atol=1e-10)


def test_v4_rejects_schema_reordering():
    values, names = build_rgb_native_v4_features(_pose(), "world")
    with pytest.raises(ValueError, match="schema"):
        validate_v4_feature_schema(values, list(reversed(names)), "world")


def test_v4_core_landmark_contract_is_five_named_points():
    assert V4_CORE_LANDMARKS == ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")


def test_v4_does_not_change_multiposture_raw_file():
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    assert digest == "d21579200db9a57dd01c24fa6ece961b0be73aa0d5d402ae8e629761c03ebe38"


def test_v4_runner_reload_and_schema_check(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    x = np.vstack([np.zeros((4, 15)), np.ones((4, 15))])
    y = np.array(["TLB"] * 4 + ["TUP"] * 4)
    model = LogisticRegression().fit(x, y)
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "feature_config.json").write_text(json.dumps({"name": "world", "columns": v4_feature_names("world"), "core_landmarks": list(V4_CORE_LANDMARKS), "adapter_version": "test"}), encoding="utf-8")
    (model_dir / "labels.json").write_text(json.dumps({"source_labels": ["TLB", "TUP"]}), encoding="utf-8")
    runner = RGBNativeV4Runner(model_dir)
    prediction = runner.predict_world_landmarks(_pose(), image_visibility=[1.0] * 33)
    assert prediction["feature_count"] == 15
    assert prediction["label"] in {"TLB", "TUP"}
