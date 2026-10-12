from pathlib import Path
import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.posture_ai.data.rgb_native_v5 import (
    V5_MANIFEST_COLUMNS,
    dataset_class_confounds,
    group_overlap,
    sample_frame_indices,
    validate_manifest_rows,
)
from src.posture_ai.features.rgb_native_v4 import build_rgb_native_v4_features
from src.posture_ai.features.rgb_native_v5 import (
    V5_CORE_LANDMARKS,
    build_rgb_native_v5_features,
    v5_feature_names,
)
from src.posture_ai.inference.rgb_native_v5 import RGBNativeV5Runner


def _pose():
    values = np.zeros((33, 3), dtype=float)
    values[0] = [0.0, -0.5, 0.1]
    values[11] = [-0.2, 0.0, 0.0]
    values[12] = [0.2, 0.0, 0.0]
    values[23] = [-0.2, 0.8, 0.0]
    values[24] = [0.2, 0.8, 0.0]
    return values


def _row(**overrides):
    row = {name: "" for name in V5_MANIFEST_COLUMNS}
    row.update({"mapping_status": "COMPATIBLE", "project_label": "TUP", "pose_detected": 1})
    row.update(overrides)
    return row


def test_v5_reuses_v4_feature_math_and_order_without_mutating_v4():
    for family in ("normalized_world", "geometry", "hybrid"):
        v4_values, v4_names = build_rgb_native_v4_features(_pose(), family)
        v5_values, v5_names = build_rgb_native_v5_features(_pose(), family)
        np.testing.assert_allclose(v4_values, v5_values)
        assert v4_names == v5_names == v5_feature_names(family)


def test_v5_core_is_the_five_landmark_contract():
    assert V5_CORE_LANDMARKS == ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")


def test_sampling_is_deterministic_and_not_adjacent_at_video_rate():
    indices = sample_frame_indices(300, 30.0, 1.0)
    assert indices.tolist() == list(range(0, 300, 30))
    assert np.all(np.diff(indices) >= 30)


def test_manifest_and_mapping_validation():
    validate_manifest_rows([_row()])
    with pytest.raises(ValueError, match="mapping status"):
        validate_manifest_rows([_row(mapping_status="MADE_UP")])


def test_group_overlap_and_dataset_confound_detection():
    assert group_overlap(["p1", "p2"], ["p2", "p3"]) == {"p2"}
    result = dataset_class_confounds(["a", "a", "b", "b"], ["TUP", "TUP", "TLB", "TLB"])
    assert result["severe"] is True
    mixed = dataset_class_confounds(["a", "b", "a", "b"], ["TUP", "TLB", "TLB", "TUP"])
    assert mixed["severe"] is False


def test_v5_directory_is_isolated_from_v4():
    root = Path(__file__).resolve().parents[1]
    assert (root / "outputs" / "models" / "rgb_native_v4").exists()
    assert (root / "outputs" / "models" / "rgb_native_v4").resolve() != (root / "outputs" / "models" / "rgb_native_v5").resolve()


def test_v5_runner_reload_and_schema_mismatch_rejection(tmp_path):
    model_dir = tmp_path / "rgb_native_v5"
    model_dir.mkdir()
    x0, names = build_rgb_native_v5_features(_pose(), "hybrid")
    x = np.vstack([x0, x0 + 0.01, x0 - 0.01, x0 + 0.02])
    y = np.array(["TUP", "TLB", "TUP", "TLB"])
    model = LogisticRegression(random_state=0).fit(x, y)
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "feature_config.json").write_text(
        json.dumps({"name": "hybrid", "columns": names, "core_landmarks": list(V5_CORE_LANDMARKS)}),
        encoding="utf-8",
    )
    (model_dir / "labels.json").write_text(json.dumps({"source_labels": ["TUP", "TLB"]}), encoding="utf-8")

    runner = RGBNativeV5Runner(model_dir)
    result = runner.predict_world_landmarks(_pose(), image_visibility=np.ones(33))
    assert result["feature_count"] == 30
    assert set(result["probabilities"]) == {"TUP", "TLB"}

    bad_config = {"name": "hybrid", "columns": names[:-1], "core_landmarks": list(V5_CORE_LANDMARKS)}
    (model_dir / "feature_config.json").write_text(json.dumps(bad_config), encoding="utf-8")
    bad_runner = RGBNativeV5Runner(model_dir)
    with pytest.raises(ValueError, match="feature adapter output order"):
        bad_runner.predict_world_landmarks(_pose(), image_visibility=np.ones(33))
