from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.posture_ai.data.multiposture import load_multiposture, validate_multiposture
from src.posture_ai.features.pose_features import (
    LANDMARK_NAMES,
    adapt_mediapipe_landmarks,
    build_features,
    expected_feature_columns,
)


ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "datasets" / "raw" / "multiposture" / "data.csv"


class DummyLandmark:
    def __init__(self, index: int, visibility: float = 1.0):
        self.x = index / 100.0
        self.y = index / 200.0
        self.z = -index / 300.0
        self.visibility = visibility


def test_multiposture_source_schema_and_audit():
    frame = load_multiposture(CSV)
    facts = validate_multiposture(frame)
    assert facts["rows"] == 4794
    assert facts["columns"] == 102
    assert len(facts["subjects"]) == 13
    assert facts["duplicate_rows"] == 0
    assert facts["non_finite_feature_cells"] == 0
    assert list(frame.columns) == ["subject", "upperbody_label", "lowerbody_label", *expected_feature_columns()]


def test_all_feature_configurations_have_stable_shapes_and_names():
    frame = load_multiposture(CSV).head(4)
    shapes = {}
    for config in ("raw", "normalized", "engineered", "hybrid"):
        values, names = build_features(frame, config)
        assert values.shape[0] == 4
        assert values.shape[1] == len(names)
        assert np.isfinite(values).all()
        shapes[config] = values.shape[1]
    assert shapes["raw"] == 99
    assert shapes["normalized"] == 99
    assert shapes["engineered"] > 10
    assert shapes["hybrid"] == shapes["raw"] + shapes["engineered"]


def test_mediapipe_adapter_preserves_33_landmark_xyz_schema():
    values = adapt_mediapipe_landmarks([DummyLandmark(index) for index in range(33)])
    assert values.shape == (1, 99)
    assert np.isfinite(values).all()
    assert len(LANDMARK_NAMES) == 33


def test_low_visibility_landmark_is_rejected():
    landmarks = [DummyLandmark(index) for index in range(33)]
    landmarks[11].visibility = 0.1
    with pytest.raises(ValueError, match="visibility"):
        adapt_mediapipe_landmarks(landmarks, visibility_threshold=0.5)

