from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from src.posture_ai.evaluation.splitting import grouped_holdout_split
from src.posture_ai.features.upperbody_v2 import (
    CORE_REQUIRED_LANDMARKS,
    adapt_mediapipe_to_upperbody_v2,
    build_upperbody_features,
    upperbody_feature_names,
)
from src.posture_ai.pose.landmarks import LANDMARK_INDEX, LANDMARK_NAMES, PoseLandmark, PoseLandmarks
from src.posture_ai.pose.robustness import evaluate_visibility_policy, pose_quality_score


ROOT = Path(__file__).resolve().parents[1]
RAW_MULTIPOSTURE = ROOT / "datasets" / "raw" / "multiposture" / "data.csv"
V1_DIR = ROOT / "outputs" / "models" / "multiposture_v1"
V2_DIR = ROOT / "outputs" / "models" / "multiposture_upperbody_v2"


def _pose(*, core_visibility: float = 1.0, low_value_visibility: float = 1.0, missing_core: str | None = None) -> PoseLandmarks:
    points = [PoseLandmark(0.0, 0.0, 0.0, low_value_visibility) for _ in LANDMARK_NAMES]
    values = {
        "nose": (0.0, -1.0, 0.0),
        "left_shoulder": (-0.5, 0.0, 0.0),
        "right_shoulder": (0.5, 0.0, 0.0),
        "left_hip": (-0.4, 1.0, 0.0),
        "right_hip": (0.4, 1.0, 0.0),
    }
    for name, (x, y, z) in values.items():
        points[LANDMARK_INDEX[name]] = PoseLandmark(x, y, z, core_visibility)
    if missing_core is not None:
        points[LANDMARK_INDEX[missing_core]] = None
    return PoseLandmarks(tuple(points))


def test_visibility_policies_are_explicit_and_core_robust():
    pose = _pose(core_visibility=1.0, low_value_visibility=0.05)
    strict = evaluate_visibility_policy(pose, "strict")
    core = evaluate_visibility_policy(pose, "core-only")
    tiered = evaluate_visibility_policy(pose, "tiered")
    quality = evaluate_visibility_policy(pose, "quality")

    assert not strict["accepted"]
    assert core["accepted"]
    assert tiered["accepted"]
    assert quality["accepted"]
    assert "optional_low_visibility:right_elbow" in tiered["warnings"]
    assert 0.0 <= pose_quality_score(pose)["score"] <= 1.0


def test_missing_core_landmark_is_rejected_without_imputation():
    pose = _pose(missing_core="right_hip")
    result = evaluate_visibility_policy(pose, "core-only")
    assert not result["accepted"]
    assert "right_hip" in result["missing_names"]


def test_upperbody_v2_feature_tracks_have_stable_order_and_adapter_schema():
    pose = _pose()
    for config, expected_count in (("raw", 15), ("normalized", 15), ("geometry", 15), ("hybrid", 30)):
        values, names = build_upperbody_features(pose, config, already_hip_centered=False)
        assert values.shape == (1, expected_count)
        assert names == upperbody_feature_names(config)
        assert np.isfinite(values).all()
        adapted, adapted_names = adapt_mediapipe_to_upperbody_v2(pose, config)
        assert adapted.shape == (1, expected_count)
        assert adapted_names == names


def test_grouped_split_and_v2_artifacts_are_isolated_from_v1():
    groups = np.repeat(np.arange(13), 3)
    split = grouped_holdout_split(groups, test_groups=2, validation_groups=2, random_state=42)
    assert set(groups[split["train"]]).isdisjoint(set(groups[split["validation"]]))
    assert set(groups[split["train"]]).isdisjoint(set(groups[split["test"]]))
    assert set(groups[split["validation"]]).isdisjoint(set(groups[split["test"]]))

    v1_meta = json.loads((V1_DIR / "metadata.json").read_text(encoding="utf-8"))
    v2_meta = json.loads((V2_DIR / "metadata.json").read_text(encoding="utf-8"))
    assert V1_DIR != V2_DIR
    assert v1_meta["feature_count"] == 126
    assert v2_meta["model_family"] == "multiposture_upperbody_v2_experimental"
    assert set(v2_meta["core_required_landmarks"]) == set(CORE_REQUIRED_LANDMARKS)
    assert len(v2_meta["feature_columns"]) == v2_meta["feature_count"]
    model = joblib.load(V2_DIR / "model.joblib")
    labels = json.loads((V2_DIR / "labels.json").read_text(encoding="utf-8"))["source_labels"]
    assert list(model.classes_) == labels

    expected_sha = v1_meta["dataset_sha256"]
    digest = hashlib.sha256(RAW_MULTIPOSTURE.read_bytes()).hexdigest()
    assert digest == expected_sha
