from types import SimpleNamespace

import numpy as np
import pytest

from src.posture_ai.features.upper_body_posture_v1 import CORE_NAMES, CORE_INDICES, FEATURE_NAMES, SCHEMA, build_features
from src.posture_ai.inference.upper_body_posture_v1 import UpperBodyPostureRunner, classify_features, verify_v4
from src.posture_ai.inference.pose_v2_stage import MediaPipeCropPoseBackend
from scripts.run_upper_body_posture import human_record
from scripts.train_upper_body_posture_v1 import metrics


def pose():
    xyz = np.full((33, 3), np.nan)
    xyz[list(CORE_INDICES)] = [[-0.2, -0.6, 0], [0.2, -0.6, 0], [-0.15, 0, 0], [0.15, 0, 0]]
    return xyz


class FixedModel:
    def __init__(self, label):
        self.label = label

    def predict(self, x):
        return np.array([self.label] * len(x))


def bundle(label="TUP"):
    return {"schema": SCHEMA, "features": FEATURE_NAMES, "labels": ["TLB", "TUP"],
            "model": FixedModel(label), "committee": [FixedModel(label)],
            "minimum": np.full(8, -10), "maximum": np.full(8, 10)}


def test_features_ignore_every_non_core_landmark_and_translation_scale():
    xyz = pose()
    expected = build_features(xyz)
    assert expected.shape == (1, 8)
    changed = xyz.copy()
    changed[25:] = [[1e9, -1e9, np.nan]] * 8
    changed[0:11] = np.inf
    np.testing.assert_allclose(build_features(changed), expected)
    np.testing.assert_allclose(build_features(xyz * 3 + [2, 4, -7]), expected, atol=1e-12)
    xyz[23] = np.nan
    with pytest.raises(ValueError, match="INVALID"):
        build_features(xyz)
    with pytest.raises(ValueError, match="DEGENERATE"):
        build_features(np.zeros((4, 3)))


def test_mapping_quality_ambiguity_uncertainty_and_ood_refuse():
    x = build_features(pose())
    b = bundle()
    assert classify_features(b, x)["output"] == "CORRECT_SITTING"
    assert classify_features(bundle("TLB"), x)["subtype"] == "BACKWARD_OR_RECLINED"
    b["committee"] = [FixedModel("TLB")]
    assert classify_features(b, x)["output"] == "UNKNOWN"
    b = bundle()
    b["maximum"] = np.full(8, -100)
    assert classify_features(b, x)["reason"] == "POSSIBLE_OOD_EXPERIMENTAL"
    runner = UpperBodyPostureRunner.__new__(UpperBodyPostureRunner)
    runner.bundle = bundle()
    evidence = SimpleNamespace(pose_state="POSE_AMBIGUOUS", core_valid=True)
    assert runner.predict(evidence)["output"] == "UNKNOWN"
    evidence.pose_state = "POSE_OK"
    evidence.landmark_visibility = dict.fromkeys(CORE_NAMES, 1.0)
    evidence.world_landmarks_xyz = pose().tolist()
    assert runner.predict(evidence)["output"] == "CORRECT_SITTING"
    evidence.landmark_visibility["left_hip"] = 0.1
    assert runner.predict(evidence)["reason"] == "LOW_CORE_VISIBILITY"


def test_blind_gt_is_separate_validated_and_uses_only_four_points():
    e = SimpleNamespace(world_landmarks_xyz=pose().tolist(), landmark_visibility=dict.fromkeys(CORE_NAMES, 1), track_id=7)
    r = human_record("P001", e, "UPRIGHT", 10, "2026-09-17T00:00:00Z")
    assert r["record_type"] == "HUMAN_GROUND_TRUTH" and r["model_prediction"] is None
    assert len(r["core_xyz"]) == 4 and r["raw_image_saved"] is False
    with pytest.raises(ValueError):
        human_record("John", e, "UPRIGHT", 10, "now")
    with pytest.raises(ValueError):
        human_record("P001", e, "SLOUCH", 10, "now")


def test_confusion_counts_off_ontology_predictions_and_frozen_v4():
    r = metrics(np.array(["TUP", "TLB"]), np.array(["TLF", "TUP"]), np.array(["1", "1"]), ["TLB", "TUP"])
    assert sum(map(sum, r["confusion_matrix"])) == 2
    assert r["binary"]["correct_recall"] == 0
    assert verify_v4() == "af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd"


def test_upper_body_backend_does_not_require_head_or_lower_body_visibility():
    points = [SimpleNamespace(x=0, y=0, z=0, visibility=0) for _ in range(33)]
    for i in CORE_INDICES:
        points[i] = SimpleNamespace(x=1, y=1, z=1, visibility=1)
    backend = MediaPipeCropPoseBackend.__new__(MediaPipeCropPoseBackend)
    backend.visibility_threshold = 0.5
    backend.core_landmark_names = CORE_NAMES
    backend._cv2 = SimpleNamespace(COLOR_BGR2RGB=0, cvtColor=lambda image, code: image)
    backend._mp = SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=0), Image=lambda **kw: kw)
    backend._landmarker = SimpleNamespace(detect=lambda image: SimpleNamespace(pose_landmarks=[points], pose_world_landmarks=[points]))
    assert backend.detect(np.zeros((10, 10, 3), dtype=np.uint8))[0].core_valid
    backend.core_landmark_names = ("nose", *CORE_NAMES)
    assert not backend.detect(np.zeros((10, 10, 3), dtype=np.uint8))[0].core_valid
