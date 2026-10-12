"""Focused V2 contract / fold isolation / persisted inference checks."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from scripts import train_upper_body_posture_v2 as training
from scripts.run_upper_body_posture_v2 import (
    hit_test_track, paired_predictions, profile_for_source, resolve_source,
    select_track_at, selection_status_text,
)
from src.posture_ai.detection.yolo_optional import PersonBox
from src.posture_ai.features.upper_body_posture_v2 import CORE_NAMES, FEATURES, INDICES, build_features, training_hip_dropout
from src.posture_ai.inference.pose_v2_stage import MediaPipeCropPoseBackend, PoseV2Stage, PoseV2Result
from src.posture_ai.inference.upper_body_posture_v1 import UpperBodyPostureRunner, verify_v4
from src.posture_ai.inference.upper_body_posture_v2 import UpperBodyPostureV2Runner


def anchors():
    return np.array([[0, -0.9, -0.1], [-0.2, -0.6, 0], [0.2, -0.6, 0], [-0.15, 0, 0], [0.15, 0, 0]])


def test_unified_runner_resolves_webcam_without_changing_file_or_rtsp_contract():
    assert resolve_source("webcam", 2) == "2"
    assert resolve_source("clip.mp4", 0) == "clip.mp4"
    assert profile_for_source("rtsp", "HIKVISION_MAIN") == "HIKVISION_MAIN"
    assert profile_for_source("clip.mp4", "HIKVISION_MAIN") is None


def test_click_hit_testing_uses_display_to_inference_coordinates():
    boxes = [PersonBox(200, 100, 600, 500, 0.9, track_id=7)]
    assert hit_test_track(boxes, 400, 300, display_size=(1280, 720), inference_size=(1280, 720)) == 7
    assert hit_test_track(boxes, 200, 150, display_size=(640, 360), inference_size=(1280, 720)) == 7


def test_selection_succeeds_and_empty_click_does_not_reassign():
    boxes = [PersonBox(20, 20, 120, 220, 0.9, track_id=1), PersonBox(150, 20, 250, 220, 0.8, track_id=2)]
    selected = select_track_at(None, boxes, 60, 100, display_size=(300, 240), inference_size=(300, 240))
    assert selected == 1
    assert select_track_at(selected, boxes, 290, 230, display_size=(300, 240), inference_size=(300, 240)) == 1


def test_selection_feedback_preserves_identity_and_reports_target_loss():
    box = PersonBox(20, 20, 120, 220, 0.9, track_id=1)
    assert selection_status_text(None, [box]) == "NO TARGET SELECTED - CLICK A PERSON"
    assert selection_status_text(1, [box]) == "SELECTED: TRACK 1"
    assert selection_status_text(1, []) == "TARGET LOST - TRACK 1"
    assert selection_status_text(1, [PersonBox(20, 20, 120, 220, 0.9, track_id=2)]) == "TARGET LOST - TRACK 1"


def test_no_lower_body_dependency_deterministic_translation_scale_and_optional_hips():
    full = np.full((33, 3), np.nan)
    full[list(INDICES)] = anchors()
    for condition in ("NO_HIP", "HIP_DROPOUT_ROBUST"):
        expected = build_features(full, condition)
        np.testing.assert_array_equal(build_features(full, condition), expected)
        full[25:] = np.inf
        np.testing.assert_array_equal(build_features(full, condition), expected)
        np.testing.assert_allclose(build_features(full * 2 + [3, 4, 5], condition), expected, atol=1e-12)
        for missing in ((23,), (24,), (23, 24)):
            view = full.copy()
            view[list(missing)] = np.nan
            result = build_features(view, condition)
            assert np.isfinite(result).all()
            np.testing.assert_array_equal(result[:, :6], expected[:, :6])
            if condition == "HIP_DROPOUT_ROBUST":
                assert result[0, -1] == 0 and np.all(result[0, 6:-1] == 0)


def test_dropout_is_training_only_and_labels_originals_validation_unchanged(monkeypatch):
    xyz = np.stack([anchors()] * 8)
    y = np.array(["TLB", "TUP"] * 4)
    groups = np.repeat(["1", "2", "3", "4"], 2)
    original, original_labels = xyz.copy(), y.copy()
    seen = []
    real = training.training_hip_dropout

    def spy(train_xyz, train_y):
        seen.append(len(train_xyz))
        return real(train_xyz, train_y)

    monkeypatch.setattr(training, "training_hip_dropout", spy)
    for train, valid in training.GroupKFold(2).split(xyz, y, groups):
        assert not set(groups[train]) & set(groups[valid])
        _, fitted_x = training.fit_training_fold(LogisticRegression(), xyz[train], y[train], "HIP_DROPOUT_ROBUST")
        assert len(fitted_x) == len(train) * 2
        np.testing.assert_array_equal(xyz[valid], original[valid])
    assert seen == [4, 4]
    np.testing.assert_array_equal(xyz, original)
    np.testing.assert_array_equal(y, original_labels)
    augmented, labels = training_hip_dropout(xyz, y)
    assert augmented.shape == (16, 5, 3)
    np.testing.assert_array_equal(labels, np.tile(y, 2))


def test_real_pose_stage_accepts_missing_hips_and_invalid_core_rejects():
    points = [SimpleNamespace(x=np.nan, y=np.nan, z=np.nan, visibility=0) for _ in range(33)]
    for index, xyz in zip(INDICES[:3], anchors()[:3]):
        points[index] = SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2], visibility=1)
    backend = MediaPipeCropPoseBackend.__new__(MediaPipeCropPoseBackend)
    backend.visibility_threshold, backend.core_landmark_names = 0.5, CORE_NAMES
    backend._cv2 = SimpleNamespace(COLOR_BGR2RGB=0, cvtColor=lambda image, code: image)
    backend._mp = SimpleNamespace(ImageFormat=SimpleNamespace(SRGB=0), Image=lambda **kw: kw)
    backend._landmarker = SimpleNamespace(detect=lambda image: SimpleNamespace(pose_landmarks=[points], pose_world_landmarks=[points]))
    stage = PoseV2Stage(backend)
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    box = PersonBox(30, 30, 260, 260, 0.9, track_id=1)
    evidence = stage.process(frame, [box], 0)[0]
    assert evidence.pose_state == "POSE_OK" and evidence.core_valid
    runner = UpperBodyPostureV2Runner()
    p = runner.predict(evidence)
    assert p["reason"] not in ("INVALID_NOSE_OR_SHOULDERS", "LOW_CORE_VISIBILITY", "MISSING_HIPS")
    v1 = UpperBodyPostureRunner()
    hips, p1, p2 = paired_predictions(evidence, v1, runner)
    assert hips == "both_hidden" and p1["reason"] == "MISSING_HIPS" and p2 == p
    points[11].visibility = 0
    invalid = stage.process(frame, [box], 1)[0]
    assert invalid.pose_state == "POSE_CORE_INVALID"
    assert runner.predict(invalid)["output"] == "UNKNOWN"
    # Runner independently checks coordinates and confidence, even with a stale valid flag.
    assert runner.predict(replace(invalid, pose_state="POSE_OK", core_valid=True))["reason"] == "INVALID_NOSE_OR_SHOULDERS"
    for index in range(3):
        bad = anchors()
        bad[index] = np.nan
        with pytest.raises(ValueError, match="INVALID"):
            build_features(bad)
    with pytest.raises(ValueError, match="DEGENERATE"):
        build_features(np.zeros((5, 3)))


def test_persisted_candidate_reload_and_frozen_v1_v4(tmp_path):
    path = Path("outputs/models/upper_body_posture_v2")
    report = json.loads((path / "metrics.json").read_text())
    assert training.digest(path / "candidate.joblib") == report["candidate_sha256"]
    assert training.frozen_hashes() == report["frozen_hashes"]
    assert verify_v4() == "af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd"
    runner = UpperBodyPostureV2Runner()
    joblib.dump(runner.bundle, tmp_path / "candidate.joblib")
    loaded = UpperBodyPostureV2Runner(tmp_path)
    x = build_features(anchors(), runner.bundle["condition"])
    np.testing.assert_array_equal(runner.bundle["model"].predict(x), loaded.bundle["model"].predict(x))
    assert tuple(loaded.bundle["features"]) == FEATURES[loaded.bundle["condition"]]
    for filename in ("src/posture_ai/features/upper_body_posture_v1.py", "src/posture_ai/inference/upper_body_posture_v1.py", "scripts/train_upper_body_posture_v1.py", "scripts/run_upper_body_posture.py"):
        frozen = subprocess.check_output(["git", "show", f"{report['starting_commit']}:{filename}"])
        assert Path(filename).read_bytes().replace(b"\r\n", b"\n") == frozen.replace(b"\r\n", b"\n")


def test_selected_model_refusal_retains_ood_and_ambiguity():
    runner = UpperBodyPostureV2Runner()
    from src.posture_ai.inference.upper_body_posture_v2 import classify_features
    extreme = np.full((1, len(runner.bundle["features"])), 1e9)
    assert classify_features(runner.bundle, extreme)["reason"] == "POSSIBLE_OOD_EXPERIMENTAL"
    assert runner.predict(SimpleNamespace(pose_state="POSE_AMBIGUOUS", core_valid=False))["output"] == "UNKNOWN"


def test_dropout_sentinels_never_expand_observed_hip_ood_ranges():
    x = np.ones((2, len(FEATURES["HIP_DROPOUT_ROBUST"])))
    x[0, 6:-1] = 5
    x[1, 6:] = 0
    bundle = training.make_bundle(None, [], x, ["TLB", "TUP"], "HIP_DROPOUT_ROBUST")
    np.testing.assert_array_equal(bundle["minimum"][6:-1], 5)
    np.testing.assert_array_equal(bundle["maximum"][6:-1], 5)


def test_persisted_real_runner_classifies_an_in_domain_pose_without_hips():
    runner = UpperBodyPostureV2Runner()
    cache = np.load("datasets/processed/rgb_native/features.npz")["world"].reshape(-1, 5, 3)
    from src.posture_ai.inference.upper_body_posture_v2 import classify_batch
    records = classify_batch(runner.bundle, build_features(cache, runner.bundle["condition"]))
    i = next(i for i, r in enumerate(records) if r["output"] != "UNKNOWN")
    world = np.full((33, 3), np.nan)
    world[list(INDICES[:3])] = cache[i, :3]
    evidence = PoseV2Result(track_id=1, bbox=(0, 0, 100, 200), bbox_confidence=0.9, pose_state="POSE_OK", sampling_reason="SAMPLED", pose_available=True, core_valid=True,
                            landmark_visibility=dict.fromkeys(CORE_NAMES, 1.0), world_landmarks_xyz=world.tolist())
    assert runner.predict(evidence) == records[i]
