import json
from types import SimpleNamespace

import joblib
import numpy as np

from scripts.run_upper_body_posture_v2 import (
    _display_prediction, invalidate_lost_target, pose_boxes_for_model, pose_points_on_frame,
    selection_status_text,
)
from src.posture_ai.detection.yolo_optional import PersonBox
from src.posture_ai.features.upper_body_posture_v2 import FEATURES
from src.posture_ai.inference.pose_v2_stage import PoseV2Result
from src.posture_ai.inference.webcam_fiveclass_exp import (
    FEATURE_SCHEMA, LABELS, MODEL_ID, WebcamFiveClassExperimentalRunner,
)


class FixedProbabilityModel:
    classes_ = np.asarray(LABELS)

    def predict_proba(self, features):
        return np.tile([0.9, 0.025, 0.025, 0.025, 0.025], (len(features), 1))


def write_artifact(path):
    path.mkdir()
    bundle = {
        "experiment": MODEL_ID, "model": FixedProbabilityModel(), "labels": list(LABELS),
        "feature_schema": FEATURE_SCHEMA, "feature_names": list(FEATURES["NO_HIP"]),
        "unknown_threshold": 0.5,
    }
    joblib.dump(bundle, path / "model.joblib")
    (path / "metadata.json").write_text(json.dumps({"status": "EXPERIMENTAL_ONLY"}), encoding="utf-8")
    (path / "feature_config.json").write_text(json.dumps({"schema": FEATURE_SCHEMA, "features": list(FEATURES["NO_HIP"])}), encoding="utf-8")
    (path / "labels.json").write_text(json.dumps({"labels": list(LABELS), "unknown": "UNKNOWN"}), encoding="utf-8")


def evidence(state="POSE_OK"):
    world = np.zeros((33, 3), dtype=float)
    world[0], world[11], world[12] = [0, -0.8, -0.2], [-0.2, -0.5, 0], [0.2, -0.5, 0]
    image = np.full((33, 3), np.nan)
    image[0], image[11], image[12], image[23], image[24] = [0.5, 0.2, 0], [0.3, 0.4, 0], [0.7, 0.4, 0], [0.35, 0.8, 0], [0.65, 0.8, 0]
    return PoseV2Result(
        track_id=7, bbox=(100, 50, 300, 250), bbox_confidence=0.9,
        pose_state=state, sampling_reason="SAMPLED", pose_available=state == "POSE_OK",
        core_valid=state == "POSE_OK", landmark_visibility={
            "nose": 1.0, "left_shoulder": 1.0, "right_shoulder": 1.0,
            "left_hip": 0.0, "right_hip": 0.0,
        }, world_landmarks_xyz=world.tolist(), image_landmarks_xyz=image.tolist(),
    )


def test_experimental_artifact_loads_exact_schema_and_predicts_real_features(tmp_path):
    model_dir = tmp_path / "model"
    write_artifact(model_dir)
    runner = WebcamFiveClassExperimentalRunner(model_dir)
    prediction = runner.predict(evidence())
    assert prediction["output"] == "TUP" and prediction["confidence"] == 0.9
    assert runner.predict(evidence("POSE_UNAVAILABLE"))["output"] == "UNKNOWN"


def test_only_selected_candidate_is_sent_to_experimental_pose_stage():
    boxes = [PersonBox(0, 0, 100, 200, 0.9, track_id=1), PersonBox(110, 0, 210, 200, 0.8, track_id=2)]
    assert [box.track_id for box in pose_boxes_for_model(boxes, {1, 2}, 2, experimental=True)] == [2]
    assert [box.track_id for box in pose_boxes_for_model(boxes, {1, 2}, None, experimental=True)] == []
    assert [box.track_id for box in pose_boxes_for_model(boxes, {1, 2}, 2, experimental=False)] == [1, 2]


def test_crop_landmarks_map_to_frame_and_invalid_pose_draws_nothing():
    points = pose_points_on_frame(evidence(), frame_size=(400, 300))
    assert points[0] == (200, 90)
    assert points[11] == (160, 130)
    assert pose_points_on_frame(evidence("POSE_UNAVAILABLE"), frame_size=(400, 300)) == []


def test_selection_and_prediction_feedback_are_distinct():
    box = PersonBox(0, 0, 100, 200, 0.9, track_id=7)
    assert selection_status_text(7, [box]) == "SELECTED: TRACK 7"
    assert selection_status_text(7, []) == "TARGET LOST - TRACK 7"
    assert "TUP" in _display_prediction({"output": "TUP", "confidence": 0.9})
    assert "ยังไม่มั่นใจ" in _display_prediction({"output": "UNKNOWN", "confidence": 0.4})


def test_lost_target_clears_stale_pose_and_prediction_without_reassignment():
    prediction, pose, status = {7: "old"}, {7: "old"}, {7: "old"}
    assert invalidate_lost_target(7, {8}, prediction, pose, status)
    assert prediction == pose == status == {}
    assert not invalidate_lost_target(7, {7, 8}, prediction, pose, status)
