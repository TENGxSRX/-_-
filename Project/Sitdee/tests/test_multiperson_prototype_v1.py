from __future__ import annotations

import json

import numpy as np

from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig
from src.posture_ai.ergonomics.activity_gate import ActivityGateConfig
from src.posture_ai.multiperson import (
    AnonymousPoseTracker,
    MultiPersonRuntime,
    MultiPersonRuntimeConfig,
    MultiPersonTelemetry,
    PoseDetection,
    dashboard_cards,
    frame_to_event,
    legacy_single_person_to_event,
    should_process_frame,
)
from src.posture_ai.pose.landmarks import PoseLandmarks
from src.posture_ai.inference.pipeline import TemporalSmoother
from src.posture_ai.ergonomics.session import SessionAccumulator


def _pose(center_x: float) -> PoseLandmarks:
    values = np.zeros((33, 3), dtype=float)
    values[0] = [center_x, 0.20, 0.0]
    values[11] = [center_x - 0.10, 0.45, 0.0]
    values[12] = [center_x + 0.10, 0.45, 0.0]
    values[23] = [center_x - 0.08, 0.75, 0.0]
    values[24] = [center_x + 0.08, 0.75, 0.0]
    values[25] = [center_x - 0.20, 0.78, 0.0]
    values[26] = [center_x + 0.20, 0.78, 0.0]
    values[27] = [center_x - 0.26, 0.62, 0.0]
    values[28] = [center_x + 0.26, 0.62, 0.0]
    return PoseLandmarks.from_xyz_array(values, visibility=[0.95] * 33)


def _detection(center_x: float) -> PoseDetection:
    pose = _pose(center_x)
    return PoseDetection(pose, pose, pose_quality=0.95, core_valid=True, center=(center_x, 0.55))


class _PositionV4:
    labels = ["TLB", "TUP"]

    def predict_world_landmarks(self, world_landmarks, *, image_visibility=None, visibility_threshold=0.5):
        center_x = float((world_landmarks.by_name("left_shoulder").x + world_landmarks.by_name("right_shoulder").x) / 2.0)
        label = "TLB" if center_x < 0.5 else "TUP"
        return {
            "label": label,
            "display_label": label,
            "confidence": 0.9,
            "probabilities": {"TLB": 0.9 if label == "TLB" else 0.1, "TUP": 0.1 if label == "TLB" else 0.9},
            "feature_count": 15,
        }


def test_two_subjects_and_detector_order_changes_keep_ids():
    tracker = AnonymousPoseTracker(max_people=2, max_association_distance=0.15, max_misses=2, reacquisition_window_seconds=2.0)
    first = tracker.update([_detection(0.25), _detection(0.75)], 0.0)
    assert [item.subject_id for item in first] == ["PERSON_01", "PERSON_02"]
    second = tracker.update([_detection(0.76), _detection(0.24)], 0.1)
    assert tracker.last_assignments == {0: "PERSON_02", 1: "PERSON_01"}
    assert [item.subject_id for item in second] == ["PERSON_01", "PERSON_02"]


def test_temporary_disappearance_reacquires_and_expiration_creates_new_id():
    tracker = AnonymousPoseTracker(max_people=2, max_association_distance=0.15, max_misses=1, reacquisition_window_seconds=2.0)
    tracker.update([_detection(0.25), _detection(0.75)], 0.0)
    lost = tracker.update([_detection(0.75)], 0.1)
    assert tracker.track_state("PERSON_01") == "TEMPORARILY_LOST"
    states = {item.subject_id: item.tracking_state for item in lost}
    assert states == {"PERSON_01": "TEMPORARILY_LOST", "PERSON_02": "ACTIVE"}
    recovered = tracker.update([_detection(0.25), _detection(0.75)], 0.2)
    assert [item.tracking_state for item in recovered] == ["ACTIVE", "ACTIVE"]

    tracker.update([], 0.3)
    tracker.update([], 0.4)
    assert tracker.track_state("PERSON_01") == "EXPIRED"
    new = tracker.update([_detection(0.25)], 0.5)
    assert new[0].subject_id == "PERSON_01"


def test_runtime_keeps_v4_geometry_and_temporal_state_per_person():
    config = PostureEngineConfig(minimum_warning_duration=1.5, high_risk_duration=3.0, minimum_persistence_duration=0.0, activity_gate=ActivityGateConfig(transition_confirmations=1))
    runtime = MultiPersonRuntime(
        _PositionV4(),
        runtime_config=MultiPersonRuntimeConfig(max_people=2, max_misses=1, reacquisition_window_seconds=2.0),
        posture_config=config,
    )
    first = runtime.process_frame([_detection(0.25), _detection(0.75)], 0.0)
    assert [item.v4_prediction for item in first.subjects] == ["TLB", "TUP"]
    second = runtime.process_frame([_detection(0.25), _detection(0.75)], 1.0)
    assert second.subjects[0].temporal_state["duration_seconds"] == 1.0
    assert second.subjects[1].temporal_state["duration_seconds"] == 0.0
    third = runtime.process_frame([_detection(0.25)], 2.0)
    assert third.subjects[0].temporal_state["risk_level"] == "WARNING"
    assert third.subjects[1].temporal_state["risk_level"] == "POSE_LOST"
    assert third.subjects[1].tracking_state == "TEMPORARILY_LOST"
    cards = dashboard_cards(third)
    assert cards[0]["ai_posture"] == "TLB"
    assert cards[1]["state"] == "POSE_LOST"


def test_per_person_calibration_is_not_shared():
    runtime = MultiPersonRuntime(_PositionV4(), posture_config=PostureEngineConfig(activity_gate=ActivityGateConfig(transition_confirmations=1)))
    runtime.process_frame([_detection(0.25), _detection(0.75)], 0.0)
    runtime.set_neutral_baseline("PERSON_01", {"trunk_inclination_deg": 0.0, "torso_lateral_displacement": 0.0, "head_forward_proxy": 0.0, "shoulder_tilt_deg": 0.0})
    assert set(runtime.neutral_baselines) == {"PERSON_01"}
    result = runtime.process_frame([_detection(0.25), _detection(0.75)], 0.1)
    assert result.subjects[0].geometry["neutral_baseline_applied"] is True
    assert result.subjects[1].geometry["neutral_baseline_applied"] is False


def test_multi_person_telemetry_and_legacy_single_person_compatibility():
    runtime = MultiPersonRuntime(_PositionV4())
    result = runtime.process_frame([_detection(0.25), _detection(0.75)], 1.0)
    event = frame_to_event(result, frame_index=7)
    json.dumps(event, ensure_ascii=False, allow_nan=False)
    telemetry = MultiPersonTelemetry(max_events=1)
    telemetry.add_frame(result, frame_index=7)
    telemetry.add_frame(result, frame_index=8, inference_skipped=True)
    assert len(telemetry.events) == 1
    assert telemetry.events[0]["inference_skipped"] is True
    legacy = legacy_single_person_to_event({"timestamp_seconds": 1.0, "pose_visible": True, "label": "TUP"})
    assert legacy["tracking_status"] == "LEGACY_SINGLE_PERSON_COMPATIBILITY"
    assert legacy["subjects"][0]["subject_id"] == "PERSON_01"
    old_session = SessionAccumulator()
    old_session.add({"timestamp_seconds": 0.0, "pose_visible": True, "direction": "TUP", "risk_level": "NORMAL"})
    assert old_session.summary()["records"] == 1


def test_low_spec_scheduler_preserves_real_timestamp_sequence():
    timestamps = [0.00, 0.04, 0.08, 0.12, 0.16, 0.20]
    processed = [index for index, timestamp in enumerate(timestamps) if should_process_frame(index, 3, has_previous_result=index > 0)]
    assert processed == [0, 3]
    assert timestamps[processed[1]] == 0.12
    assert all(timestamps[index] < timestamps[index + 1] for index in range(len(timestamps) - 1))


def test_existing_single_person_smoother_contract_is_unchanged():
    smoother = TemporalSmoother(["TUP", "TLB"], window_size=3, persistence_frames=1)
    assert smoother.update(np.array([0.9, 0.1]))["label"] == "TUP"
