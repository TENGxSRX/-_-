from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig
from src.posture_ai.ergonomics.calibration import NeutralBaselineCalibrator, relative_geometry
from src.posture_ai.inference.webcam_v1 import (
    LiveFrameProcessor,
    LiveSessionAccumulator,
    StableCoreGate,
    ValidationController,
    framing_assessment,
    live_video_path,
    parse_camera_source,
    prepare_display_frame,
    release_live_resources,
)


class FakePoint:
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _pose(*, visibility: float = 1.0):
    points = [FakePoint(0.5, 0.5, 0.0, visibility) for _ in range(33)]
    points[0] = FakePoint(0.5, 0.25, 0.0, visibility)
    points[11] = FakePoint(0.35, 0.45, 0.0, visibility)
    points[12] = FakePoint(0.65, 0.45, 0.0, visibility)
    points[23] = FakePoint(0.40, 0.75, 0.0, visibility)
    points[24] = FakePoint(0.60, 0.75, 0.0, visibility)
    return points


class FakeLandmarker:
    def __init__(self, image_pose, world_pose=None):
        self.image_pose = image_pose
        self.world_pose = world_pose or image_pose

    def detect_for_video(self, image, timestamp_ms):
        return SimpleNamespace(pose_landmarks=[self.image_pose], pose_world_landmarks=[self.world_pose])


class FakeRunner:
    labels = ["TLB", "TUP"]

    def __init__(self, label="TUP", confidence=0.99):
        self.label = label
        self.confidence = confidence

    def predict_world_landmarks(self, world_landmarks, *, image_visibility, visibility_threshold):
        other = "TLB" if self.label == "TUP" else "TUP"
        return {
            "label": self.label,
            "display_label": "Upright" if self.label == "TUP" else "Lean backward",
            "confidence": self.confidence,
            "probabilities": {"TLB": 1.0 if self.label == "TLB" else 0.01, "TUP": 1.0 if self.label == "TUP" else 0.01},
            "feature_count": 15,
            "adapter_status": "RGB_NATIVE_PIPELINE_MATCHED_FOR_TRAINING_SOURCE",
        }


def test_camera_source_parser_only_accepts_supported_indices():
    assert parse_camera_source("0") == 0
    assert parse_camera_source(2) == 2
    with pytest.raises(ValueError):
        parse_camera_source("3")
    with pytest.raises(ValueError):
        parse_camera_source("camera")


def test_validation_controller_has_beginner_sequence_and_timed_end():
    controller = ValidationController(step_seconds=1.0, poor_hold_seconds=2.0)
    assert controller.current(0.1)["phase"] == "SETUP"
    controller.start(0.0)
    assert controller.current(0.1)["phase"] == "WAITING"
    controller.advance(0.1)
    assert controller.current(0.1)["phase"] == "PREPARE"
    assert controller.current(3.1)["phase"] == "ACTIVE"
    controller.observe_input(3.1, input_good=True)
    controller.observe_input(4.1, input_good=True)
    controller.observe_input(4.2, input_good=True)
    assert controller.current(4.2)["phase"] == "COMPLETE"
    assert controller.current(5.0)["phase"] == "WAITING"
    assert controller.current(5.0)["instruction"] == "เอนตัวไปด้านหลัง"


def test_stable_core_gate_requires_continuous_good_input():
    gate = StableCoreGate(required_seconds=1.0)
    assert not gate.update(accepted=True, timestamp_seconds=0.0)["ready"]
    assert not gate.update(accepted=True, timestamp_seconds=0.6)["ready"]
    assert gate.update(accepted=True, timestamp_seconds=1.1)["ready"]
    assert not gate.update(accepted=False, timestamp_seconds=1.2)["ready"]


def test_neutral_baseline_calibration_and_relative_geometry():
    geometry = {
        "valid": True,
        "trunk_vector": [0.1, -0.5, -0.2],
        "trunk_inclination_deg": 5.0,
        "torso_lateral_displacement": 0.1,
        "head_forward_proxy": -0.6,
        "shoulder_tilt_deg": 175.0,
    }
    calibrator = NeutralBaselineCalibrator(required_seconds=2.0, max_gap_seconds=1.1)
    calibrator.start(0.0)
    calibrator.observe(geometry, accepted=True, timestamp_seconds=0.0)
    calibrator.observe(geometry, accepted=True, timestamp_seconds=1.0)
    status = calibrator.observe(geometry, accepted=True, timestamp_seconds=2.0)
    assert status["completed"]
    assert calibrator.baseline is not None
    relative = relative_geometry(geometry, calibrator.baseline)
    assert relative["neutral_baseline_applied"]
    assert abs(relative["head_forward_proxy"]) < 1e-9
    assert abs(relative["torso_lateral_displacement"]) < 1e-9
    calibrator.reset()
    assert calibrator.baseline is None
    assert not calibrator.completed


def test_validation_timer_pauses_when_core_input_is_lost():
    controller = ValidationController(step_seconds=5.0)
    controller.start(0.0)
    controller.advance(0.0)
    assert controller.current(3.0)["phase"] == "ACTIVE"
    controller.observe_input(3.0, input_good=True)
    controller.observe_input(4.0, input_good=False)
    controller.observe_input(10.0, input_good=False)
    active = controller.current(10.0)
    assert active["phase"] == "ACTIVE"
    assert active["timer_paused"]
    assert active["remaining_seconds"] == 4.0


def test_out_of_frame_step_counts_only_while_pose_is_absent():
    controller = ValidationController(
        step_seconds=1.0,
        preparation_seconds=0.0,
        completion_seconds=0.0,
    )
    controller.start(0.0)
    for _ in range(6):
        controller.advance(0.0)  # waiting -> prepare
        controller.advance(0.0)  # prepare -> active
        controller.advance(0.0)  # active -> complete
        controller.advance(0.0)  # complete -> next waiting

    assert controller.current(0.0)["step"] == 7
    controller.advance(0.0)
    assert controller.current(0.0)["phase"] == "ACTIVE"

    visible = controller.observe_input(0.0, input_good=True)
    visible = controller.observe_input(0.5, input_good=True)
    assert visible["timer_paused"]
    assert visible["input_state"] == "PERSON STILL VISIBLE"
    assert controller.current(0.5)["remaining_seconds"] == 1.0

    absent = controller.observe_input(0.5, input_good=False)
    absent = controller.observe_input(1.6, input_good=False)
    assert not absent["timer_paused"]
    assert absent["input_state"] == "OUT OF FRAME"
    next_step = controller.current(1.6)
    assert next_step["phase"] == "WAITING"
    assert next_step["step"] == 8


def test_step_eight_requires_reacquisition_before_space():
    controller = ValidationController(step_seconds=1.0)
    controller.start(0.0)
    for _ in range(7):
        controller.advance(0.0)  # waiting -> prepare
        controller.advance(0.0)  # prepare -> active
        controller.advance(0.0)  # active -> complete
        controller.advance(1.0)  # complete -> next waiting
    step = controller.current(1.0)
    assert step["phase"] == "WAITING"
    assert step["step"] == 8
    assert step["requires_reacquisition"]


def test_framing_assistant_distinguishes_core_visibility_from_pose_detection():
    pose = _pose()
    assert framing_assessment(pose)["ready"]
    pose[24] = FakePoint(pose[24].x, pose[24].y, pose[24].z, 0.1)
    assessment = framing_assessment(pose)
    assert not assessment["ready"]
    assert assessment["missing_landmarks"] == ["right_hip"]
    assert "สะโพกขวา" in assessment["message"]


def test_processing_frame_and_mirrored_display_frame_are_separate():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[:, 0, :] = 10
    original = frame.copy()
    mirrored = prepare_display_frame(frame, mirror=True)
    assert np.array_equal(frame, original)
    assert np.array_equal(mirrored[:, -1, :], original[:, 0, :])
    assert np.array_equal(prepare_display_frame(frame, mirror=False), original)


def test_live_processor_keeps_ai_and_geometry_separate():
    pose = _pose()
    processor = LiveFrameProcessor(
        FakeRunner("TUP"),
        FakeLandmarker(pose),
        PostureEngineConfig(smoothing_window=1, minimum_confidence=0.5),
        calibration=True,
    )
    record, image_pose = processor.process(np.zeros((120, 160, 3), dtype=np.uint8), frame_index=0, timestamp_seconds=0.0, fps=30.0)
    assert image_pose is not None
    assert record["ai_posture"] == "UPRIGHT"
    assert record["ai_label"] == "TUP"
    assert "forward_indicator" in record
    assert "left_indicator" in record
    assert "right_indicator" in record
    assert "calibration" in record
    assert record["risk_state"] == "NORMAL"
    assert record["ai_probabilities"]["TUP"] == 1.0


def test_live_processor_rejects_low_core_visibility_without_imputation():
    pose = _pose()
    pose[23] = FakePoint(pose[23].x, pose[23].y, pose[23].z, 0.1)
    processor = LiveFrameProcessor(
        FakeRunner(),
        FakeLandmarker(pose),
        PostureEngineConfig(smoothing_window=1),
    )
    record, _ = processor.process(np.zeros((120, 160, 3), dtype=np.uint8), frame_index=0, timestamp_seconds=0.0, fps=30.0)
    assert record["status"] == "CORE_LANDMARKS_NOT_ACCEPTED"
    assert not record["core_landmark_accepted"]
    assert record["ai_label"] is None
    assert record["risk_state"] == "POSE_LOST"


def test_live_temporal_warning_uses_configured_engine():
    pose = _pose()
    config = PostureEngineConfig(smoothing_window=1, minimum_warning_duration=1.0, high_risk_duration=5.0, minimum_confidence=0.5)
    processor = LiveFrameProcessor(FakeRunner("TLB"), FakeLandmarker(pose), config)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    first, _ = processor.process(frame, frame_index=0, timestamp_seconds=0.0, fps=30.0)
    second, _ = processor.process(frame, frame_index=1, timestamp_seconds=1.0, fps=30.0)
    assert first["risk_state"] == "OBSERVE"
    assert second["risk_state"] == "WARNING"
    assert second["warning_event"]


def test_live_session_summary_reports_requested_percentages():
    accumulator = LiveSessionAccumulator(frame_interval_seconds=1.0)
    accumulator.add({
        "timestamp": 0.0, "pose_detected": True, "core_landmark_accepted": True,
        "ai_posture": "UPRIGHT", "ai_label": "TUP", "forward_indicator": False,
        "left_indicator": False, "right_indicator": False, "risk_state": "NORMAL",
        "warning_event": False, "continuous_bad_duration": 0.0, "fps": 30.0,
    })
    accumulator.add({
        "timestamp": 1.0, "pose_detected": False, "core_landmark_accepted": False,
        "ai_posture": None, "ai_label": None, "forward_indicator": False,
        "left_indicator": False, "right_indicator": False, "risk_state": "POSE_LOST",
        "warning_event": False, "continuous_bad_duration": 0.0, "fps": 30.0,
    })
    summary = accumulator.summary(total_duration_seconds=2.0)
    assert summary["valid_pose_time_seconds"] == 1.0
    assert summary["pose_lost_percentage"] == 50.0
    assert summary["upright_percentage"] == 100.0
    assert summary["lean_backward_percentage"] == 0.0
    assert summary["pose_detection_percentage"] == 50.0
    assert summary["core_landmark_acceptance_percentage"] == 50.0
    assert summary["input_quality_status"] == "NEEDS_CALIBRATION"


def test_live_session_summary_quality_status_is_explicit():
    accumulator = LiveSessionAccumulator(frame_interval_seconds=1.0)
    for timestamp in range(4):
        accumulator.add({
            "timestamp": float(timestamp), "pose_detected": True, "core_landmark_accepted": True,
            "ai_posture": "UPRIGHT", "ai_label": "TUP", "forward_indicator": False,
            "left_indicator": False, "right_indicator": False, "risk_state": "NORMAL",
            "warning_event": False, "continuous_bad_duration": 0.0, "fps": 30.0,
        })
    summary = accumulator.summary(total_duration_seconds=4.0)
    assert summary["input_quality_status"] == "PASS"
    assert summary["pose_detection_percentage"] == 100.0
    assert summary["core_landmark_acceptance_percentage"] == 100.0


def test_recording_path_is_under_live_video_output():
    assert str(live_video_path("outputs/live", "session-1")).endswith("outputs\\live\\videos\\webcam_session-1.mp4")


def test_clean_shutdown_releases_partial_startup_resources(monkeypatch):
    class Resource:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

        def release(self):
            self.closed = True

    capture, landmarker, writer = Resource(), Resource(), Resource()
    destroyed = []
    monkeypatch.setattr("src.posture_ai.inference.webcam_v1.cv2.destroyAllWindows", lambda: destroyed.append(True))
    release_live_resources(capture, landmarker, writer)
    assert capture.closed and landmarker.closed and writer.closed
    assert destroyed == [True]
