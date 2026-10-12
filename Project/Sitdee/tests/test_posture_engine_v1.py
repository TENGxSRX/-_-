from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.posture_ai.ergonomics.geometry import GEOMETRY_FEATURE_NAMES, calculate_geometry, geometry_feature_vector
from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig, TemporalRiskEngine
from src.posture_ai.ergonomics.session import SessionAccumulator, write_jsonl, write_summary
from src.posture_ai.pose.adapter import adapt_mediapipe_to_multiposture, validate_model_feature_schema
from src.posture_ai.pose.landmarks import PoseLandmark, PoseLandmarks
from src.posture_ai.pose.validation import validate_pose_landmarks


class Point:
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def _pose(*, shoulder_shift: float = 0.0) -> PoseLandmarks:
    points = [Point(0.0, 0.0, 0.0) for _ in range(33)]
    points[0] = Point(0.0, -0.5, 0.1)
    points[11] = Point(-0.2 + shoulder_shift, 0.0, 0.0)
    points[12] = Point(0.2 + shoulder_shift, 0.0, 0.0)
    points[23] = Point(-0.15, 1.0, 0.0)
    points[24] = Point(0.15, 1.0, 0.0)
    return PoseLandmarks.from_sequence(points)


def _probability(direction: str) -> np.ndarray:
    labels = ["TLB", "TLF", "TLL", "TLR", "TUP"]
    values = np.full(5, 0.01, dtype=float)
    values[labels.index(direction)] = 0.96
    return values


def test_canonical_pose_validation_and_missing_handling():
    pose = _pose()
    result = validate_pose_landmarks(pose, visibility_threshold=0.5)
    assert result.valid
    missing = list(pose.points)
    missing[11] = None
    result = validate_pose_landmarks(PoseLandmarks(tuple(missing)), required_names=("left_shoulder",))
    assert not result.valid
    assert result.missing_landmarks == ("left_shoulder",)


def test_multiposture_adapter_exact_126_feature_schema():
    adapted = adapt_mediapipe_to_multiposture(_pose())
    assert adapted.values.shape == (1, 126)
    assert len(adapted.names) == 126
    assert adapted.schema_identity_verified
    assert not adapted.semantic_compatibility_validated
    assert validate_model_feature_schema("outputs/models/multiposture_v1")["exact_name_order_match"]


def test_geometry_midpoints_and_trunk_angle():
    upright = calculate_geometry(_pose())
    shifted = calculate_geometry(_pose(shoulder_shift=0.5))
    assert upright["valid"]
    assert abs(float(upright["trunk_inclination_deg"])) < 1e-8
    assert abs(float(upright["shoulder_tilt_deg"])) < 1e-8
    assert abs(float(shifted["trunk_inclination_deg"]) - 26.565051) < 1e-4
    assert float(shifted["torso_lateral_displacement"]) > 0.0
    tilted_points = list(_pose().points)
    tilted_points[12] = Point(0.2, 0.2, 0.0)
    tilted = calculate_geometry(PoseLandmarks(tuple(tilted_points)))
    assert abs(float(tilted["shoulder_tilt_deg"]) - 26.565051) < 1e-4


def test_geometry_feature_order_is_stable():
    vector, names = geometry_feature_vector(_pose())
    assert vector.shape == (1, len(GEOMETRY_FEATURE_NAMES))
    assert names == list(GEOMETRY_FEATURE_NAMES)
    assert np.isfinite(vector).all()


def test_temporal_warning_high_risk_recovery_and_cooldown():
    config = PostureEngineConfig(
        minimum_warning_duration=2.0,
        high_risk_duration=4.0,
        smoothing_window=1,
        recovery_duration=1.0,
        warning_cooldown=10.0,
    )
    engine = TemporalRiskEngine(config)
    assert engine.update(0.0, pose_visible=True, probabilities=_probability("TLF"))["risk_level"] == "OBSERVE"
    assert engine.update(1.0, pose_visible=True, probabilities=_probability("TLF"))["risk_level"] == "OBSERVE"
    warning = engine.update(2.0, pose_visible=True, probabilities=_probability("TLF"))
    assert warning["risk_level"] == "WARNING"
    assert warning["warning_event"]
    assert engine.update(4.0, pose_visible=True, probabilities=_probability("TLF"))["risk_level"] == "HIGH_DURATION_RISK"
    recovering = engine.update(5.0, pose_visible=True, probabilities=_probability("TUP"))
    assert recovering["risk_level"] == "NORMAL"
    assert recovering["continuous_duration"] == 0.0
    engine.update(6.0, pose_visible=True, probabilities=_probability("TLF"))
    rewarning = engine.update(8.0, pose_visible=True, probabilities=_probability("TLF"))
    assert rewarning["risk_level"] == "WARNING"
    assert not rewarning["warning_event"]
    lost = engine.update(9.0, pose_visible=False)
    assert lost["risk_level"] == "POSE_LOST"


def test_rolling_majority_and_minimum_direction_persistence():
    config = PostureEngineConfig(smoothing_window=3, minimum_persistence_duration=2.0)
    engine = TemporalRiskEngine(config)
    assert engine.update(0.0, pose_visible=True, probabilities=_probability("TUP"))["direction"] == "TUP"
    assert engine.update(0.5, pose_visible=True, probabilities=_probability("TUP"))["direction"] == "TUP"
    assert engine.update(1.0, pose_visible=True, probabilities=_probability("TLF"))["majority_direction"] == "TUP"
    assert engine.update(2.0, pose_visible=True, probabilities=_probability("TLF"))["direction"] == "TUP"
    switched = engine.update(3.0, pose_visible=True, probabilities=_probability("TLF"))
    assert switched["majority_direction"] == "TLF"
    assert switched["direction"] == "TUP"
    assert engine.update(4.0, pose_visible=True, probabilities=_probability("TLF"))["direction"] == "TLF"


def test_session_summary_and_event_outputs(tmp_path: Path):
    accumulator = SessionAccumulator(frame_interval_seconds=1.0)
    accumulator.add({"timestamp_seconds": 0.0, "pose_visible": True, "direction": "TUP", "risk_level": "NORMAL", "continuous_duration": 0.0, "warning_event": False})
    accumulator.add({"timestamp_seconds": 1.0, "pose_visible": True, "direction": "TLF", "risk_level": "WARNING", "continuous_duration": 5.0, "warning_event": True})
    summary = accumulator.summary(total_duration_seconds=2.0)
    assert summary["records"] == 2
    assert summary["warning_count"] == 1
    assert summary["longest_continuous_poor_posture_seconds"] == 5.0
    events = tmp_path / "events.jsonl"
    report = tmp_path / "summary.json"
    write_jsonl(events, accumulator.records)
    write_summary(report, summary)
    assert events.read_text(encoding="utf-8").count("\n") == 2
    assert '"warning_count": 1' in report.read_text(encoding="utf-8")


def test_engine_does_not_open_webcam_source_zero():
    from src.posture_ai.inference.ergonomic_pipeline import run_posture_engine

    with pytest.raises(ValueError, match="source 0 is intentionally disabled"):
        run_posture_engine("0", "missing-model", "missing-pose-model")
