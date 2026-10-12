from __future__ import annotations

import numpy as np

from src.posture_ai.ergonomics.activity_gate import ActivityGate, ActivityGateConfig, classify_activity_geometry
from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig
from src.posture_ai.multiperson import MultiPersonRuntime, MultiPersonRuntimeConfig, PoseDetection
from src.posture_ai.pose.landmarks import PoseLandmarks


def _pose(kind: str, *, lower_visibility: float = 0.95) -> PoseLandmarks:
    values = np.zeros((33, 3), dtype=float)
    values[0] = [0.5, 0.2, 0.0]
    values[11] = [0.42, 0.4, 0.0]
    values[12] = [0.58, 0.4, 0.0]
    values[23] = [0.45, 0.55, 0.0]
    values[24] = [0.55, 0.55, 0.0]
    if kind == "sitting":
        values[25] = [0.25, 0.58, 0.0]
        values[26] = [0.75, 0.58, 0.0]
        values[27] = [0.18, 0.42, 0.0]
        values[28] = [0.82, 0.42, 0.0]
    else:
        values[25] = [0.45, 0.72, 0.0]
        values[26] = [0.55, 0.72, 0.0]
        values[27] = [0.45, 0.92, 0.0]
        values[28] = [0.55, 0.92, 0.0]
    visibility = [0.95] * 33
    for index in (25, 26, 27, 28):
        visibility[index] = lower_visibility
    return PoseLandmarks.from_xyz_array(values, visibility=visibility)


def _detection(pose: PoseLandmarks, x: float = 0.5) -> PoseDetection:
    return PoseDetection(pose, pose, pose_quality=0.95, core_valid=True, center=(x, 0.5))


class _TlbV4:
    labels = ["TLB", "TUP"]

    def predict_world_landmarks(self, world_landmarks, *, image_visibility=None, visibility_threshold=0.5):
        return {"label": "TLB", "confidence": 0.9, "probabilities": {"TLB": 0.9, "TUP": 0.1}}


def test_activity_gate_requires_lower_body_and_debounces_transitions():
    sitting = _pose("sitting")
    standing = _pose("standing")
    assert classify_activity_geometry(sitting).raw_state == "SITTING"
    assert classify_activity_geometry(standing).raw_state == "STANDING"
    assert classify_activity_geometry(_pose("sitting", lower_visibility=0.1)).raw_state == "UNKNOWN"
    gate = ActivityGate(ActivityGateConfig(transition_confirmations=2))
    assert gate.update(0.0, sitting)["state"] == "UNKNOWN"
    assert gate.update(0.1, sitting)["state"] == "SITTING"
    assert gate.update(0.2, standing)["state"] == "UNKNOWN"


def test_standing_pauses_sitting_posture_and_unknown_pauses_risk():
    runtime = MultiPersonRuntime(
        _TlbV4(),
        runtime_config=MultiPersonRuntimeConfig(
            max_people=2,
            activity_gate=ActivityGateConfig(transition_confirmations=1),
        ),
        posture_config=PostureEngineConfig(
            minimum_warning_duration=5.0,
            high_risk_duration=10.0,
            activity_gate=ActivityGateConfig(transition_confirmations=1),
        ),
    )
    result = runtime.process_frame([_detection(_pose("sitting"), 0.3), _detection(_pose("standing"), 0.7)], 0.0)
    sitting, standing = result.subjects
    assert sitting.activity_state == "SITTING"
    assert sitting.v4_prediction == "TLB"
    assert standing.activity_state == "STANDING"
    assert standing.v4_prediction is None
    assert standing.temporal_state["risk_level"] == "MONITOR_PAUSED"
    result = runtime.process_frame([_detection(_pose("sitting"), 0.3), _detection(_pose("sitting", lower_visibility=0.1), 0.7)], 1.0)
    assert result.subjects[1].activity_state == "UNKNOWN"
    assert result.subjects[1].temporal_state["risk_level"] == "MONITOR_PAUSED"
    assert result.subjects[1].temporal_state["duration_seconds"] == 0.0


def test_lifecycle_events_and_bounded_anonymous_id_reuse():
    from src.posture_ai.multiperson import AnonymousPoseTracker

    tracker = AnonymousPoseTracker(max_people=1, max_misses=0, reacquisition_window_seconds=1.0)
    detection = _detection(_pose("standing"))
    tracker.update([detection], 0.0)
    assert tracker.last_events[0]["event"] == "PERSON_APPEARED"
    tracker.update([], 0.1)
    assert tracker.last_events[0]["event"] == "PERSON_EXITED"
    tracker.update([detection], 0.2)
    assert tracker.last_events[0]["event"] == "PERSON_APPEARED"
    assert tracker.last_assignments[0] == "PERSON_01"
    tracker.update([detection], 0.3)
    assert tracker.last_events[0]["event"] == "PERSON_ACTIVE"
    assert tracker.track_state("PERSON_01") == "ACTIVE"
