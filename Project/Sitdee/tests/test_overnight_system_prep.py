from __future__ import annotations

from src.posture_ai.inference.webcam_v1 import LiveSessionAccumulator, resolve_performance_preset, should_process_frame


def test_low_spec_frame_skipping_preserves_wall_clock_duration():
    preset = resolve_performance_preset("LOW_SPEC")
    processed = [index for index in range(6) if should_process_frame(index, preset["process_every"], index > 0)]
    assert processed == [0, 3]

    accumulator = LiveSessionAccumulator(frame_interval_seconds=1.0 / 30.0)
    accumulator.add({
        "timestamp": 0.0, "pose_detected": True, "core_landmark_accepted": True,
        "ai_posture": "UPRIGHT", "ai_label": "TUP", "risk_state": "NORMAL",
        "warning_event": False, "continuous_bad_duration": 0.0, "fps": 10.0,
    })
    accumulator.add({
        "timestamp": 0.3, "pose_detected": True, "core_landmark_accepted": True,
        "ai_posture": "UPRIGHT", "ai_label": "TUP", "risk_state": "NORMAL",
        "warning_event": False, "continuous_bad_duration": 0.0, "fps": 10.0,
    })
    summary = accumulator.summary(total_duration_seconds=1.0)
    assert summary["total_session_duration_seconds"] == 1.0
    assert summary["valid_pose_time_seconds"] == 1.0
