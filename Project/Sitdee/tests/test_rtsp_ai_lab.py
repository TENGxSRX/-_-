from __future__ import annotations

import numpy as np
import pytest

from src.posture_ai.inference.rtsp_ai_lab import draw_ai_lab_overlay, merge_source_metadata, monotonic_pose_timestamp_seconds, profile_for_source, source_status_text, summarize_timing
from src.posture_ai.pose.landmarks import PoseLandmarks


def test_ai_lab_source_metadata_is_non_secret_and_separate_from_posture():
    record = {"ai_label": "TUP", "risk_state": "NORMAL"}
    result = merge_source_metadata(
        record,
        {
            "source_status": "SOURCE_READY",
            "resolution": {"width": 1920, "height": 1080},
            "reported_fps": 25.0,
            "effective_received_fps": 24.4,
            "last_frame_age_seconds": 0.01,
            "reconnect_count": 0,
        },
        source_type="RTSP",
        camera_id="CAM01",
        profile="HIKVISION_MAIN",
        frame_scale=0.75,
    )
    assert result["ai_label"] == "TUP"
    assert result["risk_state"] == "NORMAL"
    assert result["source_status"] == "SOURCE_READY"
    assert result["source_resolution"] == {"width": 1920, "height": 1080}
    assert result["inference_frame_scale"] == 0.75
    assert "password" not in str(result).lower()


def test_source_status_defaults_without_becoming_a_posture_state():
    assert source_status_text({}) == "SOURCE_UNKNOWN"


def test_hikvision_profile_does_not_override_file_or_usb_source():
    assert profile_for_source("rtsp", "HIKVISION_MAIN") == "HIKVISION_MAIN"
    assert profile_for_source("HIKVISION_SUB", "HIKVISION_MAIN") is None
    assert profile_for_source("clip.mp4", "HIKVISION_MAIN") is None
    assert profile_for_source("0", "HIKVISION_MAIN") is None


def test_timing_metrics_report_observed_latency_and_drop_counts():
    metrics = summarize_timing(
        elapsed_seconds=2.0,
        display_count=40,
        inference_count=20,
        capture_age_ms=(10.0, 20.0, 30.0),
        inference_time_ms=(40.0, 50.0),
        render_time_ms=(5.0, 7.0),
        source_fps=20.0,
        frames_captured=42,
        frames_dropped=22,
        stale_events=1,
    )
    assert metrics["display_fps"] == 20.0
    assert metrics["inference_fps"] == 10.0
    assert metrics["frames_intentionally_dropped"] == 22
    assert metrics["capture_age_ms"]["median_ms"] == 20.0
    assert np.isfinite(metrics["inference_time_ms"]["p95_ms"])


def test_clean_debug_and_landmark_modes_render_without_changing_prediction():
    points = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9} for _ in range(33)]
    pose = PoseLandmarks.from_sequence(points)
    record = {
        "pose_detected": True,
        "core_landmark_accepted": True,
        "ai_label": "TUP",
        "ai_confidence": 0.98,
        "ai_probabilities": {"TUP": 0.98, "TLB": 0.02},
        "risk_state": "NORMAL",
        "source_status": "SOURCE_READY",
        "source_type": "RTSP",
        "frame_age_ms": 18.0,
        "display_fps": 20.0,
        "inference_fps": 18.0,
        "calibration_status": "READY",
    }
    clean = draw_ai_lab_overlay(np.zeros((240, 320, 3), dtype=np.uint8), record, pose)
    debug = draw_ai_lab_overlay(np.zeros((240, 320, 3), dtype=np.uint8), record, pose, debug=True, detailed_landmarks=True)
    assert clean.shape == debug.shape == (240, 320, 3)
    assert not np.array_equal(clean, np.zeros_like(clean))
    assert record["ai_label"] == "TUP"


def test_pose_timestamp_is_strictly_increasing_for_bursty_capture():
    assert monotonic_pose_timestamp_seconds(0.0002, None) == 0.0002
    assert monotonic_pose_timestamp_seconds(0.0003, 0.0002) == pytest.approx(0.0012)
    assert monotonic_pose_timestamp_seconds(0.25, 0.0012) == 0.25
