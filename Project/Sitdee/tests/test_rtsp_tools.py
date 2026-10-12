from __future__ import annotations

from pathlib import Path

from scripts.estimate_network_bandwidth import estimate_bandwidth
from src.posture_ai.inference.stream_runtime import resize_for_display, resize_for_inference, resolve_rtsp_preset


def test_bandwidth_estimator_is_explicit_and_does_not_assume_camera_bitrate():
    result = estimate_bandwidth(4.0, 2, hours=1.0, overhead_percent=10.0)
    assert result["estimate_label"] == "ESTIMATE_ONLY"
    assert result["aggregate_raw_mbps"] == 8.0
    assert result["aggregate_planned_mbps"] == 8.8
    assert result["storage_planned_gib_for_requested_hours"] > 0


def test_rtsp_presets_and_dual_resolution_copy():
    assert resolve_rtsp_preset("RTSP_DESKTOP")["process_every"] == 1
    balanced = resolve_rtsp_preset("RTSP_BALANCED")
    assert balanced["process_every"] == 2
    import numpy as np

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    reduced, scale = resize_for_inference(frame, max_width=1280, max_height=720)
    assert reduced.shape[:2] == (720, 1280)
    assert scale == 2 / 3
    assert frame.shape[:2] == (1080, 1920)


def test_resize_for_display_is_a_separate_aspect_preserving_copy():
    import numpy as np

    frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
    reduced, scale = resize_for_display(frame)
    assert reduced.shape[:2] == (720, 1280)
    assert scale == 0.5
    assert frame.shape[:2] == (1440, 2560)
