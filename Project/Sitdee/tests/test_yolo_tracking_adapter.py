from __future__ import annotations

import numpy as np

from src.posture_ai.detection.yolo_optional import PersonBox, UltralyticsPersonTracker
from scripts.run_yolo_rtsp_tracking import _subject_snapshots


def test_person_box_preserves_anonymous_tracker_id_when_clipped():
    box = PersonBox(-10, 5, 120, 80, 0.8, track_id=17)
    clipped = box.clipped(100, 60)
    assert clipped.track_id == 17
    assert clipped.normalized(100, 60) == (0.0, 5 / 60, 1.0, 1.0)


def test_tracking_adapter_can_be_imported_without_ultralytics_in_main_environment():
    # Constructor remains lazy so the working MediaPipe environment never needs
    # an Ultralytics dependency merely to import project modules.
    assert UltralyticsPersonTracker.__name__ == "UltralyticsPersonTracker"
    assert np.zeros((1,), dtype=np.uint8).size == 1


def test_controlled_subject_snapshot_keeps_human_label_separate_from_track_id():
    boxes = [PersonBox(10, 20, 30, 60, 0.8, track_id=41)]
    snapshots = _subject_snapshots(
        boxes,
        {"A": 41, "B": 99},
        {"A": 12, "B": 99},
        frame_width=100,
        frame_height=80,
        fps=14.0,
        latency_ms=25.0,
        timestamp_seconds=3.5,
    )
    assert snapshots[0]["operator_subject"] == "A"
    assert snapshots[0]["track_id"] == 41
    assert snapshots[0]["previous_track_id"] == 12
    assert snapshots[0]["detection_present"] is True
    assert snapshots[0]["bbox"] == [10.0, 20.0, 30.0, 60.0]
    assert snapshots[1]["operator_subject"] == "B"
    assert snapshots[1]["track_id"] == 99
    assert snapshots[1]["detection_present"] is False
