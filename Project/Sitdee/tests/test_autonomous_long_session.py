from __future__ import annotations

import argparse
import json
import sys
import types
from urllib.request import urlopen

import numpy as np
import pytest

from scripts import run_autonomous_long_session as lab
from scripts.run_autonomous_long_session import draw_gui
from scripts.run_autonomous_long_session import profile_for_source
from src.posture_ai.detection.yolo_optional import PersonBox
from src.posture_ai.monitoring.long_session import (
    CandidateSelector,
    CandidateSelectorConfig,
    LiveSessionState,
    LocalDashboardServer,
    atomic_write_json,
    iou,
)
from src.posture_ai.monitoring.shadow import RotatingJsonlWriter


def test_candidate_selector_reduces_duplicates_without_labels_or_pixels():
    selector = CandidateSelector(CandidateSelectorConfig(representative_interval_seconds=30, event_cooldown_seconds=2))
    box = PersonBox(10, 10, 50, 90, 0.9, track_id=7)
    first = selector.inspect([box], timestamp_seconds=0, frame_width=100, frame_height=100, new_ids={7}, reacquired_ids=set())
    duplicate = selector.inspect([box], timestamp_seconds=1, frame_width=100, frame_height=100, new_ids=set(), reacquired_ids=set())
    assert first[0]["label_status"] == "UNLABELED_SHADOW_CANDIDATE"
    assert first[0]["ground_truth"] is None and first[0]["pseudo_label"] is None
    assert first[0]["raw_video_recorded"] is False
    assert first[0]["bbox_normalized"] == [0.1, 0.1, 0.5, 0.9]
    assert duplicate == []
    assert selector.summary()["duplicates_rejected"] == 1
    assert selector.summary()["candidate_pool_size"] == 1


def test_candidate_selector_marks_overlap_and_low_confidence():
    selector = CandidateSelector(CandidateSelectorConfig(occlusion_iou_threshold=0.1, low_confidence_threshold=0.5))
    boxes = [PersonBox(0, 0, 60, 100, 0.4, track_id=1), PersonBox(40, 0, 100, 100, 0.9, track_id=2)]
    records = selector.inspect(boxes, timestamp_seconds=0, frame_width=100, frame_height=100, new_ids=set(), reacquired_ids=set())
    assert "LOW_CONFIDENCE" in records[0]["reasons"]
    assert all("OCCLUSION" in record["reasons"] for record in records)
    assert iou((0, 0, 60, 100), (40, 0, 100, 100)) > 0.1


def test_event_candidates_are_cooled_down_and_anonymous():
    selector = CandidateSelector(CandidateSelectorConfig(event_cooldown_seconds=3))
    first = selector.event_candidate(5, "TRACK_LOSS", 1.0)
    second = selector.event_candidate(5, "TRACK_LOSS", 2.0)
    assert first is not None and first["anonymous_track_id"] == 5
    assert first["ground_truth"] is None
    assert second is None


def test_candidate_selector_track_memory_is_bounded():
    selector = CandidateSelector(CandidateSelectorConfig(max_track_memory=2))
    for track_id in (1, 2, 3):
        selector.inspect(
            [PersonBox(0, 0, 10, 20, 0.9, track_id=track_id)],
            timestamp_seconds=float(track_id), frame_width=100, frame_height=100,
            new_ids={track_id}, reacquired_ids=set(),
        )
    assert selector.summary()["track_memory_size"] == 2


def test_gui_overlay_renders_without_mutating_source_frame():
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    state = LiveSessionState("SESSION_TEST", "2026-01-01T00:00:00Z").snapshot()
    state.update({"people_now": 1, "duration_seconds": 2.0, "fps": 14.0, "latency_median_ms": 27.0})
    output = draw_gui(frame, [PersonBox(20, 30, 80, 110, 0.9, track_id=4)], state, cuda_name="TEST_GPU")
    assert output.shape == frame.shape
    assert np.count_nonzero(output) > 0
    assert np.count_nonzero(frame) == 0


def test_file_source_does_not_inherit_rtsp_profile():
    assert profile_for_source("datasets/video.mp4", "HIKVISION_MAIN") is None
    assert profile_for_source("rtsp", "HIKVISION_MAIN") == "HIKVISION_MAIN"


def test_local_dashboard_serves_only_live_state_and_stops():
    state = LiveSessionState("SESSION_TEST", "2026-01-01T00:00:00Z")
    state.update(people_now=2)
    server = LocalDashboardServer(state, port=0)
    url = server.start()
    try:
        with urlopen(url + "api/status", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["people_now"] == 2
        assert payload["privacy"]["raw_video_recorded"] is False
    finally:
        server.stop()
    assert server.thread is None


def test_atomic_status_write_leaves_valid_json(tmp_path):
    path = tmp_path / "live_status.json"
    atomic_write_json(path, {"status": "READY", "raw_video_recorded": False})
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "READY"
    assert not path.with_suffix(".json.tmp").exists()


def _offline_session(monkeypatch, tmp_path, *, tiny_writer: str | None):
    """Exercise the real session loop/writers with synthetic frames, never RTSP."""

    resources = {}

    class FakeSource:
        def __init__(self, *args, **kwargs):
            pass

    class FakeInput:
        def __init__(self, source, *, asynchronous):
            self.reads = 0
            self.closed = False
            resources["source"] = self

        def open(self):
            return True

        def read(self):
            self.reads += 1
            frame = np.zeros((120, 240, 3), dtype=np.uint8)
            return types.SimpleNamespace(ok=True, frame=frame, frame_index=self.reads,
                                         frame_age_ms=0.0, health={"source_status": "SOURCE_READY"})

        def health(self):
            return {"source_status": "SOURCE_READY", "reconnect_count": 0}

        def close(self):
            self.closed = True

    class FakeTracker:
        def __init__(self, *args, **kwargs):
            self.calls = 0
            resources["tracker"] = self

        def track(self, frame):
            self.calls += 1
            return [PersonBox(10, 10, 80, 100, 0.9, track_id=self.calls)]

    class FakeDashboard:
        def __init__(self, *args, **kwargs):
            self.closed = False
            resources["dashboard"] = self

        def start(self):
            return "http://127.0.0.1:0/"

        def stop(self):
            self.closed = True

    def writer_factory(directory, stem, *, max_bytes, max_files, flush_seconds):
        writer = RotatingJsonlWriter(directory, stem,
                                    max_bytes=1024 if stem == tiny_writer else max_bytes,
                                    max_files=1 if stem == tiny_writer else 8,
                                    flush_seconds=flush_seconds)
        resources[stem] = writer
        return writer

    cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(lab, "VideoSource", FakeSource)
    monkeypatch.setattr(lab, "LatestFrameInput", FakeInput)
    monkeypatch.setattr(lab, "UltralyticsPersonTracker", FakeTracker)
    monkeypatch.setattr(lab, "LocalDashboardServer", FakeDashboard)
    monkeypatch.setattr(lab, "RotatingJsonlWriter", writer_factory)
    if tiny_writer == "pose_telemetry":
        class FakeCropBackend:
            def __init__(self, *args, **kwargs):
                self.closed = False

            def detect(self, _crop):
                return []

            def close(self):
                self.closed = True

        monkeypatch.setattr(lab, "MediaPipeCropPoseBackend", FakeCropBackend)
    args = argparse.Namespace(
        session_id="TEST_CAPACITY", output_root=str(tmp_path), weights="unused.pt", device="cpu",
        confidence=0.35, iou=0.7, source="synthetic", camera_id="CAM_TEST",
        profile="HIKVISION_MAIN", rtsp_config="unused.yaml", max_reconnects=0,
        stale_after_seconds=1.5, dashboard_port=0, no_browser=True,
        max_log_mib=1, max_log_files=8, flush_seconds=0.01,
        representative_interval_seconds=30.0, transition_distance=0.12,
        low_confidence_threshold=0.45, stable_confidence_threshold=0.7,
        occlusion_iou_threshold=0.25, event_cooldown_seconds=0.0,
        duration_minutes=0.001 if tiny_writer is None else 1.0,
        telemetry_hz=100000.0, max_width=240, short_track_seconds=2.0,
        max_track_memory=4096, dashboard_refresh_seconds=0.01, no_display=True,
        enable_pose_v2=tiny_writer == "pose_telemetry", max_pose_per_second=2.0,
        pose_model="unused.task", pose_visibility_threshold=0.5,
    )
    return lab.run(args), resources, tmp_path / "TEST_CAPACITY"


@pytest.mark.parametrize("tiny_writer", ["telemetry", "candidates", "pose_telemetry"])
def test_writer_capacity_gracefully_stops_and_finalizes(monkeypatch, tmp_path, tiny_writer):
    summary, resources, session_dir = _offline_session(monkeypatch, tmp_path, tiny_writer=tiny_writer)
    assert summary["stopped_reason"] == "TELEMETRY_CAPACITY_REACHED"
    assert summary["writers"][tiny_writer]["capacity_reached"] is True
    assert resources["tracker"].calls < 20  # no silent tracking-only continuation
    assert resources["source"].closed and resources["dashboard"].closed
    assert resources["telemetry"]._handle is None and resources["candidates"]._handle is None
    if tiny_writer == "pose_telemetry":
        assert resources["pose_telemetry"]._handle is None
    assert json.loads((session_dir / "session_summary.json").read_text())["stopped_reason"] == "TELEMETRY_CAPACITY_REACHED"
    manifest = json.loads((session_dir / "session_manifest.json").read_text())
    assert manifest["stopped_reason"] == "TELEMETRY_CAPACITY_REACHED"
    assert manifest["summary"] == "session_summary.json" and "ended_utc" in manifest
    assert json.loads((session_dir / "live_status.json").read_text())["stopped_reason"] == "TELEMETRY_CAPACITY_REACHED"
    assert "TELEMETRY_CAPACITY_REACHED" in (session_dir / "session_report_th.md").read_text(encoding="utf-8")
    for path in session_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(line)


def test_normal_writer_capacity_does_not_change_completion(monkeypatch, tmp_path):
    summary, resources, session_dir = _offline_session(monkeypatch, tmp_path, tiny_writer=None)
    assert summary["stopped_reason"] == "DURATION_COMPLETED"
    assert summary["processed_frames"] > 0
    assert not summary["writers"]["telemetry"]["capacity_reached"]
    assert not summary["writers"]["candidates"]["capacity_reached"]
    assert summary["pose_pipeline"] == "POSE_V2_OPTIONAL_DISABLED"
    assert "pose_telemetry" not in summary["writers"]
    assert resources["source"].closed and resources["dashboard"].closed
    assert json.loads((session_dir / "session_manifest.json").read_text())["stopped_reason"] == "DURATION_COMPLETED"
