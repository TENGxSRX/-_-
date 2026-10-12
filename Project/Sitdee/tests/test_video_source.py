from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.posture_ai.inference.video_source import (
    LatestFrameBuffer,
    SOURCE_CLOSED,
    SOURCE_READY,
    SOURCE_STALE,
    SourceConfigurationError,
    SourceReadResult,
    VideoSource,
    parse_source_spec,
    redact_url,
)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class FakeCapture:
    def __init__(self, reads, *, opened=True):
        self.reads = list(reads)
        self.opened = opened
        self.released = False
        self.settings = []

    def isOpened(self):
        return self.opened

    def read(self):
        value = self.reads.pop(0) if self.reads else None
        if value is None:
            return False, None
        return True, value

    def release(self):
        self.released = True

    def get(self, prop):
        return {3: 640, 4: 480, 5: 30.0}.get(prop, 0.0)

    def set(self, prop, value):
        self.settings.append((prop, value))
        return True

    def getBackendName(self):
        return "FAKE"


def test_source_parsing_supports_usb_file_and_redacted_rtsp():
    assert parse_source_spec("0").source_type == "USB"
    assert parse_source_spec("clip.mp4").source_type == "FILE"
    spec = parse_source_spec("rtsp", environ={"POSTURE_RTSP_URL": "rtsp://alice:secret@10.0.0.5/Streaming/Channels/101"})
    assert spec.source_type == "RTSP"
    assert "secret" not in spec.source_display
    assert "alice" not in spec.source_display
    assert "10.0.0.5" in spec.source_display
    assert redact_url("rtsp://user:password@example.test/path?token=abc") == "rtsp://***:***@example.test/path?token=%2A%2A%2A"


def test_rtsp_profile_requires_local_host_without_guessing(tmp_path):
    with pytest.raises(SourceConfigurationError, match="POSTURE_CAMERA_HOST"):
        parse_source_spec("HIKVISION_MAIN", environ={}, env_file=tmp_path / "no_real_env_file")


def test_read_failure_reconnects_with_cap_and_returns_latest_frame():
    clock = FakeClock()
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    captures = iter([FakeCapture([None]), FakeCapture([frame])])
    source = VideoSource(
        "rtsp://camera.local/stream",
        capture_factory=lambda target, source_type: next(captures),
        clock=clock,
        sleeper=lambda seconds: None,
        max_reconnects=2,
    )
    result = source.read()
    assert result.ok is True
    assert result.frame_index == 0
    assert result.health["reconnect_count"] == 1
    assert result.health["source_status"] == SOURCE_READY
    source.close()
    assert source.health()["source_status"] == SOURCE_CLOSED


def test_stale_state_is_not_a_pose_state():
    clock = FakeClock()
    frame = np.ones((2, 2, 3), dtype=np.uint8)
    source = VideoSource("rtsp://camera.local/stream", capture_factory=lambda *_: FakeCapture([frame]), clock=clock, max_reconnects=0, stale_after_seconds=2.0)
    assert source.read().ok
    clock.value = 3.0
    health = source.health()
    assert health["source_status"] == SOURCE_STALE
    assert health["last_frame_age_seconds"] == 3.0
    assert "risk_state" not in health


def test_file_end_of_stream_closes_without_reopening():
    capture = FakeCapture([None])
    source = VideoSource("clip.mp4", capture_factory=lambda *_: capture, max_reconnects=0)
    result = source.read()
    assert result.ok is False
    assert result.reason == "end_of_stream"
    assert result.state == SOURCE_CLOSED
    assert source.read().reason == "source_closed"
    assert capture.released is True


def test_mocked_webcam_opens_reads_applies_preferences_and_releases():
    frame = np.ones((4, 5, 3), dtype=np.uint8)
    capture = FakeCapture([frame])
    opened = []
    source = VideoSource(
        "0", capture_factory=lambda target, kind: opened.append((target, kind)) or capture,
        preferred_width=1280, preferred_height=720, preferred_fps=30,
    )
    result = source.read()
    assert opened == [(0, "USB")]
    assert result.ok and result.frame is frame
    assert capture.settings == [(3, 1280), (4, 720), (5, 30.0)]
    source.close()
    assert capture.released and source.health()["source_status"] == SOURCE_CLOSED


def test_mocked_webcam_open_failure_is_graceful_and_releases():
    capture = FakeCapture([], opened=False)
    source = VideoSource("1", capture_factory=lambda *_: capture, max_reconnects=0)
    result = source.read()
    assert not result.ok and result.reason == "source_open_failed"
    assert result.health["last_failure_reason"] == "source_open_failed"
    assert capture.released


def test_invalid_webcam_preferences_are_rejected_before_open():
    with pytest.raises(SourceConfigurationError, match="preferences must be positive"):
        VideoSource("0", preferred_fps=float("nan"))


def test_real_local_video_uses_same_source_contract():
    path = "validation_data/external/labeled/upright/DriveNet_v6_SS_1.mp4"
    source = VideoSource(path, max_reconnects=0)
    try:
        result = source.read()
        assert result.ok and result.frame is not None
        assert result.health["source_type"] == "FILE"
    finally:
        source.close()


def test_latest_frame_buffer_is_bounded_and_marks_old_frame_stale():
    clock = FakeClock()
    buffer = LatestFrameBuffer(clock=clock, stale_after_seconds=2.0)
    health = {"source_status": SOURCE_READY, "connected": True}
    first = SourceReadResult(True, np.zeros((2, 2, 3), dtype=np.uint8), 0.0, 0, None, health)
    second = SourceReadResult(True, np.ones((2, 2, 3), dtype=np.uint8), 1.0, 1, None, health)
    assert buffer.publish(first)
    assert buffer.publish(second)
    assert buffer.overwritten_frame_count == 1
    assert buffer.snapshot().frame_index == 1
    clock.value = 3.5
    assert buffer.snapshot().health["source_status"] == SOURCE_STALE


def test_latest_frame_wait_returns_newest_without_growing_queue():
    clock = FakeClock()
    buffer = LatestFrameBuffer(clock=clock, stale_after_seconds=2.0)
    health = {"source_status": SOURCE_READY, "connected": True}
    first = SourceReadResult(True, np.zeros((2, 2, 3), dtype=np.uint8), 0.0, 0, None, health)
    second = SourceReadResult(True, np.ones((2, 2, 3), dtype=np.uint8), 0.1, 1, None, health)
    assert buffer.publish(first)
    assert buffer.publish(second)
    item = buffer.wait_for_newer(0, timeout_seconds=0.0)
    assert item is not None
    assert item.frame_index == 1
    assert len(buffer) == 1
