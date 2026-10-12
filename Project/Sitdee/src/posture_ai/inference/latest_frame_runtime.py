"""Small latest-frame input facade for live RTSP diagnostics and collection."""

from __future__ import annotations

import time
from typing import Any

from src.posture_ai.inference.video_source import (
    SOURCE_CLOSED,
    SOURCE_FAILED,
    SOURCE_RTSP,
    AsyncVideoSourceReader,
    SourceReadResult,
    VideoSource,
)


class LatestFrameInput:
    """Use one newest frame for RTSP and preserve synchronous file/USB behavior."""

    def __init__(self, source: VideoSource, *, asynchronous: bool | None = None, wait_timeout_seconds: float = 0.25):
        self.source = source
        self.asynchronous = source.source_type == SOURCE_RTSP if asynchronous is None else bool(asynchronous)
        self.wait_timeout_seconds = max(0.01, float(wait_timeout_seconds))
        self.reader: AsyncVideoSourceReader | None = None
        self._last_frame_index: int | None = None
        self._started = False

    def open(self) -> bool:
        if self._started:
            return True
        if not self.source.open():
            return False
        if self.asynchronous:
            self.reader = AsyncVideoSourceReader(self.source)
            self.reader.start()
        self._started = True
        return True

    def read(self) -> SourceReadResult:
        if not self._started:
            if not self.open():
                return SourceReadResult(False, None, 0.0, None, "source_open_failed", self.health())
        if self.reader is None:
            result = self.source.read()
            if result.ok:
                self._last_frame_index = result.frame_index
            return result
        item = self.reader.wait_for_newer(self._last_frame_index, timeout_seconds=self.wait_timeout_seconds)
        if item is None:
            health = self.health()
            state = str(health.get("source_status", SOURCE_FAILED))
            reason = "source_closed" if state == SOURCE_CLOSED else "no_new_frame"
            return SourceReadResult(False, None, float(health.get("last_frame_age_seconds") or 0.0), None, reason, health)
        self._last_frame_index = item.frame_index
        return SourceReadResult(
            True,
            item.frame,
            item.timestamp_seconds,
            item.frame_index,
            None,
            item.health,
            max(0.0, (time.monotonic() - item.published_monotonic) * 1000.0),
        )

    def health(self) -> dict[str, Any]:
        if self.reader is not None:
            return self.reader.health()
        return self.source.health()

    @property
    def dropped_frames(self) -> int:
        return int(self.reader.buffer.overwritten_frame_count) if self.reader is not None else 0

    def close(self) -> None:
        if self.reader is not None:
            self.reader.stop()
            self.reader = None
        else:
            self.source.close()
        self._started = False


__all__ = ["LatestFrameInput"]
