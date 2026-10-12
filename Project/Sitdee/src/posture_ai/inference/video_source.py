"""Unified, secret-safe source handling for USB, file and RTSP inputs.

The module is intentionally independent from the posture model.  It provides
bounded source I/O, health state, reconnect behavior and an optional
latest-frame reader so the AI loop can treat USB, prerecorded files and an
Hikvision RTSP URL through the same contract.

No function in this module starts a listener, changes routes/firewall rules, or
opens a source until the caller explicitly calls ``open``/``read``.
"""

from __future__ import annotations

import os
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import cv2
import numpy as np
import yaml


SOURCE_CONNECTING = "SOURCE_CONNECTING"
SOURCE_READY = "SOURCE_READY"
SOURCE_STALE = "SOURCE_STALE"
SOURCE_RECONNECTING = "SOURCE_RECONNECTING"
SOURCE_FAILED = "SOURCE_FAILED"
SOURCE_CLOSED = "SOURCE_CLOSED"

SOURCE_USB = "USB"
SOURCE_FILE = "FILE"
SOURCE_RTSP = "RTSP"

SUPPORTED_CAMERA_INDICES = (0, 1, 2)
DEFAULT_PROFILE_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "rtsp_profiles.yaml"
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


class SourceConfigurationError(ValueError):
    """Raised when a source cannot be resolved without guessing secrets/settings."""


def load_env_file(path: str | Path = DEFAULT_ENV_FILE) -> dict[str, str]:
    """Read a small KEY=VALUE file without printing or mutating the process environment."""

    source = Path(path)
    if not source.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def merged_source_environment(
    environ: Mapping[str, str] | None = None,
    *,
    env_file: str | Path = DEFAULT_ENV_FILE,
) -> dict[str, str]:
    """Combine ignored local settings with process variables; process wins."""

    values = load_env_file(env_file)
    selected_environment = os.environ if environ is None else environ
    values.update({str(key): str(value) for key, value in selected_environment.items()})
    return values


def redact_url(value: str | None) -> str | None:
    """Redact URL credentials and common credential query parameters."""

    if value is None:
        return None
    text = str(value)
    try:
        parsed = urlsplit(text)
    except ValueError:
        return "<redacted-source>"
    if not parsed.scheme:
        return text
    try:
        hostname = parsed.hostname or ""
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        hostname, port = "", ""
    if parsed.username is not None or parsed.password is not None:
        netloc = f"***:***@{hostname}{port}"
    else:
        netloc = f"{hostname}{port}"
    sensitive_query_keys = {"password", "pass", "pwd", "token", "secret", "user", "username"}
    query_items = []
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        query_items.append((key, "***" if key.lower() in sensitive_query_keys else item))
    return urlunsplit((parsed.scheme, netloc, parsed.path, urlencode(query_items), ""))


def _profile_data(config_path: str | Path = DEFAULT_PROFILE_CONFIG) -> dict[str, dict[str, Any]]:
    path = Path(config_path)
    if not path.exists():
        return {
            "HIKVISION_MAIN": {"camera_id": "CAM01", "stream_path": "/Streaming/Channels/101", "stream_kind": "main"},
            "HIKVISION_SUB": {"camera_id": "CAM01", "stream_path": "/Streaming/Channels/102", "stream_kind": "sub"},
        }
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = raw.get("profiles", {})
    if not isinstance(profiles, Mapping):
        raise SourceConfigurationError("RTSP profile config must contain a profiles mapping")
    return {str(key).upper(): dict(value or {}) for key, value in profiles.items() if isinstance(value, Mapping)}


def build_hikvision_rtsp_url(
    profile: str = "HIKVISION_MAIN",
    *,
    environ: Mapping[str, str] | None = None,
    config_path: str | Path = DEFAULT_PROFILE_CONFIG,
    env_file: str | Path = DEFAULT_ENV_FILE,
) -> tuple[str, str]:
    """Build a private RTSP URL from a profile plus local environment settings."""

    env = merged_source_environment(environ, env_file=env_file)
    name = str(profile).upper()
    profiles = _profile_data(config_path)
    if name not in profiles:
        raise SourceConfigurationError(f"unknown RTSP profile: {name}")
    item = profiles[name]
    host = str(env.get("POSTURE_CAMERA_HOST", "")).strip()
    if not host:
        raise SourceConfigurationError("POSTURE_CAMERA_HOST is required for a configured Hikvision profile")
    if "://" in host or "/" in host:
        raise SourceConfigurationError("POSTURE_CAMERA_HOST must contain a host/IP only, without scheme or path")
    port = str(env.get("POSTURE_CAMERA_PORT", item.get("port", "554"))).strip()
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise SourceConfigurationError("POSTURE_CAMERA_PORT must be a valid TCP port")
    path = str(env.get("POSTURE_CAMERA_STREAM_PATH", item.get("stream_path", ""))).strip()
    if not path.startswith("/"):
        path = "/" + path
    username = str(env.get("POSTURE_CAMERA_USER", ""))
    password = str(env.get("POSTURE_CAMERA_PASSWORD", ""))
    if bool(username) != bool(password):
        raise SourceConfigurationError("POSTURE_CAMERA_USER and POSTURE_CAMERA_PASSWORD must be provided together")
    auth = f"{quote(username, safe='')}:{quote(password, safe='')}@" if username and password else ""
    scheme = str(env.get("POSTURE_RTSP_SCHEME", "rtsp")).lower()
    if scheme not in {"rtsp", "rtsps"}:
        raise SourceConfigurationError("POSTURE_RTSP_SCHEME must be rtsp or rtsps")
    return f"{scheme}://{auth}{host}:{port}{path}", name


@dataclass(frozen=True)
class VideoSourceSpec:
    source_type: str
    raw_source: str
    resolved_source: str | int
    source_display: str
    camera_id: str | None = None
    profile: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source": self.source_display,
            "camera_id": self.camera_id,
            "profile": self.profile,
        }


def parse_source_spec(
    source: str | int,
    *,
    camera_id: str | None = None,
    profile: str | None = None,
    environ: Mapping[str, str] | None = None,
    config_path: str | Path = DEFAULT_PROFILE_CONFIG,
    env_file: str | Path = DEFAULT_ENV_FILE,
) -> VideoSourceSpec:
    """Resolve USB index, file path, direct RTSP URL or local RTSP profile."""

    raw = str(source).strip()
    if raw.isdigit():
        index = int(raw)
        if index not in SUPPORTED_CAMERA_INDICES:
            raise SourceConfigurationError("USB source must be camera index 0, 1, or 2")
        return VideoSourceSpec(SOURCE_USB, raw, index, f"camera_index:{index}", camera_id=camera_id)

    env = merged_source_environment(environ, env_file=env_file)
    profiles = _profile_data(config_path)
    upper = raw.upper()
    is_direct_rtsp = raw.lower().startswith(("rtsp://", "rtsps://"))
    is_rtsp_alias = raw.lower() in {"rtsp", "hikvision", "camera"}
    is_profile = upper in profiles
    if is_direct_rtsp or is_rtsp_alias or is_profile or profile is not None:
        selected_profile = str(profile or (upper if is_profile else env.get("POSTURE_HIKVISION_PROFILE", "HIKVISION_MAIN"))).upper()
        if is_direct_rtsp and profile is None:
            resolved = raw
        elif is_rtsp_alias and profile is None and env.get("POSTURE_RTSP_URL", "").strip():
            resolved = env["POSTURE_RTSP_URL"].strip()
            if not resolved.lower().startswith(("rtsp://", "rtsps://")):
                raise SourceConfigurationError("POSTURE_RTSP_URL must use rtsp:// or rtsps://")
        else:
            resolved, selected_profile = build_hikvision_rtsp_url(selected_profile, environ=env, config_path=config_path, env_file=env_file)
        resolved_camera_id = camera_id or env.get("POSTURE_CAMERA_ID") or profiles.get(selected_profile, {}).get("camera_id") or "CAM01"
        return VideoSourceSpec(SOURCE_RTSP, raw, resolved, redact_url(resolved) or "rtsp://<redacted>", camera_id=str(resolved_camera_id), profile=selected_profile)

    path = str(Path(raw).expanduser())
    return VideoSourceSpec(SOURCE_FILE, raw, path, path, camera_id=camera_id)


@dataclass
class SourceHealth:
    source_type: str
    source_display: str
    camera_id: str | None
    state: str = SOURCE_CONNECTING
    connected: bool = False
    frames_received: int = 0
    decode_failures: int = 0
    consecutive_failures: int = 0
    reported_fps: float = 0.0
    effective_received_fps: float = 0.0
    width: int = 0
    height: int = 0
    last_frame_age_seconds: float | None = None
    reconnect_count: int = 0
    late_frame_count: int = 0
    overwritten_frame_count: int = 0
    last_failure_reason: str | None = None
    backend: str | None = None
    buffer_control_requested: bool = False
    buffer_control_applied: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source": self.source_display,
            "camera_id": self.camera_id,
            "source_status": self.state,
            "connected": self.connected,
            "frames_received": self.frames_received,
            "decode_failures": self.decode_failures,
            "consecutive_failures": self.consecutive_failures,
            "reported_fps": self.reported_fps,
            "effective_received_fps": self.effective_received_fps,
            "resolution": {"width": self.width, "height": self.height},
            "last_frame_age_seconds": self.last_frame_age_seconds,
            "reconnect_count": self.reconnect_count,
            "late_frame_count": self.late_frame_count,
            "overwritten_frame_count": self.overwritten_frame_count,
            "last_failure_reason": self.last_failure_reason,
            "backend": self.backend,
            "buffer_control_requested": self.buffer_control_requested,
            "buffer_control_applied": self.buffer_control_applied,
            "network_packet_loss": "NOT_OBSERVABLE_FROM_OPENCV_SOURCE",
        }


@dataclass(frozen=True)
class SourceReadResult:
    ok: bool
    frame: np.ndarray | None
    timestamp_seconds: float
    frame_index: int | None
    reason: str | None
    health: dict[str, Any]
    frame_age_ms: float | None = None

    @property
    def state(self) -> str:
        return str(self.health.get("source_status", SOURCE_FAILED))


CaptureFactory = Callable[[str | int, str], Any]


class VideoSource:
    """Bounded source adapter with capped reconnects and redacted diagnostics."""

    def __init__(
        self,
        source: str | int,
        *,
        camera_id: str | None = None,
        profile: str | None = None,
        environ: Mapping[str, str] | None = None,
        config_path: str | Path = DEFAULT_PROFILE_CONFIG,
        reconnect: bool = True,
        max_reconnects: int = 5,
        backoff_initial_seconds: float = 0.25,
        backoff_max_seconds: float = 5.0,
        stale_after_seconds: float = 2.0,
        low_latency: bool = False,
        preferred_width: int | None = None,
        preferred_height: int | None = None,
        preferred_fps: float | None = None,
        capture_factory: CaptureFactory | None = None,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ):
        self.spec = parse_source_spec(source, camera_id=camera_id, profile=profile, environ=environ, config_path=config_path)
        self.reconnect = bool(reconnect) and self.spec.source_type != SOURCE_FILE
        self.max_reconnects = max(0, int(max_reconnects))
        self.backoff_initial_seconds = max(0.0, float(backoff_initial_seconds))
        self.backoff_max_seconds = max(self.backoff_initial_seconds, float(backoff_max_seconds))
        self.stale_after_seconds = max(0.0, float(stale_after_seconds))
        self.low_latency = bool(low_latency)
        if any(value is not None and (not math.isfinite(float(value)) or float(value) <= 0)
               for value in (preferred_width, preferred_height, preferred_fps)):
            raise SourceConfigurationError("camera width, height, and FPS preferences must be positive")
        self.preferred_width = None if preferred_width is None else int(preferred_width)
        self.preferred_height = None if preferred_height is None else int(preferred_height)
        self.preferred_fps = None if preferred_fps is None else float(preferred_fps)
        self.capture_factory = capture_factory
        self.clock = clock or time.monotonic
        self.sleeper = sleeper or time.sleep
        self._capture: Any | None = None
        self._capture_lock = threading.RLock()
        self._closed = False
        self._session_start = self.clock()
        self._last_frame_clock: float | None = None
        self._frame_index = -1
        self._reconnect_attempts = 0
        self._backoff_seconds = self.backoff_initial_seconds
        self._receive_times: deque[float] = deque(maxlen=30)
        self._health = SourceHealth(self.spec.source_type, self.spec.source_display, self.spec.camera_id)

    @property
    def source_type(self) -> str:
        return self.spec.source_type

    @property
    def camera_id(self) -> str | None:
        return self.spec.camera_id

    def _factory_open(self) -> Any:
        if self.capture_factory is not None:
            return self.capture_factory(self.spec.resolved_source, self.spec.source_type)
        if self.spec.source_type == SOURCE_USB:
            index = int(self.spec.resolved_source)
            if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
                capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                if capture.isOpened():
                    return capture
                capture.release()
            return cv2.VideoCapture(index)
        return cv2.VideoCapture(str(self.spec.resolved_source))

    def _release_capture(self) -> None:
        with self._capture_lock:
            if self._capture is not None:
                try:
                    self._capture.release()
                finally:
                    self._capture = None

    def open(self) -> bool:
        if self._closed:
            self._health.state = SOURCE_CLOSED
            self._health.connected = False
            return False
        self._health.state = SOURCE_CONNECTING
        self._release_capture()
        try:
            capture = self._factory_open()
        except Exception:
            self._health.state = SOURCE_FAILED
            self._health.connected = False
            self._health.last_failure_reason = "capture_open_exception"
            return False
        if capture is None or not bool(capture.isOpened()):
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            self._health.state = SOURCE_FAILED
            self._health.connected = False
            self._health.last_failure_reason = "source_open_failed"
            return False
        with self._capture_lock:
            self._capture = capture
        if self.spec.source_type == SOURCE_USB:
            for prop, value in (
                (cv2.CAP_PROP_FRAME_WIDTH, self.preferred_width),
                (cv2.CAP_PROP_FRAME_HEIGHT, self.preferred_height),
                (cv2.CAP_PROP_FPS, self.preferred_fps),
            ):
                if value is not None:
                    try:
                        capture.set(prop, value)
                    except Exception:
                        pass  # Preferences are best-effort; health reports actual values.
        self._health.state = SOURCE_READY
        self._health.connected = True
        self._health.last_failure_reason = None
        self._health.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._health.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._health.reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        try:
            self._health.backend = str(capture.getBackendName())
        except Exception:
            self._health.backend = None
        self._health.buffer_control_requested = bool(self.low_latency and self.spec.source_type == SOURCE_RTSP)
        self._health.buffer_control_applied = None
        if self._health.buffer_control_requested and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            try:
                self._health.buffer_control_applied = bool(capture.set(cv2.CAP_PROP_BUFFERSIZE, 1))
            except Exception:
                self._health.buffer_control_applied = False
        return True

    def _update_age(self, now: float | None = None) -> None:
        current = self.clock() if now is None else float(now)
        if self._last_frame_clock is None:
            age = max(0.0, current - self._session_start)
        else:
            age = max(0.0, current - self._last_frame_clock)
        self._health.last_frame_age_seconds = age
        if self._health.connected and self.stale_after_seconds > 0 and age > self.stale_after_seconds:
            self._health.state = SOURCE_STALE

    def _effective_fps(self) -> float:
        if len(self._receive_times) < 2:
            return 0.0
        elapsed = self._receive_times[-1] - self._receive_times[0]
        return float((len(self._receive_times) - 1) / elapsed) if elapsed > 0 else 0.0

    def _attempt_reconnect(self) -> bool:
        if not self.reconnect or self._reconnect_attempts >= self.max_reconnects:
            return False
        self._reconnect_attempts += 1
        self._health.reconnect_count = self._reconnect_attempts
        self._health.state = SOURCE_RECONNECTING
        self._health.connected = False
        if self._backoff_seconds > 0:
            self.sleeper(self._backoff_seconds)
        self._backoff_seconds = min(self.backoff_max_seconds, max(self.backoff_initial_seconds, self._backoff_seconds * 2.0 or 0.25))
        return self.open()

    def read(self) -> SourceReadResult:
        """Read one frame; failures are bounded and never become posture states."""

        if self._closed:
            return SourceReadResult(False, None, max(0.0, self.clock() - self._session_start), None, "source_closed", self.health())
        if self._capture is None and not self.open():
            attempts = 0
            while self.reconnect and attempts < self.max_reconnects and self._capture is None:
                if not self._attempt_reconnect():
                    break
                attempts += 1
            if self._capture is None:
                return SourceReadResult(False, None, max(0.0, self.clock() - self._session_start), None, "source_open_failed", self.health())

        attempts = 0
        while self._capture is not None:
            read_started = self.clock()
            with self._capture_lock:
                capture = self._capture
                if capture is None:
                    break
                try:
                    ok, frame = capture.read()
                except Exception:
                    ok, frame = False, None
            now = self.clock()
            if bool(ok) and frame is not None and getattr(frame, "size", 0) > 0:
                self._frame_index += 1
                self._health.frames_received += 1
                self._health.consecutive_failures = 0
                self._health.state = SOURCE_READY
                self._health.connected = True
                self._health.last_failure_reason = None
                self._last_frame_clock = now
                self._health.last_frame_age_seconds = 0.0
                self._receive_times.append(now)
                self._health.effective_received_fps = self._effective_fps()
                if self.stale_after_seconds > 0 and now - read_started > self.stale_after_seconds:
                    self._health.late_frame_count += 1
                timestamp = max(0.0, now - self._session_start)
                return SourceReadResult(True, frame, timestamp, self._frame_index, None, self.health())

            self._health.decode_failures += 1
            self._health.consecutive_failures += 1
            self._health.last_failure_reason = "read_failed_or_empty_frame"
            self._update_age(now)
            if self.spec.source_type == SOURCE_FILE:
                self._health.state = SOURCE_CLOSED
                self._health.connected = False
                self._closed = True
                self._release_capture()
                return SourceReadResult(False, None, max(0.0, now - self._session_start), None, "end_of_stream", self.health())
            if attempts >= self.max_reconnects or not self._attempt_reconnect():
                self._health.state = SOURCE_STALE if self._health.last_frame_age_seconds and self._health.last_frame_age_seconds > self.stale_after_seconds else SOURCE_FAILED
                self._health.connected = False
                return SourceReadResult(False, None, max(0.0, now - self._session_start), None, "source_unavailable", self.health())
            attempts += 1
        return SourceReadResult(False, None, max(0.0, self.clock() - self._session_start), None, "source_unavailable", self.health())

    def health(self) -> dict[str, Any]:
        self._update_age()
        return self._health.to_dict()

    def close(self) -> None:
        self._closed = True
        self._release_capture()
        self._health.state = SOURCE_CLOSED
        self._health.connected = False


@dataclass(frozen=True)
class BufferedFrame:
    frame: np.ndarray
    timestamp_seconds: float
    frame_index: int
    health: dict[str, Any]
    published_monotonic: float


class LatestFrameBuffer:
    """Thread-safe one-frame buffer; newer frames replace older frames."""

    def __init__(self, *, clock: Callable[[], float] | None = None, stale_after_seconds: float = 2.0):
        self.clock = clock or time.monotonic
        self.stale_after_seconds = max(0.0, float(stale_after_seconds))
        self._lock = threading.Lock()
        self._updated = threading.Event()
        self._latest: BufferedFrame | None = None
        self.overwritten_frame_count = 0

    def publish(self, result: SourceReadResult) -> bool:
        if not result.ok or result.frame is None or result.frame.size == 0 or result.frame_index is None:
            return False
        item = BufferedFrame(result.frame.copy(), result.timestamp_seconds, int(result.frame_index), dict(result.health), self.clock())
        with self._lock:
            if self._latest is not None:
                self.overwritten_frame_count += 1
            self._latest = item
            self._updated.set()
        return True

    def snapshot(self) -> BufferedFrame | None:
        with self._lock:
            item = self._latest
            if item is None:
                return None
            health = dict(item.health)
            age = max(0.0, self.clock() - item.published_monotonic)
            # The source health age is sufficient for synchronous callers; this
            # bounded buffer also marks a frame stale if the consumer waits.
            if self.stale_after_seconds > 0 and age > self.stale_after_seconds:
                health["source_status"] = SOURCE_STALE
                health["connected"] = False
            health["overwritten_frame_count"] = self.overwritten_frame_count
            return BufferedFrame(item.frame.copy(), item.timestamp_seconds, item.frame_index, health, item.published_monotonic)

    def __len__(self) -> int:
        with self._lock:
            return 0 if self._latest is None else 1

    def wait_for_newer(self, frame_index: int | None = None, *, timeout_seconds: float = 0.25) -> BufferedFrame | None:
        """Wait for a frame newer than ``frame_index`` without making a queue."""

        deadline = self.clock() + max(0.0, float(timeout_seconds))
        while True:
            item = self.snapshot()
            if item is not None and (frame_index is None or item.frame_index > int(frame_index)):
                return item
            remaining = deadline - self.clock()
            if remaining <= 0:
                return None
            self._updated.wait(timeout=remaining)
            self._updated.clear()


class AsyncVideoSourceReader:
    """Optional latest-frame reader thread with bounded memory and clean stop."""

    def __init__(self, source: VideoSource, *, poll_sleep_seconds: float = 0.005):
        self.source = source
        self.poll_sleep_seconds = max(0.0, float(poll_sleep_seconds))
        self.buffer = LatestFrameBuffer(stale_after_seconds=source.stale_after_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "AsyncVideoSourceReader":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="posture-video-source", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            result = self.source.read()
            self.buffer.publish(result)
            if result.state in {SOURCE_CLOSED, SOURCE_FAILED} and self.source.source_type == SOURCE_FILE:
                break
            if not result.ok:
                self._stop.wait(self.poll_sleep_seconds)

    def snapshot(self) -> BufferedFrame | None:
        return self.buffer.snapshot()

    def wait_for_newer(self, frame_index: int | None = None, *, timeout_seconds: float = 0.25) -> BufferedFrame | None:
        return self.buffer.wait_for_newer(frame_index, timeout_seconds=timeout_seconds)

    def health(self) -> dict[str, Any]:
        health = self.source.health()
        health["overwritten_frame_count"] = self.buffer.overwritten_frame_count
        health["reader_thread_alive"] = bool(self._thread is not None and self._thread.is_alive())
        return health

    def stop(self) -> None:
        self._stop.set()
        self.source.close()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None


__all__ = [
    "AsyncVideoSourceReader",
    "BufferedFrame",
    "LatestFrameBuffer",
    "SOURCE_CLOSED",
    "SOURCE_CONNECTING",
    "SOURCE_FAILED",
    "SOURCE_FILE",
    "SOURCE_READY",
    "SOURCE_RECONNECTING",
    "SOURCE_RTSP",
    "SOURCE_STALE",
    "SOURCE_USB",
    "SourceConfigurationError",
    "SourceHealth",
    "SourceReadResult",
    "VideoSource",
    "VideoSourceSpec",
    "build_hikvision_rtsp_url",
    "load_env_file",
    "merged_source_environment",
    "parse_source_spec",
    "redact_url",
]
