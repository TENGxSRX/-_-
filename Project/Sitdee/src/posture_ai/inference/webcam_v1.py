"""Hardware-ready live webcam prototype for the frozen RGB-native V4 pilot.

This module deliberately keeps camera I/O separate from V4 training and from
the offline inference runner.  It accepts only the two labels actually
supported by the frozen V4 artifact (TUP and TLB), while geometry indicators
remain explicitly separate, camera-relative rules.
"""

from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.posture_ai.ergonomics.geometry import calculate_geometry
from src.posture_ai.ergonomics.calibration import NeutralBaselineCalibrator, relative_geometry
from src.posture_ai.ergonomics.indicators import (
    evaluate_directional_geometry,
    evaluate_indicators,
)
from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig, TemporalRiskEngine
from src.posture_ai.inference.pipeline import create_pose_landmarker, extract_pose_result
from src.posture_ai.inference.rgb_native_v4 import RGBNativeV4Runner
from src.posture_ai.inference.video_source import SourceConfigurationError, SOURCE_USB, parse_source_spec
from src.posture_ai.pose.landmarks import PoseLandmarks
from src.posture_ai.pose.robustness import evaluate_visibility_policy, pose_quality_score
from src.posture_ai.pose.validation import PoseValidationError


SUPPORTED_CAMERA_INDICES = (0, 1, 2)
V4_LABELS = frozenset({"TUP", "TLB"})
DISPLAY_LABELS = {"TUP": "UPRIGHT", "TLB": "LEAN BACKWARD"}
PERFORMANCE_PRESETS = {
    "DESKTOP": {
        "process_every": 1,
        "capture_width": None,
        "capture_height": None,
        "description": "Process every captured frame.",
    },
    "BALANCED": {
        "process_every": 2,
        "capture_width": 960,
        "capture_height": 540,
        "description": "Process every second captured frame.",
    },
    "LOW_SPEC": {
        "process_every": 3,
        "capture_width": 640,
        "capture_height": 480,
        "description": "Process every third captured frame at a smaller capture size.",
    },
}
CORE_LANDMARKS = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
CORE_DISPLAY_NAMES = {
    "nose": "Head",
    "left_shoulder": "L Shoulder",
    "right_shoulder": "R Shoulder",
    "left_hip": "L Hip",
    "right_hip": "R Hip",
}
CORE_THAI_NAMES = {
    "nose": "ศีรษะ",
    "left_shoulder": "ไหล่ซ้าย",
    "right_shoulder": "ไหล่ขวา",
    "left_hip": "สะโพกซ้าย",
    "right_hip": "สะโพกขวา",
}
THAI_VALIDATION_INSTRUCTIONS = (
    "นั่งตัวตรง",
    "เอนตัวไปด้านหลัง",
    "โน้มตัวไปด้านหน้า",
    "เอียงตัวไปทางซ้าย",
    "เอียงตัวไปทางขวา",
    "กลับมานั่งตัวตรง",
    "ออกจากภาพ",
    "กลับเข้ามาในภาพ",
    "ค้างท่าที่ไม่เหมาะสมเพื่อทดสอบการแจ้งเตือน",
)
SKELETON_CONNECTIONS = (
    (0, 11), (0, 12), (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (27, 29), (27, 31), (28, 30), (28, 32),
)


def resolve_performance_preset(preset: str = "DESKTOP", process_every: int | None = None) -> dict[str, Any]:
    """Resolve an explicit runtime preset without changing model semantics."""

    name = str(preset).upper()
    if name not in PERFORMANCE_PRESETS:
        raise ValueError(f"Unknown performance preset: {preset!r}")
    interval = int(process_every) if process_every is not None else int(PERFORMANCE_PRESETS[name]["process_every"])
    if interval < 1:
        raise ValueError("process_every must be at least 1")
    result = dict(PERFORMANCE_PRESETS[name])
    result.update({"name": name, "process_every": interval, "override": process_every is not None})
    return result


def should_process_frame(frame_index: int, process_every: int, has_previous_result: bool) -> bool:
    """Return whether MediaPipe/V4 should run for this captured frame.

    The first frame is always processed. Later skipped frames may reuse the
    latest display result, but callers must keep their original timestamps and
    must not count skipped frames as newly inferred evidence.
    """

    interval = int(process_every)
    if interval < 1:
        raise ValueError("process_every must be at least 1")
    return not has_previous_result or int(frame_index) % interval == 0


@dataclass(frozen=True)
class ValidationStep:
    number: int
    instruction: str
    duration_seconds: float


@dataclass
class StableCoreGate:
    """Require continuously accepted core landmarks before a user can start."""

    required_seconds: float = 1.5
    max_gap_seconds: float = 0.75
    _good_since: float | None = None
    _last_timestamp: float | None = None

    def __post_init__(self) -> None:
        if self.required_seconds <= 0 or self.max_gap_seconds <= 0:
            raise ValueError("stable core gate durations must be positive")

    def reset(self) -> None:
        self._good_since = None
        self._last_timestamp = None

    def update(self, *, accepted: bool, timestamp_seconds: float) -> dict[str, Any]:
        timestamp = float(timestamp_seconds)
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        if not accepted or (self._last_timestamp is not None and timestamp - self._last_timestamp > self.max_gap_seconds):
            self._good_since = None
        if accepted and self._good_since is None:
            self._good_since = timestamp
        self._last_timestamp = timestamp
        stable_seconds = 0.0 if self._good_since is None else max(0.0, timestamp - self._good_since)
        return {
            "ready": stable_seconds >= self.required_seconds,
            "stable_seconds": stable_seconds,
            "required_seconds": self.required_seconds,
            "status": "INPUT GOOD" if accepted else "INPUT LOST",
        }


class ValidationController:
    """User-started, timed, non-labeling validation instructions."""

    INSTRUCTIONS = (
        "SIT UPRIGHT",
        "LEAN BACKWARD",
        "LEAN FORWARD",
        "LEAN LEFT",
        "LEAN RIGHT",
        "RETURN UPRIGHT",
        "MOVE OUT OF FRAME",
        "RETURN TO FRAME",
        "HOLD A POOR POSTURE LONG ENOUGH TO TEST WARNING",
    )

    def __init__(self, *, step_seconds: float = 5.0, poor_hold_seconds: float = 20.0, preparation_seconds: float = 3.0, completion_seconds: float = 0.8, final_summary_seconds: float = 3.0, manual_step_start: bool = True):
        if step_seconds <= 0 or poor_hold_seconds <= 0 or preparation_seconds < 0 or completion_seconds < 0 or final_summary_seconds < 0:
            raise ValueError("validation durations must be positive")
        self.steps = tuple(
            ValidationStep(index + 1, THAI_VALIDATION_INSTRUCTIONS[index], poor_hold_seconds if index == 8 else step_seconds)
            for index, instruction in enumerate(self.INSTRUCTIONS)
        )
        self.preparation_seconds = float(preparation_seconds)
        self.completion_seconds = float(completion_seconds)
        self.final_summary_seconds = float(final_summary_seconds)
        self.manual_step_start = bool(manual_step_start)
        self.started = False
        self._phase = "SETUP"
        self._step_index = 0
        self._phase_started_at = 0.0
        self._complete_until = 0.0
        self._done_until = 0.0
        self._completed_step: ValidationStep | None = None
        self._active_input_elapsed = 0.0
        self._last_input_timestamp: float | None = None
        self._last_input_good = False

    @property
    def total_duration_seconds(self) -> float:
        return float(self.preparation_seconds + sum(step.duration_seconds for step in self.steps) + self.completion_seconds * len(self.steps) + self.final_summary_seconds)

    @property
    def finished(self) -> bool:
        return self._phase == "DONE"

    def start(self, timestamp_seconds: float, *, wait_for_confirmation: bool | None = None) -> None:
        self.started = True
        wait = self.manual_step_start if wait_for_confirmation is None else bool(wait_for_confirmation)
        self._phase = "WAITING" if wait else "PREPARE"
        self._step_index = 0
        self._phase_started_at = float(timestamp_seconds)
        self._complete_until = 0.0
        self._done_until = 0.0
        self._completed_step = None
        self._active_input_elapsed = 0.0
        self._last_input_timestamp = float(timestamp_seconds)
        self._last_input_good = False

    def restart_current(self, timestamp_seconds: float) -> None:
        if not self.started:
            self.start(timestamp_seconds)
            return
        self._phase = "PREPARE"
        self._phase_started_at = float(timestamp_seconds)
        self._complete_until = 0.0
        self._done_until = 0.0
        self._completed_step = None
        self._active_input_elapsed = 0.0
        self._last_input_timestamp = float(timestamp_seconds)
        self._last_input_good = False

    def advance(self, timestamp_seconds: float) -> None:
        """Start, skip preparation, or move to the next validation step."""
        timestamp = float(timestamp_seconds)
        if not self.started:
            self.start(timestamp)
            return
        if self._phase == "WAITING":
            self._phase = "PREPARE"
            self._phase_started_at = timestamp
            self._active_input_elapsed = 0.0
            self._last_input_timestamp = timestamp
            self._last_input_good = False
            return
        if self._phase == "PREPARE":
            self._phase = "ACTIVE"
            self._phase_started_at = timestamp
            self._active_input_elapsed = 0.0
            self._last_input_timestamp = timestamp
            self._last_input_good = False
            return
        if self._phase == "ACTIVE":
            self._completed_step = self.steps[self._step_index]
            self._step_index += 1
            self._phase = "COMPLETE"
            self._phase_started_at = timestamp
            self._complete_until = timestamp + self.completion_seconds
            return
        if self._phase == "COMPLETE":
            self._finish_or_start_next(timestamp)

    def _finish_or_start_next(self, timestamp_seconds: float) -> None:
        if self._step_index >= len(self.steps):
            self._phase = "DONE"
            self._done_until = float(timestamp_seconds) + self.final_summary_seconds
            return
        self._phase = "WAITING"
        self._phase_started_at = float(timestamp_seconds)
        self._active_input_elapsed = 0.0
        self._last_input_timestamp = float(timestamp_seconds)
        self._last_input_good = False

    def observe_input(self, timestamp_seconds: float, *, input_good: bool) -> dict[str, Any]:
        """Advance the timer only while the current step's input condition is met.

        Most validation steps require accepted core landmarks.  Step 7 is the
        deliberate out-of-frame check, so its expected condition is the exact
        opposite: the timer advances only while the pose is absent.
        """

        timestamp = float(timestamp_seconds)
        if self._last_input_timestamp is not None and timestamp < self._last_input_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        delta = 0.0 if self._last_input_timestamp is None else timestamp - self._last_input_timestamp
        out_of_frame_step = (
            self._phase == "ACTIVE"
            and self._step_index < len(self.steps)
            and self.steps[self._step_index].number == 7
        )
        expected_input_satisfied = not bool(input_good) if out_of_frame_step else bool(input_good)
        if self._phase == "ACTIVE" and self._last_input_good:
            self._active_input_elapsed += delta
        self._last_input_timestamp = timestamp
        self._last_input_good = expected_input_satisfied
        if out_of_frame_step:
            input_state = "OUT OF FRAME" if expected_input_satisfied else "PERSON STILL VISIBLE"
        else:
            input_state = "INPUT GOOD" if input_good else "INPUT LOST"
        return {
            "input_good": bool(input_good),
            "expected_input_satisfied": expected_input_satisfied,
            "input_state": input_state,
            "timer_paused": self._phase == "ACTIVE" and not expected_input_satisfied,
            "active_input_elapsed": self._active_input_elapsed,
        }

    def current(self, elapsed_seconds: float) -> dict[str, Any] | None:
        timestamp = max(0.0, float(elapsed_seconds))
        if not self.started:
            return {
                "phase": "SETUP",
                "title": "CAMERA SETUP",
                "instruction": "จัดตำแหน่งให้เห็นศีรษะ ไหล่ทั้งสอง และสะโพกทั้งสอง",
                "ready_to_start": False,
            }
        if self._phase == "PREPARE":
            elapsed = max(0.0, timestamp - self._phase_started_at)
            remaining = max(0.0, self.preparation_seconds - elapsed)
            if elapsed >= self.preparation_seconds:
                self._phase = "ACTIVE"
                self._phase_started_at = self._phase_started_at + self.preparation_seconds
                self._active_input_elapsed = 0.0
                self._last_input_timestamp = timestamp
                self._last_input_good = False
                return self.current(timestamp)
            return {
                "phase": "PREPARE",
                "step": self._step_index + 1,
                "total_steps": len(self.steps),
                "instruction": "GET READY",
                "thai_instruction": "เตรียมตัว",
                "countdown": max(1, int(np.ceil(remaining))),
                "remaining_seconds": remaining,
            }
        if self._phase == "WAITING":
            step = self.steps[self._step_index]
            return {
                "phase": "WAITING",
                "step": step.number,
                "total_steps": len(self.steps),
                "instruction": step.instruction,
                "thai_instruction": step.instruction,
                "remaining_seconds": 0.0,
                "awaiting_space": True,
                "requires_reacquisition": step.number == 8,
                "input_state": "INPUT GOOD" if self._last_input_good else "INPUT LOST",
            }
        if self._phase == "ACTIVE":
            step = self.steps[self._step_index]
            elapsed = max(0.0, self._active_input_elapsed)
            remaining = max(0.0, step.duration_seconds - elapsed)
            if elapsed >= step.duration_seconds:
                self._completed_step = step
                self._step_index += 1
                self._phase = "COMPLETE"
                self._complete_until = timestamp + self.completion_seconds
                self._phase_started_at = timestamp
                return self.current(timestamp)
            return {
                "phase": "ACTIVE",
                "step": step.number,
                "total_steps": len(self.steps),
                "instruction": step.instruction,
                "thai_instruction": step.instruction,
                "remaining_seconds": remaining,
                "duration_seconds": step.duration_seconds,
                "input_state": "INPUT GOOD" if self._last_input_good else "INPUT LOST",
                "timer_paused": not self._last_input_good,
            }
        if self._phase == "COMPLETE":
            if timestamp >= self._complete_until:
                self._finish_or_start_next(timestamp)
                return self.current(timestamp)
            return {
                "phase": "COMPLETE",
                "step": self._completed_step.number if self._completed_step else self._step_index,
                "total_steps": len(self.steps),
                "instruction": "STEP COMPLETE",
                "thai_instruction": "จบขั้นตอนนี้แล้ว",
                "remaining_seconds": max(0.0, self._complete_until - timestamp),
            }
        if self._phase == "DONE":
            if timestamp < self._done_until:
                return {
                    "phase": "DONE",
                    "step": len(self.steps),
                    "total_steps": len(self.steps),
                    "instruction": "LIVE TEST COMPLETE",
                    "thai_instruction": "จบการทดสอบแล้ว",
                    "remaining_seconds": max(0.0, self._done_until - timestamp),
                }
            return None
        return None


def core_visibility_summary(values, *, threshold: float = 0.5) -> dict[str, Any]:
    """Return beginner-readable five-core visibility without changing gating."""

    if values is None:
        return {
            name: {"label": CORE_DISPLAY_NAMES[name], "thai_label": CORE_THAI_NAMES[name], "visible": False, "visibility": None}
            for name in CORE_LANDMARKS
        }
    pose = PoseLandmarks.from_any(values)
    result = {}
    for name in CORE_LANDMARKS:
        point = pose.by_name(name)
        visibility = None if point is None or point.visibility is None else _float_or_none(point.visibility)
        result[name] = {
            "label": CORE_DISPLAY_NAMES[name],
            "thai_label": CORE_THAI_NAMES[name],
            "visible": bool(point is not None and point.present and visibility is not None and visibility >= threshold),
            "visibility": visibility,
        }
    return result


def framing_assessment(values, *, threshold: float = 0.5) -> dict[str, Any]:
    """Assess framing heuristically for setup guidance, not scientific labels."""

    visibility = core_visibility_summary(values, threshold=threshold)
    missing = [name for name in CORE_LANDMARKS if not visibility[name]["visible"]]
    result = {
        "status": "ADJUST_POSITION" if missing else "READY_TO_TEST",
        "ready": not missing,
        "missing_landmarks": missing,
        "message": "จัดตำแหน่งให้เห็นจุดสำคัญครบก่อนเริ่ม" if missing else "ดีมาก — ตำแหน่งพร้อมทดสอบ",
        "message_en": "Adjust position so the head, both shoulders, and both hips are visible" if missing else "Great — body framing is ready",
        "core_visibility": visibility,
        "heuristic_note": "Framing guidance uses normalized image coverage only; it is not a distance measurement.",
    }
    if values is None or missing:
        first = missing[0] if missing else None
        if first:
            result["message"] = f"ไม่เห็น{CORE_THAI_NAMES[first]} — ปรับตำแหน่งกล้องหรือตัวผู้ใช้"
            result["message_en"] = {
                "nose": "Head not visible",
                "left_shoulder": "Left shoulder not visible",
                "right_shoulder": "Right shoulder not visible",
                "left_hip": "Left hip not visible",
                "right_hip": "Right hip not visible",
            }[first]
        return result
    pose = PoseLandmarks.from_any(values)
    core_points = [pose.by_name(name) for name in CORE_LANDMARKS]
    xs = [float(point.x) for point in core_points if point is not None and point.present]
    ys = [float(point.y) for point in core_points if point is not None and point.present]
    if xs and ys:
        span_y = max(ys) - min(ys)
        center_x = (max(xs) + min(xs)) / 2.0
        if span_y > 0.90:
            result.update({"status": "ADJUST_POSITION", "ready": False, "message": "ตัวใหญ่เกินไป — ขยับห่างจากกล้องเล็กน้อย", "message_en": "Move farther from the camera"})
        elif span_y < 0.20:
            result.update({"status": "ADJUST_POSITION", "ready": False, "message": "ตัวเล็กเกินไป — ขยับเข้าใกล้กล้องเล็กน้อย", "message_en": "Move closer to the camera"})
        elif center_x < 0.25:
            result.update({"status": "ADJUST_POSITION", "ready": False, "message": "ขยับตัวไปทางขวาเล็กน้อย", "message_en": "Move slightly right"})
        elif center_x > 0.75:
            result.update({"status": "ADJUST_POSITION", "ready": False, "message": "ขยับตัวไปทางซ้ายเล็กน้อย", "message_en": "Move slightly left"})
    return result


def prepare_display_frame(processing_frame: np.ndarray, *, mirror: bool = True) -> np.ndarray:
    """Create the user-facing preview without changing the processing frame."""

    if processing_frame is None:
        return processing_frame
    return cv2.flip(processing_frame, 1) if mirror else processing_frame.copy()


def display_window_size(width: int, height: int, *, max_width: int = 1280, max_height: int = 720) -> tuple[int, int]:
    """Choose a large display size while preserving the camera aspect ratio."""

    if width <= 0 or height <= 0:
        return 960, 720
    scale = min(float(max_width) / width, float(max_height) / height)
    scale = max(scale, 0.1)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def parse_camera_source(value: str | int) -> int:
    """Parse and constrain the supported Windows camera index choices."""

    try:
        spec = parse_source_spec(value)
    except SourceConfigurationError as exc:
        raise ValueError("camera source must be one of the indices 0, 1, or 2") from exc
    if spec.source_type != SOURCE_USB:
        raise ValueError("camera source must be one of the indices 0, 1, or 2")
    return int(spec.resolved_source)


def _windows_camera_capture(index: int) -> cv2.VideoCapture:
    """Open one explicitly requested camera without probing other indices."""

    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(index)


def probe_camera(index: int) -> dict[str, Any]:
    """Probe exactly one camera index and release it promptly."""

    camera_index = parse_camera_source(index)
    capture = _windows_camera_capture(camera_index)
    try:
        opened = bool(capture.isOpened())
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if opened else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if opened else 0
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0
        readable = False
        if opened:
            ok, frame = capture.read()
            readable = bool(ok and frame is not None and frame.size > 0)
        return {
            "index": camera_index,
            "opened": opened,
            "readable_frame": readable,
            "width": width,
            "height": height,
            "reported_fps": fps,
            "backend": capture.getBackendName() if opened else None,
        }
    finally:
        capture.release()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _warning_message(risk_state: str) -> str:
    return {
        "NORMAL": "Posture OK",
        "OBSERVE": "Please adjust your sitting posture",
        "WARNING": "Poor posture detected",
        "HIGH_DURATION_RISK": "Sustained poor posture",
        "POSE_LOST": "Pose not visible",
    }.get(risk_state, "Posture status unavailable")


def _directional_indicator_records(flags: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for key, label in (("forward", "forward_lean"), ("left", "left_lean"), ("right", "right_lean")):
        if flags.get(key):
            records.append({
                "name": label,
                "kind": "EXPERIMENTAL",
                "status": "GEOMETRIC_INDICATOR_ONLY",
                "limitation": "Camera-relative geometry; not a V4 AI class or clinical measurement.",
            })
    return records


def _calibration_snapshot(world_pose: PoseLandmarks) -> dict[str, Any]:
    points = {
        name: world_pose.by_name(name)
        for name in ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
    }
    if any(point is None or not point.present for point in points.values()):
        return {}
    values = {
        name: np.asarray([point.x, point.y, point.z], dtype=float)
        for name, point in points.items()
    }
    shoulder_midpoint = (values["left_shoulder"] + values["right_shoulder"]) / 2.0
    hip_midpoint = (values["left_hip"] + values["right_hip"]) / 2.0
    trunk_vector = shoulder_midpoint - hip_midpoint
    return {
        "trunk_vector": [float(value) for value in trunk_vector],
        "trunk_inclination": _float_or_none(
            np.degrees(np.arctan2(trunk_vector[0], max(abs(trunk_vector[1]), 1e-8)))
        ),
        "lateral_offset": _float_or_none(trunk_vector[0]),
        "forward_back_component": _float_or_none(trunk_vector[2]),
        "shoulder_tilt": _float_or_none(
            np.degrees(np.arctan2(
                values["right_shoulder"][1] - values["left_shoulder"][1],
                values["right_shoulder"][0] - values["left_shoulder"][0] + 1e-8,
            ))
        ),
    }


def _base_record(
    *,
    frame_index: int,
    timestamp: float,
    fps: float,
    validation_step: dict[str, Any] | None,
    pose_detected: bool,
    pose_quality: float,
    core_accepted: bool,
    status: str,
    state: dict[str, Any],
    geometry: dict[str, Any] | None = None,
    geometry_relative: dict[str, Any] | None = None,
    prediction: dict[str, Any] | None = None,
    detail: str | None = None,
    calibration: dict[str, Any] | None = None,
    framing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prediction = prediction or {}
    record = {
        "timestamp": float(timestamp),
        "frame": int(frame_index),
        "pose_detected": bool(pose_detected),
        "pose_quality": float(pose_quality),
        "core_landmark_accepted": bool(core_accepted),
        "ai_posture": DISPLAY_LABELS.get(str(prediction.get("label"))) if prediction.get("label") else None,
        "ai_label": str(prediction["label"]) if prediction.get("label") else None,
        "ai_confidence": _float_or_none(prediction.get("confidence")),
        "ai_probabilities": {str(key): float(value) for key, value in (prediction.get("probabilities") or {}).items()},
        "ai_smoothed_probabilities": {str(key): float(value) for key, value in (state.get("probabilities") or {}).items()},
        "v4_feature_names": list(prediction.get("feature_names") or []),
        "v4_feature_values": [float(value) for value in (prediction.get("feature_values") or [])],
        "forward_indicator": False,
        "left_indicator": False,
        "right_indicator": False,
        "trunk_inclination": _float_or_none((geometry or {}).get("trunk_inclination_deg")),
        "shoulder_tilt": _float_or_none((geometry or {}).get("shoulder_tilt_deg")),
        "continuous_bad_duration": float(state.get("continuous_duration", 0.0) or 0.0),
        "risk_state": str(state.get("risk_level", "POSE_LOST")),
        "fps": float(fps),
        "status": status,
        "warning_event": bool(state.get("warning_event", False)),
        "risk_indicators": list(state.get("risk_indicators", [])),
        "geometry": geometry or {},
        "geometry_relative": geometry_relative or {},
        "neutral_calibration_applied": bool((geometry_relative or {}).get("neutral_baseline_applied", False)),
        "framing_status": (framing or {}).get("status", "ADJUST_POSITION"),
        "framing_message": (framing or {}).get("message", "จัดตำแหน่งให้เห็นร่างกายครบ"),
        "core_visibility": (framing or {}).get("core_visibility", core_visibility_summary(None)),
    }
    if validation_step is not None:
        record["validation_phase"] = validation_step.get("phase")
        if validation_step.get("step") is not None:
            record["validation_step"] = int(validation_step["step"])
        record["validation_instruction"] = validation_step.get("thai_instruction", validation_step.get("instruction"))
    if detail:
        record["detail"] = detail
    if calibration:
        record["calibration"] = calibration
    return record


class LiveFrameProcessor:
    """Process one RGB camera frame with the frozen V4 contract."""

    def __init__(
        self,
        runner: RGBNativeV4Runner,
        landmarker,
        config: PostureEngineConfig,
        *,
        calibration: bool = False,
    ):
        labels = frozenset(str(label) for label in runner.labels)
        if labels != V4_LABELS:
            raise ValueError("Live Webcam V1 requires the frozen two-class RGB-native V4 artifact")
        self.runner = runner
        self.landmarker = landmarker
        self.config = config
        self.calibration = calibration
        self.engine = TemporalRiskEngine(config, runner.labels)
        self.neutral_baseline: dict[str, Any] | None = None

    def set_neutral_baseline(self, baseline: dict[str, Any] | None) -> None:
        """Set or clear the runtime-only geometry reference."""

        self.neutral_baseline = None if baseline is None else dict(baseline)

    def reset_neutral_baseline(self) -> None:
        self.neutral_baseline = None

    def process(
        self,
        frame: np.ndarray,
        *,
        frame_index: int,
        timestamp_seconds: float,
        fps: float,
        validation_step: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], PoseLandmarks | None]:
        if frame is None or frame.size == 0:
            state = self.engine.update(timestamp_seconds, pose_visible=False)
            framing = framing_assessment(None, threshold=self.config.minimum_pose_visibility)
            return _base_record(
                frame_index=frame_index, timestamp=timestamp_seconds, fps=fps,
                validation_step=validation_step, pose_detected=False, pose_quality=0.0,
                core_accepted=False, status="INVALID_FRAME", state=state, framing=framing,
            ), None

        result = extract_pose_result(
            frame,
            self.landmarker,
            timestamp_ms=int(float(timestamp_seconds) * 1000.0),
        )
        if not result.pose_landmarks:
            state = self.engine.update(timestamp_seconds, pose_visible=False)
            framing = framing_assessment(None, threshold=self.config.minimum_pose_visibility)
            return _base_record(
                frame_index=frame_index, timestamp=timestamp_seconds, fps=fps,
                validation_step=validation_step, pose_detected=False, pose_quality=0.0,
                core_accepted=False, status="POSE_LOST", state=state, framing=framing,
            ), None

        image_pose = PoseLandmarks.from_any(result.pose_landmarks[0])
        framing = framing_assessment(image_pose, threshold=self.config.minimum_pose_visibility)
        if not result.pose_world_landmarks:
            state = self.engine.update(timestamp_seconds, pose_visible=False)
            return _base_record(
                frame_index=frame_index, timestamp=timestamp_seconds, fps=fps,
                validation_step=validation_step, pose_detected=True, pose_quality=0.0,
                core_accepted=False, status="WORLD_LANDMARKS_UNAVAILABLE", state=state, framing=framing,
            ), image_pose

        quality = pose_quality_score(image_pose, names=("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"), coverage_threshold=0.3)
        policy = evaluate_visibility_policy(
            image_pose,
            "core-only",
            strict_threshold=self.config.minimum_pose_visibility,
        )
        if not policy["accepted"]:
            state = self.engine.update(timestamp_seconds, pose_visible=False)
            detail = "core_visibility=" + ",".join(policy["low_names"] or policy["missing_names"])
            return _base_record(
                frame_index=frame_index, timestamp=timestamp_seconds, fps=fps,
                validation_step=validation_step, pose_detected=True,
                pose_quality=float(quality["score"]), core_accepted=False,
                status="CORE_LANDMARKS_NOT_ACCEPTED", state=state, detail=detail, framing=framing,
            ), image_pose

        world_pose = PoseLandmarks.from_any(result.pose_world_landmarks[0])
        try:
            prediction = self.runner.predict_world_landmarks(
                result.pose_world_landmarks[0],
                image_visibility=result.pose_landmarks[0],
                visibility_threshold=self.config.minimum_pose_visibility,
            )
        except (ValueError, PoseValidationError) as exc:
            state = self.engine.update(timestamp_seconds, pose_visible=False)
            return _base_record(
                frame_index=frame_index, timestamp=timestamp_seconds, fps=fps,
                validation_step=validation_step, pose_detected=True,
                pose_quality=float(quality["score"]), core_accepted=True,
                status="V4_ADAPTER_REJECTED", state=state, detail=str(exc), framing=framing,
            ), image_pose

        geometry = calculate_geometry(world_pose)
        geometry_relative = relative_geometry(geometry, self.neutral_baseline)
        flags = evaluate_directional_geometry(geometry_relative, thresholds=self.config.indicator_thresholds)
        indicator_records = evaluate_indicators(
            geometry_relative,
            str(prediction["label"]),
            confidence=float(prediction["confidence"]),
            thresholds=self.config.indicator_thresholds,
        )
        indicator_records.extend(_directional_indicator_records(flags))
        calibration_snapshot = _calibration_snapshot(world_pose) if self.calibration else None

        if float(prediction["confidence"]) < self.config.minimum_confidence:
            state = self.engine.update(timestamp_seconds, pose_visible=False, geometry=geometry, indicators=indicator_records)
            record = _base_record(
                frame_index=frame_index, timestamp=timestamp_seconds, fps=fps,
                validation_step=validation_step, pose_detected=True,
                pose_quality=float(quality["score"]), core_accepted=True,
                status="AI_LOW_CONFIDENCE", state=state, geometry=geometry,
                geometry_relative=geometry_relative, prediction=prediction,
                calibration=calibration_snapshot, framing=framing,
            )
            record.update({
                "forward_indicator": bool(flags["forward"]),
                "left_indicator": bool(flags["left"]),
                "right_indicator": bool(flags["right"]),
            })
            return record, image_pose

        state = self.engine.update(
            timestamp_seconds,
            pose_visible=True,
            probabilities=np.asarray([prediction["probabilities"][label] for label in self.runner.labels], dtype=float),
            confidence=float(prediction["confidence"]),
            geometry=geometry,
            indicators=indicator_records,
        )
        record = _base_record(
            frame_index=frame_index, timestamp=timestamp_seconds, fps=fps,
            validation_step=validation_step, pose_detected=True,
            pose_quality=float(quality["score"]), core_accepted=True,
            status="OK", state=state, geometry=geometry,
            geometry_relative=geometry_relative, prediction=prediction,
            calibration=calibration_snapshot, framing=framing,
        )
        record.update({
            "forward_indicator": bool(flags["forward"]),
            "left_indicator": bool(flags["left"]),
            "right_indicator": bool(flags["right"]),
        })
        return record, image_pose


class LiveSessionAccumulator:
    """Collect live telemetry and compute a descriptive session summary."""

    def __init__(self, *, frame_interval_seconds: float = 1.0 / 30.0):
        if frame_interval_seconds <= 0:
            raise ValueError("frame_interval_seconds must be positive")
        self.frame_interval_seconds = float(frame_interval_seconds)
        self.records: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def summary(self, *, total_duration_seconds: float | None = None) -> dict[str, Any]:
        if not self.records:
            return {
                "records": 0,
                "total_session_duration_seconds": float(total_duration_seconds or 0.0),
                "valid_pose_time_seconds": 0.0,
                "pose_lost_percentage": 0.0,
                "pose_detection_percentage": 0.0,
                "core_landmark_acceptance_percentage": 0.0,
                "input_quality_status": "NEEDS_CALIBRATION",
                "upright_percentage": 0.0,
                "lean_backward_percentage": 0.0,
                "forward_indicator_percentage": 0.0,
                "left_indicator_percentage": 0.0,
                "right_indicator_percentage": 0.0,
                "warning_count": 0,
                "longest_continuous_poor_posture_seconds": 0.0,
                "average_fps": 0.0,
                "interpretation": "Descriptive session telemetry only; not a medical or health score.",
            }

        timestamps = [float(record.get("timestamp", 0.0)) for record in self.records]
        total = float(total_duration_seconds) if total_duration_seconds is not None else max(timestamps[-1], 0.0) + self.frame_interval_seconds
        total = max(total, self.frame_interval_seconds)
        durations: list[float] = []
        for index, timestamp in enumerate(timestamps):
            next_timestamp = timestamps[index + 1] if index + 1 < len(timestamps) else total
            durations.append(max(0.0, min(total - timestamp, next_timestamp - timestamp)))

        def duration_where(predicate) -> float:
            return float(sum(delta for delta, record in zip(durations, self.records) if predicate(record)))

        valid = lambda record: bool(record.get("pose_detected")) and bool(record.get("core_landmark_accepted")) and bool(record.get("ai_posture"))
        valid_time = duration_where(valid)
        pose_detection_time = duration_where(lambda record: bool(record.get("pose_detected")))
        core_acceptance_time = duration_where(lambda record: bool(record.get("pose_detected")) and bool(record.get("core_landmark_accepted")))
        pose_lost_time = duration_where(lambda record: str(record.get("risk_state")) == "POSE_LOST" or not record.get("pose_detected"))
        denominator = valid_time or total
        result = {
            "records": len(self.records),
            "total_session_duration_seconds": total,
            "valid_pose_time_seconds": valid_time,
            "valid_pose_percentage": 100.0 * valid_time / total if total else 0.0,
            "pose_detection_percentage": 100.0 * pose_detection_time / total if total else 0.0,
            "core_landmark_acceptance_percentage": 100.0 * core_acceptance_time / total if total else 0.0,
            "input_quality_status": "PASS" if (100.0 * core_acceptance_time / total if total else 0.0) >= 80.0 else "NEEDS_CALIBRATION",
            "pose_lost_percentage": 100.0 * pose_lost_time / total if total else 0.0,
            "upright_percentage": 100.0 * duration_where(lambda record: valid(record) and record.get("ai_label") == "TUP") / denominator if denominator else 0.0,
            "lean_backward_percentage": 100.0 * duration_where(lambda record: valid(record) and record.get("ai_label") == "TLB") / denominator if denominator else 0.0,
            "forward_indicator_percentage": 100.0 * duration_where(lambda record: valid(record) and record.get("forward_indicator")) / denominator if denominator else 0.0,
            "left_indicator_percentage": 100.0 * duration_where(lambda record: valid(record) and record.get("left_indicator")) / denominator if denominator else 0.0,
            "right_indicator_percentage": 100.0 * duration_where(lambda record: valid(record) and record.get("right_indicator")) / denominator if denominator else 0.0,
            "warning_count": int(sum(bool(record.get("warning_event")) for record in self.records)),
            "longest_continuous_poor_posture_seconds": max(float(record.get("continuous_bad_duration", 0.0) or 0.0) for record in self.records),
            "average_fps": float(np.mean([float(record.get("fps", 0.0) or 0.0) for record in self.records])),
            "interpretation": "Descriptive session telemetry only; not a medical or health score.",
        }
        step_numbers = sorted({record.get("validation_step") for record in self.records if record.get("validation_step") is not None})
        if step_numbers:
            result["validation_steps"] = {
                str(step): {
                    "frames": sum(1 for record in self.records if record.get("validation_step") == step),
                    "pose_detection_rate": float(np.mean([bool(record.get("pose_detected")) for record in self.records if record.get("validation_step") == step])),
                    "core_acceptance_rate": float(np.mean([bool(record.get("core_landmark_accepted")) for record in self.records if record.get("validation_step") == step])),
                    "input_quality_status": "PASS" if float(np.mean([bool(record.get("core_landmark_accepted")) for record in self.records if record.get("validation_step") == step])) >= 0.8 else "NEEDS_CALIBRATION",
                    "ai_labels_observed": sorted({record.get("ai_label") for record in self.records if record.get("validation_step") == step and record.get("ai_label")}),
                    "risk_states_observed": sorted({record.get("risk_state") for record in self.records if record.get("validation_step") == step}),
                }
                for step in step_numbers
            }
            result["validation_note"] = "Validation steps are behavioral checks only; no automatic labels or accuracy are produced."
        return result


def write_live_outputs(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    output_root: str | Path = "outputs/live",
    session_id: str | None = None,
) -> dict[str, str]:
    root = Path(output_root)
    session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = root / "logs" / f"webcam_{session_id}.jsonl"
    summary_path = root / "session_reports" / f"webcam_{session_id}_summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"event_log": str(log_path), "session_report": str(summary_path)}


def live_video_path(output_root: str | Path, session_id: str) -> Path:
    """Return the deterministic optional annotated-recording destination."""

    return Path(output_root) / "videos" / f"webcam_{session_id}.mp4"


def release_live_resources(capture=None, landmarker=None, writer=None) -> None:
    """Release every live resource; safe to call during partial startup."""

    if landmarker is not None:
        landmarker.close()
    if capture is not None:
        capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


class SoundNotifier:
    """Optional Windows warning sound; the live prototype never depends on it."""

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)

    def notify(self) -> None:
        if not self.enabled or os.name != "nt":
            return
        try:
            import winsound

            winsound.MessageBeep(winsound.MB_ICONWARNING)
        except Exception:
            return


def draw_live_overlay(
    frame: np.ndarray,
    record: dict[str, Any],
    image_pose: PoseLandmarks | None = None,
    *,
    validation_step: dict[str, Any] | None = None,
    calibration: bool = False,
    mirror: bool = True,
    final_summary: dict[str, Any] | None = None,
) -> np.ndarray:
    """Draw small edge overlays while keeping the camera image dominant."""

    output = frame.copy()
    height, width = output.shape[:2]
    coords: dict[int, tuple[int, int]] = {}
    if image_pose is not None:
        for index, point in enumerate(image_pose.points):
            if point is None or not point.present:
                continue
            normalized_x = 1.0 - float(point.x) if mirror else float(point.x)
            x = int(np.clip(normalized_x, 0.0, 1.0) * max(width - 1, 1))
            y = int(np.clip(point.y, 0.0, 1.0) * max(height - 1, 1))
            coords[index] = (x, y)
            cv2.circle(output, (x, y), 2, (0, 220, 255), -1)
        for left, right in SKELETON_CONNECTIONS:
            if left in coords and right in coords:
                cv2.line(output, coords[left], coords[right], (255, 180, 0), 1)

    risk_state = str(record.get("risk_state", "POSE_LOST"))
    framing = record.get("framing", {})
    framing_status = str(record.get("framing_status", framing.get("status", "ADJUST_POSITION")))
    core_visibility = record.get("core_visibility", {})
    stable_ready = bool(record.get("stable_ready", framing_status == "READY_TO_TEST"))
    framing_ok = framing_status == "READY_TO_TEST" and stable_ready

    def _panel(x: int, y: int, panel_width: int, panel_height: int, color: tuple[int, int, int], alpha: float = 0.56) -> None:
        x2 = min(width - 1, max(x, 0) + panel_width)
        y2 = min(height - 1, max(y, 0) + panel_height)
        overlay = output.copy()
        cv2.rectangle(overlay, (max(x, 0), max(y, 0)), (x2, y2), color, -1)
        output[:] = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0)

    def _draw_text(lines: list[str], x: int, y: int, *, scale: float = 0.48, spacing: int = 21, color: tuple[int, int, int] = (255, 255, 255)) -> None:
        if not any(any(ord(char) > 127 for char in line) for line in lines):
            for offset, line in enumerate(lines):
                yy = y + offset * spacing
                cv2.putText(output, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(output, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            return
        try:
            from PIL import Image, ImageDraw, ImageFont

            font = ImageFont.truetype("C:\\Windows\\Fonts\\tahoma.ttf", max(14, int(scale * 32)))
            pil_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            drawer = ImageDraw.Draw(pil_image)
            for offset, line in enumerate(lines):
                yy = y + offset * spacing
                drawer.text((x + 1, yy - 2), line, font=font, fill=(0, 0, 0), stroke_width=1, stroke_fill=(0, 0, 0))
                drawer.text((x, yy - 3), line, font=font, fill=tuple(reversed(color)))
            output[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        except Exception:
            for offset, line in enumerate(lines):
                ascii_line = line.encode("ascii", "replace").decode("ascii")
                yy = y + offset * spacing
                cv2.putText(output, ascii_line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(output, ascii_line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    # Small top-left AI/status card.
    _panel(10, 10, 300, 178 if calibration else 146, (20, 55, 30) if risk_state == "NORMAL" else (55, 45, 25), 0.58)
    posture_text = str(record.get("ai_posture") or "—")
    confidence_text = f"{100.0 * float(record['ai_confidence']):.0f}%" if record.get("ai_confidence") is not None else "n/a"
    ai_lines = [
        f"POSE: {'DETECTED' if record.get('pose_detected') else 'LOST'}",
        "AI POSTURE: " + posture_text,
        "CONFIDENCE: " + confidence_text,
        "STATE: " + {"HIGH_DURATION_RISK": "HIGH DURATION RISK"}.get(risk_state, risk_state),
        f"BAD POSTURE: {float(record.get('continuous_bad_duration', 0.0)):.1f}s",
    ]
    if calibration:
        probabilities = record.get("ai_probabilities", {})
        ai_lines.extend([
            f"V4 TUP p: {100.0 * float(probabilities.get('TUP', 0.0)):.1f}%",
            f"V4 TLB p: {100.0 * float(probabilities.get('TLB', 0.0)):.1f}%",
        ])
    _draw_text(ai_lines, 20, 35, spacing=20)

    # Small top-right framing/visibility card.
    _panel(max(10, width - 325), 10, 315, 190, (20, 55, 30) if framing_ok else (65, 45, 25), 0.58)
    if framing_ok:
        visibility_status = "STATUS: READY TO TEST"
    elif framing_status == "READY_TO_TEST":
        visibility_status = "STATUS: STABILIZING INPUT"
    else:
        visibility_status = "STATUS: ADJUST POSITION"
    visibility_lines = ["BODY VISIBILITY", visibility_status]
    for name in CORE_LANDMARKS:
        item = core_visibility.get(name, {})
        visibility_lines.append(f"{CORE_DISPLAY_NAMES[name]} {'✓' if item.get('visible') else '✗'}")
    visibility_lines.append(str(record.get("framing_message", "จัดตำแหน่งให้เห็นร่างกายครบ")))
    visibility_lines.append(str(record.get("input_state", "INPUT GOOD" if record.get("core_landmark_accepted") else "INPUT LOST")))
    _draw_text(visibility_lines, max(20, width - 315), 35, scale=0.43, spacing=20)

    # Compact bottom-left geometry/risk card.
    _panel(10, max(10, height - 158), 365, 148, (35, 45, 70), 0.56)
    geometry_lines = [
        "GEOMETRY - EXPERIMENTAL",
        "NEUTRAL RELATIVE" if record.get("neutral_calibration_applied") else "ABSOLUTE / UNCALIBRATED",
        f"Forward Lean: {'YES' if record.get('forward_indicator') else 'NO'}",
        f"Left Lean:    {'YES' if record.get('left_indicator') else 'NO'}",
        f"Right Lean:   {'YES' if record.get('right_indicator') else 'NO'}",
        _warning_message(risk_state),
        "Q / ESC: EXIT",
    ]
    _draw_text(geometry_lines, 20, max(35, height - 133), scale=0.45, spacing=21)

    # Calibration is intentionally small and appears only when requested.
    if calibration and record.get("calibration"):
        calibration_values = record["calibration"]
        _panel(max(10, width - 375), max(10, height - 160), 365, 150, (55, 35, 70), 0.56)
        _draw_text([
            "CALIBRATION",
            f"trunk: {np.round(calibration_values.get('trunk_vector', []), 2).tolist()}",
            f"inclination: {_float_or_none(calibration_values.get('trunk_inclination'))}",
            f"lateral: {_float_or_none(calibration_values.get('lateral_offset'))}",
            f"forward/back: {_float_or_none(calibration_values.get('forward_back_component'))}",
            f"shoulder tilt: {_float_or_none(calibration_values.get('shoulder_tilt'))}",
        ], max(20, width - 365), max(35, height - 133), scale=0.40, spacing=20)

    # One instruction card at the bottom center, never a full-screen panel.
    if validation_step:
        phase = validation_step.get("phase")
        card_width, card_height = min(600, max(420, width // 2)), 118
        card_x = max(10, (width - card_width) // 2)
        card_y = max(10, height - 145)
        _panel(card_x, card_y, card_width, card_height, (20, 60, 90), 0.72)
        if phase == "SETUP":
            title = "CAMERA SETUP"
            instruction = validation_step.get("instruction", "จัดตำแหน่งให้เห็นศีรษะ ไหล่ทั้งสอง และสะโพกทั้งสอง")
            action = "กด SPACE เมื่อพร้อมเริ่ม" if validation_step.get("ready_to_start") else "ปรับตำแหน่งก่อน แล้วกด SPACE"
            card_lines = [title, instruction, action]
        elif phase == "NEUTRAL_CALIBRATION":
            card_lines = [
                "NEUTRAL CALIBRATION",
                "นั่งตัวตรงตามธรรมชาติ แล้วค้างไว้",
                f"ค้างอีก {max(0.0, float(validation_step.get('remaining_seconds', 0.0))):.1f} วินาที",
                f"INPUT: {validation_step.get('input_state', 'INPUT LOST')}",
            ]
        elif phase == "WAITING":
            if validation_step.get("requires_reacquisition"):
                action = "รอให้เห็นร่างกายครบ แล้วกด SPACE"
            else:
                action = "พร้อมแล้วกด SPACE เพื่อเริ่ม"
            card_lines = [
                f"TEST {validation_step['step']} / {validation_step['total_steps']}",
                validation_step.get("thai_instruction", "เตรียมท่า"),
                action,
            ]
        elif phase == "PREPARE":
            card_lines = [f"TEST {validation_step['step']} / {validation_step['total_steps']}", "เตรียมตัว", f"เริ่มใน {validation_step.get('countdown', 1)}"]
        elif phase == "COMPLETE":
            card_lines = ["STEP COMPLETE", validation_step.get("thai_instruction", "จบขั้นตอนนี้แล้ว"), "กด SPACE เพื่อไปต่อ หรือรออัตโนมัติ"]
        elif phase == "DONE":
            summary = final_summary or {}
            camera_status = "PASS" if summary.get("records") else "NEEDS_CALIBRATION"
            mediapipe_status = "PASS" if float(summary.get("pose_detection_percentage", 0.0) or 0.0) >= 80.0 else "NEEDS_CALIBRATION"
            body_status = "PASS" if float(summary.get("core_landmark_acceptance_percentage", 0.0) or 0.0) >= 80.0 else "NEEDS_CALIBRATION"
            card_lines = [
                "LIVE TEST COMPLETE",
                f"Camera: {camera_status}    MediaPipe: {mediapipe_status}",
                f"Body visibility: {body_status}    Input quality: {summary.get('input_quality_status', 'NEEDS_CALIBRATION')}",
                f"Core accepted: {summary.get('core_landmark_acceptance_percentage', 0.0):.0f}%    Average FPS: {summary.get('average_fps', 0.0):.1f}",
                "AI / geometry: diagnostic only; no accuracy claim",
            ]
        else:
            card_lines = [
                f"TEST {validation_step['step']} / {validation_step['total_steps']}",
                validation_step.get("thai_instruction", validation_step.get("instruction", "ทำตามคำแนะนำ")),
                f"ค้างท่านี้อีก {max(0.0, float(validation_step.get('remaining_seconds', 0.0))):.1f} วินาที",
                validation_step.get("input_state", "INPUT GOOD") if validation_step.get("timer_paused") else "INPUT GOOD — timer running",
            ]
        _draw_text(card_lines, card_x + 18, card_y + 28, scale=0.47, spacing=22)

    # Persistent bottom control hint.
    controls = "SPACE: START / NEXT    R: RESTART STEP    Q / ESC: EXIT"
    cv2.putText(output, controls, (max(10, width // 2 - 260), height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, controls, (max(10, width // 2 - 260), height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def run_webcam_session(
    source: str | int = "0",
    *,
    model_dir: str | Path = "outputs/models/rgb_native_v4",
    pose_model: str | Path = "metadata/pose_landmarker_heavy.task",
    config_path: str | Path = "configs/posture_engine.yaml",
    validation: bool = False,
    calibration: bool = False,
    sound: bool = False,
    record: bool = False,
    output_root: str | Path = "outputs/live",
    validation_step_seconds: float = 5.0,
    validation_poor_hold_seconds: float = 20.0,
    neutral_calibration_seconds: float = 4.0,
    stable_core_seconds: float = 1.5,
    mirror: bool = True,
    max_frames: int | None = None,
    performance_preset: str = "DESKTOP",
    process_every: int | None = None,
) -> dict[str, Any]:
    """Run the live prototype after the caller explicitly requests a camera."""

    performance = resolve_performance_preset(performance_preset, process_every)
    camera_index = parse_camera_source(source)
    config = PostureEngineConfig.from_yaml(config_path)
    runner = RGBNativeV4Runner(model_dir)
    capture = _windows_camera_capture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Camera {camera_index} could not be opened. Run check_camera.bat and retry with --source 0, 1, or 2."
        )

    validation_controller = ValidationController(
        step_seconds=validation_step_seconds,
        poor_hold_seconds=validation_poor_hold_seconds,
    ) if validation else None
    neutral_calibrator = NeutralBaselineCalibrator(required_seconds=neutral_calibration_seconds) if validation else None
    stable_core_gate = StableCoreGate(required_seconds=stable_core_seconds) if validation else None
    landmarker = None
    writer = None
    records: list[dict[str, Any]] = []
    session = LiveSessionAccumulator(frame_interval_seconds=config.frame_interval_seconds)
    notifier = SoundNotifier(sound)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    video_path = None
    started = time.perf_counter()
    frame_index = 0
    processed_frames = 0
    skipped_frames = 0
    last_record: dict[str, Any] | None = None
    last_image_pose: PoseLandmarks | None = None
    capture_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture_fps = capture_fps if capture_fps > 0 else 30.0

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if performance["capture_width"]:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(performance["capture_width"]))
        if performance["capture_height"]:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(performance["capture_height"]))
        window_name = "Office Posture - Live Webcam V1"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        display_width, display_height = display_window_size(width, height)
        cv2.resizeWindow(window_name, display_width, display_height)
        if record and width > 0 and height > 0:
            video_path = live_video_path(output_root, session_id)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                capture_fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create recording: {video_path}")
        landmarker = create_pose_landmarker(pose_model, running_mode="video")
        processor = LiveFrameProcessor(runner, landmarker, config, calibration=calibration)
        while max_frames is None or frame_index < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                break
            if record and writer is None:
                frame_height, frame_width = frame.shape[:2]
                video_path = live_video_path(output_root, session_id)
                video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    capture_fps,
                    (frame_width, frame_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create recording: {video_path}")
            elapsed = time.perf_counter() - started
            if validation_controller is None:
                current_step = None
            elif neutral_calibrator is not None and neutral_calibrator.active:
                neutral_status = neutral_calibrator.status(elapsed)
                current_step = {
                    "phase": "NEUTRAL_CALIBRATION",
                    "title": "NEUTRAL CALIBRATION",
                    "instruction": "นั่งตัวตรงตามธรรมชาติ แล้วค้างไว้",
                    "thai_instruction": "นั่งตัวตรงตามธรรมชาติ แล้วค้างไว้",
                    "remaining_seconds": max(0.0, neutral_status["required_seconds"] - neutral_status["stable_seconds"]),
                    "stable_seconds": neutral_status["stable_seconds"],
                    "required_seconds": neutral_status["required_seconds"],
                    "observation_count": neutral_status["observation_count"],
                    "input_state": "INPUT GOOD" if neutral_status["observation_count"] else "INPUT LOST",
                }
            else:
                current_step = validation_controller.current(elapsed)
            if validation_controller is not None and validation_controller.started and current_step is None:
                break
            wall_elapsed = max(time.perf_counter() - started, 1e-8)
            display_fps = (frame_index + 1) / wall_elapsed
            if should_process_frame(frame_index, performance["process_every"], last_record is not None):
                record_item, image_pose = processor.process(
                    frame,
                    frame_index=frame_index,
                    timestamp_seconds=elapsed,
                    fps=display_fps,
                    validation_step=current_step,
                )
                record_item["inference_skipped"] = False
                if current_step and current_step.get("phase") == "SETUP":
                    current_step = dict(current_step)
                    stable = stable_core_gate.update(
                        accepted=bool(record_item.get("core_landmark_accepted")),
                        timestamp_seconds=elapsed,
                    ) if stable_core_gate is not None else {"ready": False}
                    record_item["stable_core_gate"] = stable
                    record_item["stable_ready"] = bool(stable["ready"])
                    current_step["ready_to_start"] = bool(stable["ready"])
                elif stable_core_gate is not None:
                    stable = stable_core_gate.update(
                        accepted=bool(record_item.get("core_landmark_accepted")),
                        timestamp_seconds=elapsed,
                    )
                    record_item["stable_core_gate"] = stable
                    record_item["stable_ready"] = bool(stable["ready"])

                input_good = bool(record_item.get("core_landmark_accepted"))
                if validation_controller is not None:
                    input_status = validation_controller.observe_input(elapsed, input_good=input_good)
                    record_item.update({
                        "input_state": input_status["input_state"],
                        "input_timer_paused": bool(input_status["timer_paused"]),
                    })
                    if current_step is not None:
                        current_step = dict(current_step)
                        current_step.update(input_status)
                if neutral_calibrator is not None and neutral_calibrator.active:
                    neutral_status = neutral_calibrator.observe(
                        record_item.get("geometry"),
                        accepted=input_good,
                        timestamp_seconds=elapsed,
                    )
                    record_item["neutral_calibration_status"] = neutral_status
                    if neutral_status["completed"]:
                        processor.set_neutral_baseline(neutral_calibrator.baseline)
                        validation_controller.start(elapsed, wait_for_confirmation=True)
                        record_item["neutral_baseline"] = neutral_calibrator.baseline
                elif neutral_calibrator is not None and neutral_calibrator.completed:
                    record_item["neutral_calibration_status"] = neutral_calibrator.status(elapsed)
                records.append(record_item)
                session.add(record_item)
                last_record = deepcopy(record_item)
                last_image_pose = image_pose
                processed_frames += 1
                if record_item.get("warning_event"):
                    notifier.notify()
            else:
                # Keep the preview responsive without claiming a fresh pose.
                record_item = deepcopy(last_record)
                image_pose = last_image_pose
                record_item.update({
                    "frame": int(frame_index),
                    "timestamp": float(elapsed),
                    "fps": float(display_fps),
                    "inference_skipped": True,
                    "inference_skip_reason": "performance_preset_frame_skip",
                })
                records.append(record_item)
                skipped_frames += 1
            display_frame = prepare_display_frame(frame, mirror=mirror)
            annotated = draw_live_overlay(
                display_frame,
                record_item,
                image_pose,
                validation_step=current_step,
                calibration=calibration,
                mirror=mirror,
                final_summary=session.summary(total_duration_seconds=elapsed) if current_step and current_step.get("phase") == "DONE" else None,
            )
            if writer is not None:
                writer.write(annotated)
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and validation_controller is not None:
                if not validation_controller.started:
                    if neutral_calibrator is not None and neutral_calibrator.active:
                        pass
                    elif record_item.get("stable_ready"):
                        neutral_calibrator.start(elapsed) if neutral_calibrator is not None else validation_controller.start(elapsed)
                elif current_step and current_step.get("phase") == "WAITING":
                    if record_item.get("stable_ready"):
                        validation_controller.advance(elapsed)
                elif current_step and current_step.get("phase") in {"PREPARE", "ACTIVE", "COMPLETE"}:
                    validation_controller.advance(elapsed)
            elif key == ord("r") and validation_controller is not None:
                if neutral_calibrator is not None and neutral_calibrator.active:
                    neutral_calibrator.reset()
                    stable_core_gate.reset() if stable_core_gate is not None else None
                elif not validation_controller.started:
                    neutral_calibrator.reset() if neutral_calibrator is not None else None
                    stable_core_gate.reset() if stable_core_gate is not None else None
                else:
                    validation_controller.restart_current(elapsed)
            frame_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        release_live_resources(capture, landmarker, writer)

    total_duration = max(0.0, time.perf_counter() - started)
    summary = session.summary(total_duration_seconds=total_duration)
    summary.update({
        "session_id": session_id,
        "camera_index": camera_index,
        "validation_mode": bool(validation),
        "calibration_mode": bool(calibration),
        "recording_enabled": bool(record),
        "sound_enabled": bool(sound),
        "performance_preset": performance["name"],
        "performance_description": performance["description"],
        "process_every": int(performance["process_every"]),
        "displayed_frames": int(frame_index),
        "processed_frames": int(processed_frames),
        "inference_skipped_frames": int(skipped_frames),
        "capture_requested_width": performance["capture_width"],
        "capture_requested_height": performance["capture_height"],
        "software_status": "READY_FOR_HARDWARE_TEST",
        "hardware_status": "HARDWARE_TEST_COMPLETED" if records else "WAITING_FOR_WEBCAM",
        "validation_flow": "CAMERA_SETUP -> NEUTRAL_CALIBRATION -> USER_CONFIRMED_STEPS" if validation else "LIVE_DEMO",
        "stable_core_gate_seconds": stable_core_gate.required_seconds if stable_core_gate is not None else None,
        "neutral_calibration_status": "COMPLETE" if neutral_calibrator is not None and neutral_calibrator.completed else ("NOT_COMPLETED" if neutral_calibrator is not None else "NOT_REQUESTED"),
    })
    if neutral_calibrator is not None and neutral_calibrator.baseline is not None:
        summary["neutral_baseline"] = neutral_calibrator.baseline
    paths = write_live_outputs(records, summary, output_root=output_root, session_id=session_id)
    if video_path is not None:
        paths["annotated_video"] = str(video_path)
    return {"summary": summary, "paths": paths, "records": records}
