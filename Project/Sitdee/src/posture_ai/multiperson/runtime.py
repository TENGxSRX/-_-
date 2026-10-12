"""Per-person V4, geometry, temporal, and low-spec orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.posture_ai.ergonomics.calibration import relative_geometry
from src.posture_ai.ergonomics.activity_gate import ActivityGate, ActivityGateConfig
from src.posture_ai.ergonomics.hierarchical import HierarchicalEvidence, decide_hierarchical_posture
from src.posture_ai.ergonomics.geometry import calculate_geometry
from src.posture_ai.ergonomics.indicators import evaluate_directional_geometry, evaluate_indicators
from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig, TemporalRiskEngine

from .schema import MultiPersonFrameResult
from .tracking import AnonymousPoseTracker, PoseDetection


PERFORMANCE_PRESETS = {
    "DESKTOP": {"process_every": 1, "capture_width": None, "capture_height": None, "description": "Process every captured frame."},
    "BALANCED": {"process_every": 2, "capture_width": 960, "capture_height": 540, "description": "Process every second captured frame."},
    "LOW_SPEC": {"process_every": 3, "capture_width": 640, "capture_height": 480, "description": "Process every third captured frame at a smaller capture size."},
    "RTSP_CCTV": {"process_every": 1, "capture_width": 960, "capture_height": 540, "description": "RTSP latest-frame mode with a 960x540 inference copy."},
}


def resolve_performance_preset(name: str = "LOW_SPEC", process_every: int | None = None) -> dict[str, Any]:
    """Return explicit frame-skip settings without changing V4 features."""

    preset = str(name).upper()
    if preset not in PERFORMANCE_PRESETS:
        raise ValueError(f"Unknown performance preset: {name!r}")
    interval = int(process_every) if process_every is not None else int(PERFORMANCE_PRESETS[preset]["process_every"])
    if interval < 1:
        raise ValueError("process_every must be at least 1")
    return {
        "name": preset,
        "process_every": interval,
        "capture_width": PERFORMANCE_PRESETS[preset]["capture_width"],
        "capture_height": PERFORMANCE_PRESETS[preset]["capture_height"],
        "description": PERFORMANCE_PRESETS[preset]["description"],
        "override": process_every is not None,
    }


def should_process_frame(frame_index: int, process_every: int, *, has_previous_result: bool = False) -> bool:
    """Schedule inference while keeping captured-frame timestamps unchanged."""

    interval = int(process_every)
    if interval < 1:
        raise ValueError("process_every must be at least 1")
    return not has_previous_result or int(frame_index) % interval == 0


@dataclass(frozen=True)
class MultiPersonRuntimeConfig:
    max_people: int = 3
    max_association_distance: float = 0.20
    max_misses: int = 3
    reacquisition_window_seconds: float = 1.0
    visibility_threshold: float = 0.5
    camera_id: str = "CAM01"
    activity_gate: ActivityGateConfig | None = None


class MultiPersonRuntime:
    """Enrich anonymous tracks with independent frozen-V4 and risk state."""

    def __init__(
        self,
        v4_runner,
        *,
        runtime_config: MultiPersonRuntimeConfig | None = None,
        posture_config: PostureEngineConfig | None = None,
    ) -> None:
        self.config = runtime_config or MultiPersonRuntimeConfig()
        self.posture_config = posture_config or PostureEngineConfig()
        self.v4_runner = v4_runner
        self.labels = list(getattr(v4_runner, "labels", ["TLB", "TUP"]))
        if not self.labels:
            raise ValueError("V4 runner must expose at least one label")
        self.tracker = AnonymousPoseTracker(
            max_people=self.config.max_people,
            max_association_distance=self.config.max_association_distance,
            max_misses=self.config.max_misses,
            reacquisition_window_seconds=self.config.reacquisition_window_seconds,
        )
        self._risk_engines: dict[str, TemporalRiskEngine] = {}
        self._neutral_baselines: dict[str, dict[str, Any]] = {}
        self._activity_gates: dict[str, ActivityGate] = {}
        self._activity_gate_config = self.config.activity_gate or self.posture_config.activity_gate
        self._last_frame: MultiPersonFrameResult | None = None

    @property
    def last_frame(self) -> MultiPersonFrameResult | None:
        return self._last_frame

    @property
    def neutral_baselines(self) -> dict[str, dict[str, Any]]:
        return {subject_id: dict(value) for subject_id, value in self._neutral_baselines.items()}

    def reset(self) -> None:
        self.tracker.reset()
        self._risk_engines.clear()
        self._neutral_baselines.clear()
        self._activity_gates.clear()
        self._last_frame = None

    def set_neutral_baseline(self, subject_id: str, baseline: dict[str, Any]) -> None:
        """Set a runtime-only baseline for one anonymous subject, never globally."""

        subject = str(subject_id)
        if self.tracker.track_state(subject) is None:
            raise KeyError(f"Unknown runtime subject: {subject}")
        self._neutral_baselines[subject] = dict(baseline)

    def _engine(self, subject_id: str) -> TemporalRiskEngine:
        return self._risk_engines.setdefault(
            subject_id,
            TemporalRiskEngine(self.posture_config, labels=self.labels),
        )

    def _activity_gate(self, subject_id: str) -> ActivityGate:
        return self._activity_gates.setdefault(subject_id, ActivityGate(self._activity_gate_config))

    def _lost_geometry(self) -> dict[str, Any]:
        return {
            "valid": False,
            "missing_landmarks": [],
            "status": "POSE_LOST_OR_CORE_INVALID",
            "neutral_baseline_applied": False,
        }

    def _hierarchical_state(self, track) -> dict[str, Any]:
        """Expose an evidence-gated hierarchy without inventing missing models."""

        activity_label = track.activity_state if track.activity_state in {"SITTING", "STANDING"} else None
        decision = decide_hierarchical_posture(
            HierarchicalEvidence(
                pose_valid=bool(track.core_valid),
                activity_label=activity_label,
                activity_confidence=track.activity_confidence,
                v4_prediction=track.v4_prediction,
                v4_probabilities=track.v4_probabilities or {},
                geometry=track.geometry,
                quality_reason=None if track.core_valid else "CORE_INVALID",
            )
        )
        result = decision.to_dict()
        result["activity_model_status"] = "NOT_TRAINED"
        result["activity_gate_status"] = "EXPERIMENTAL_GEOMETRY_GATE"
        result["sitting_quality_model_status"] = "NOT_TRAINED"
        result["subtype_model_status"] = "NOT_TRAINED"
        return result

    def _update_activity(self, track, timestamp_seconds: float) -> dict[str, Any] | None:
        previous = track.activity_state
        if track.tracking_state not in {"NEW", "ACTIVE"} or not track.core_valid:
            track.activity_state = "UNKNOWN"
            track.activity_raw_state = "UNKNOWN"
            track.activity_confidence = 0.0
            track.activity_method = "GEOMETRY_EXPERIMENTAL"
            track.activity_evidence = {"state": "UNKNOWN", "reason": "POSE_OR_CORE_INVALID", "method": "GEOMETRY_EXPERIMENTAL"}
            return None
        result = self._activity_gate(track.subject_id).update(timestamp_seconds, track.landmarks)
        track.activity_state = str(result["state"])
        track.activity_raw_state = str(result["raw_state"])
        track.activity_confidence = float(result["confidence"])
        track.activity_method = str(result["method"])
        track.activity_evidence = dict(result)
        if track.activity_state == previous:
            return None
        event_name = {
            "SITTING": "SIT_DETECTED",
            "STANDING": "STAND_DETECTED",
            "UNKNOWN": "ACTIVITY_UNCERTAIN",
        }[track.activity_state]
        return {"event": event_name, "subject_id": track.subject_id, "timestamp_seconds": float(timestamp_seconds), "method": track.activity_method}

    def _pause_posture(self, track, engine: TemporalRiskEngine, timestamp_seconds: float, reason: str) -> None:
        track.v4_result = {"status": "SITTING_POSTURE_MONITOR_PAUSED", "activity_state": track.activity_state}
        track.v4_prediction = None
        track.v4_probabilities = None
        track.geometry = {
            "valid": False,
            "status": "SITTING_POSTURE_MONITOR_PAUSED",
            "activity_state": track.activity_state,
            "method": track.activity_method,
        }
        if reason == "POSE_LOST":
            track.temporal_state = engine.update(timestamp_seconds, pose_visible=False, geometry=track.geometry)
        else:
            track.temporal_state = engine.pause(
                timestamp_seconds,
                reason=reason,
                activity_state=track.activity_state,
            )

    def _enrich_track(self, track, timestamp_seconds: float) -> dict[str, Any] | None:
        engine = self._engine(track.subject_id)
        activity_event = self._update_activity(track, timestamp_seconds)
        if track.activity_state != "SITTING":
            if track.tracking_state not in {"NEW", "ACTIVE"} or not track.core_valid:
                reason = "POSE_LOST"
            else:
                reason = "STANDING_POSTURE_MONITOR_PAUSED" if track.activity_state == "STANDING" else "ACTIVITY_UNKNOWN_POSTURE_MONITOR_PAUSED"
            self._pause_posture(track, engine, timestamp_seconds, reason)
            track.hierarchical_state = self._hierarchical_state(track)
            return activity_event
        if track.tracking_state in {"NEW", "ACTIVE"} and track.core_valid and track.world_landmarks is not None:
            try:
                prediction = self.v4_runner.predict_world_landmarks(
                    track.world_landmarks,
                    image_visibility=track.landmarks,
                    visibility_threshold=self.config.visibility_threshold,
                )
                raw_geometry = calculate_geometry(track.world_landmarks)
                baseline = self._neutral_baselines.get(track.subject_id)
                geometry = relative_geometry(raw_geometry, baseline)
                directional = evaluate_directional_geometry(
                    geometry,
                    thresholds=self.posture_config.indicator_thresholds,
                )
                geometry = dict(geometry)
                geometry["directional_indicators"] = directional
                indicators = evaluate_indicators(
                    geometry,
                    prediction.get("label"),
                    confidence=prediction.get("confidence"),
                    thresholds=self.posture_config.indicator_thresholds,
                )
                probabilities = prediction.get("probabilities", {})
                probability_vector = np.asarray([float(probabilities[label]) for label in self.labels], dtype=float)
                temporal = engine.update(
                    timestamp_seconds,
                    pose_visible=True,
                    probabilities=probability_vector,
                    geometry=geometry,
                    indicators=indicators,
                )
                track.v4_result = dict(prediction)
                track.v4_prediction = str(prediction.get("label"))
                track.v4_probabilities = {str(key): float(value) for key, value in probabilities.items()}
                track.geometry = geometry
                track.temporal_state = temporal
                track.hierarchical_state = self._hierarchical_state(track)
                return activity_event
            except (KeyError, TypeError, ValueError) as exc:
                track.v4_result = {"status": "ERROR", "detail": str(exc)}
                track.geometry = self._lost_geometry()
                track.temporal_state = engine.update(
                    timestamp_seconds,
                    pose_visible=False,
                    geometry=track.geometry,
                )
                track.hierarchical_state = self._hierarchical_state(track)
                return activity_event

        track.geometry = self._lost_geometry()
        track.temporal_state = engine.update(
            timestamp_seconds,
            pose_visible=False,
            geometry=track.geometry,
        )
        track.hierarchical_state = self._hierarchical_state(track)
        return activity_event

    def process_frame(
        self,
        detections: list[PoseDetection] | tuple[PoseDetection, ...],
        timestamp_seconds: float,
        *,
        camera_id: str | None = None,
    ) -> MultiPersonFrameResult:
        """Track and independently enrich every pose in one processed frame."""

        self.tracker.update(detections, timestamp_seconds)
        events = list(self.tracker.last_events)
        active_tracks = self.tracker.active_track_objects()
        for track in active_tracks:
            activity_event = self._enrich_track(track, float(timestamp_seconds))
            if activity_event is not None:
                events.append(activity_event)
        active_ids = {track.subject_id for track in active_tracks}
        self._risk_engines = {key: value for key, value in self._risk_engines.items() if key in active_ids}
        self._activity_gates = {key: value for key, value in self._activity_gates.items() if key in active_ids}
        self._neutral_baselines = {key: value for key, value in self._neutral_baselines.items() if key in active_ids}
        result = MultiPersonFrameResult(
            timestamp_seconds=float(timestamp_seconds),
            subjects=tuple(track.snapshot() for track in self.tracker.active_track_objects()),
            camera_id=camera_id or self.config.camera_id,
            tracking_status="MULTI_PERSON_SOFTWARE_PROTOTYPE",
            events=tuple(events),
        )
        self._last_frame = result
        return result
