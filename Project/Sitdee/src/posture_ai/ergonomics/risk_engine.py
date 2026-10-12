"""Configurable temporal posture-risk state machine.

The durations in this module are product/engineering settings. They are not
medical recommendations or injury thresholds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .activity_gate import ActivityGateConfig
from .indicators import IndicatorThresholds


POOR_DIRECTIONS = frozenset({"TLF", "TLB", "TLL", "TLR"})


@dataclass(frozen=True)
class PostureEngineConfig:
    minimum_warning_duration: float = 15.0
    high_risk_duration: float = 60.0
    smoothing_window: int = 5
    minimum_pose_visibility: float = 0.5
    recovery_duration: float = 3.0
    warning_cooldown: float = 15.0
    minimum_confidence: float = 0.5
    frame_interval_seconds: float = 1.0 / 30.0
    minimum_persistence_duration: float = 0.5
    indicator_thresholds: IndicatorThresholds = field(default_factory=IndicatorThresholds)
    activity_gate: ActivityGateConfig = field(default_factory=ActivityGateConfig)

    def __post_init__(self) -> None:
        if self.minimum_warning_duration < 0 or self.high_risk_duration < self.minimum_warning_duration:
            raise ValueError("risk durations must be non-negative and high_risk_duration >= minimum_warning_duration")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be positive")
        for name in ("minimum_pose_visibility", "minimum_confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.recovery_duration < 0 or self.warning_cooldown < 0 or self.frame_interval_seconds <= 0 or self.minimum_persistence_duration < 0:
            raise ValueError("recovery/cooldown must be non-negative and frame interval positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PostureEngineConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        indicator = raw.get("indicator_thresholds", {})
        return cls(
            minimum_warning_duration=float(raw.get("minimum_warning_duration", cls.minimum_warning_duration)),
            high_risk_duration=float(raw.get("high_risk_duration", cls.high_risk_duration)),
            smoothing_window=int(raw.get("smoothing_window", cls.smoothing_window)),
            minimum_pose_visibility=float(raw.get("minimum_pose_visibility", cls.minimum_pose_visibility)),
            recovery_duration=float(raw.get("recovery_duration", cls.recovery_duration)),
            warning_cooldown=float(raw.get("warning_cooldown", cls.warning_cooldown)),
            minimum_confidence=float(raw.get("minimum_confidence", cls.minimum_confidence)),
            frame_interval_seconds=float(raw.get("frame_interval_seconds", cls.frame_interval_seconds)),
            minimum_persistence_duration=float(raw.get("minimum_persistence_duration", cls.minimum_persistence_duration)),
            indicator_thresholds=IndicatorThresholds(
                shoulder_tilt_deg=float(indicator.get("shoulder_tilt_deg", IndicatorThresholds.shoulder_tilt_deg)),
                head_forward_proxy=float(indicator.get("head_forward_proxy", IndicatorThresholds.head_forward_proxy)),
                slouch_trunk_angle_deg=float(indicator.get("slouch_trunk_angle_deg", IndicatorThresholds.slouch_trunk_angle_deg)),
                lateral_lean=float(indicator.get("lateral_lean", IndicatorThresholds.lateral_lean)),
            ),
            activity_gate=ActivityGateConfig.from_mapping(raw.get("activity_gate")),
        )


class TemporalRiskEngine:
    """Turn per-frame predictions into persistent, explainable risk states."""

    def __init__(self, config: PostureEngineConfig | None = None, labels: list[str] | tuple[str, ...] | None = None):
        self.config = config or PostureEngineConfig()
        self.labels = list(labels or ["TLB", "TLF", "TLL", "TLR", "TUP"])
        if not self.labels:
            raise ValueError("at least one model label is required")
        self._history: deque[np.ndarray] = deque(maxlen=self.config.smoothing_window)
        self._direction_history: deque[str] = deque(maxlen=self.config.smoothing_window)
        self.reset()

    def reset(self) -> None:
        self._last_timestamp: float | None = None
        self._stable_direction: str | None = None
        self._candidate_direction: str | None = None
        self._candidate_duration = 0.0
        self._continuous_duration = 0.0
        self._recovery_elapsed = 0.0
        self._last_risk_level = "NORMAL"
        self._cooldown_until = float("-inf")
        self._pose_lost = False
        self._history.clear()
        self._direction_history.clear()

    @property
    def continuous_duration(self) -> float:
        return self._continuous_duration

    def _smoothed_probabilities(self, probabilities) -> tuple[np.ndarray, str, float, str]:
        values = np.asarray(probabilities, dtype=float)
        if values.shape != (len(self.labels),):
            raise ValueError(f"Expected probabilities shape {(len(self.labels),)}, got {values.shape}")
        if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
            raise ValueError("probabilities must be finite, non-negative, and have a positive sum")
        values = values / values.sum()
        self._history.append(values)
        smoothed = np.mean(np.vstack(self._history), axis=0)
        index = int(np.argmax(smoothed))
        instantaneous = self.labels[index]
        self._direction_history.append(instantaneous)
        counts = {label: self._direction_history.count(label) for label in self.labels}
        majority = max(self.labels, key=lambda label: (counts[label], -self.labels.index(label)))
        return smoothed, instantaneous, float(smoothed[index]), majority

    def update(
        self,
        timestamp_seconds: float,
        *,
        pose_visible: bool,
        probabilities=None,
        direction: str | None = None,
        confidence: float | None = None,
        geometry: dict | None = None,
        indicators: list[dict] | None = None,
    ) -> dict:
        timestamp = float(timestamp_seconds)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_seconds must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        delta = 0.0 if self._last_timestamp is None else timestamp - self._last_timestamp
        self._last_timestamp = timestamp
        geometry = geometry or {}
        indicators = indicators or []
        indicator_names = [str(item.get("name")) for item in indicators if item.get("name")]

        if not pose_visible:
            self._pose_lost = True
            self._continuous_duration = 0.0
            self._recovery_elapsed = 0.0
            self._candidate_direction = None
            self._candidate_duration = 0.0
            self._last_risk_level = "POSE_LOST"
            return {
                "timestamp_seconds": timestamp,
                "direction": self._stable_direction,
                "majority_direction": self._stable_direction,
                "direction_confidence": confidence,
                "probabilities": {},
                "geometry": geometry,
                "risk_indicators": indicator_names,
                "indicator_details": indicators,
                "continuous_duration": self._continuous_duration,
                "duration_seconds": self._continuous_duration,
                "risk_level": "POSE_LOST",
                "pose_visible": False,
                "warning_event": False,
                "recovery_active": False,
            }

        smoothed = None
        majority_direction = None
        if probabilities is not None:
            smoothed, direction_from_model, smoothed_confidence, majority_direction = self._smoothed_probabilities(probabilities)
            direction = majority_direction
            confidence = smoothed_confidence
        if direction is None:
            raise ValueError("direction or probabilities is required when pose_visible=True")
        confidence = None if confidence is None else float(confidence)
        direction = str(direction)
        previous_direction = self._stable_direction
        if self._candidate_direction == direction:
            self._candidate_duration += delta
        else:
            self._candidate_direction = direction
            self._candidate_duration = delta
        if self._stable_direction is None or direction == self._stable_direction or self._candidate_duration >= self.config.minimum_persistence_duration:
            self._stable_direction = direction
        self._pose_lost = False

        previous_was_poor = previous_direction in POOR_DIRECTIONS
        current_is_poor = self._stable_direction in POOR_DIRECTIONS
        recovery_active = False
        if current_is_poor:
            if previous_was_poor and previous_direction == self._stable_direction:
                self._continuous_duration += delta
            else:
                self._continuous_duration = 0.0
            self._recovery_elapsed = 0.0
        else:
            if previous_was_poor and self._continuous_duration > 0 and self._recovery_elapsed < self.config.recovery_duration:
                self._recovery_elapsed += delta
                recovery_active = self._recovery_elapsed < self.config.recovery_duration
                if not recovery_active:
                    self._continuous_duration = 0.0
                    self._recovery_elapsed = 0.0
            else:
                self._continuous_duration = 0.0
                self._recovery_elapsed = 0.0

        if current_is_poor:
            if self._continuous_duration >= self.config.high_risk_duration:
                risk_level = "HIGH_DURATION_RISK"
            elif self._continuous_duration >= self.config.minimum_warning_duration:
                risk_level = "WARNING"
            else:
                risk_level = "OBSERVE"
        elif recovery_active or any(item.get("kind") == "EXPERIMENTAL" for item in indicators):
            risk_level = "OBSERVE"
        else:
            risk_level = "NORMAL"

        warning_event = False
        entered_warning = risk_level in {"WARNING", "HIGH_DURATION_RISK"} and self._last_risk_level not in {"WARNING", "HIGH_DURATION_RISK"}
        if entered_warning and timestamp >= self._cooldown_until:
            warning_event = True
            self._cooldown_until = timestamp + self.config.warning_cooldown
        self._last_risk_level = risk_level
        probabilities_dict = {} if smoothed is None else {label: float(smoothed[index]) for index, label in enumerate(self.labels)}
        return {
            "timestamp_seconds": timestamp,
            "direction": self._stable_direction,
            "majority_direction": majority_direction or self._stable_direction,
            "direction_confidence": confidence,
            "probabilities": probabilities_dict,
            "geometry": geometry,
            "risk_indicators": indicator_names,
            "indicator_details": indicators,
            "continuous_duration": self._continuous_duration,
            "duration_seconds": self._continuous_duration,
            "risk_level": risk_level,
            "pose_visible": True,
            "warning_event": warning_event,
            "recovery_active": recovery_active,
            "candidate_persistence_seconds": self._candidate_duration,
        }

    def pause(self, timestamp_seconds: float, *, reason: str, activity_state: str) -> dict:
        """Pause sitting-risk accumulation without converting it to pose loss.

        The next active sitting update starts with zero elapsed delta, so an
        UNKNOWN/STANDING interval cannot silently add to poor-posture time.
        """

        timestamp = float(timestamp_seconds)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        self._last_timestamp = None
        self._candidate_direction = None
        self._candidate_duration = 0.0
        self._last_risk_level = "MONITOR_PAUSED"
        return {
            "timestamp_seconds": timestamp,
            "direction": self._stable_direction,
            "majority_direction": self._stable_direction,
            "direction_confidence": None,
            "probabilities": {},
            "geometry": {},
            "risk_indicators": [],
            "indicator_details": [],
            "continuous_duration": self._continuous_duration,
            "duration_seconds": self._continuous_duration,
            "risk_level": "MONITOR_PAUSED",
            "pose_visible": True,
            "monitoring_paused": True,
            "pause_reason": str(reason),
            "activity_state": str(activity_state),
            "warning_event": False,
            "recovery_active": False,
        }
