"""Seat-aware assignment and per-seat posture state for fixed workstations.

This module is intentionally anonymous and camera-relative.  A seat is a
deployment context, not a person's identity.  Short-lived ``PERSON_XX``
tracks are used only to associate detections between nearby frames.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from src.posture_ai.ergonomics.calibration import relative_geometry
from src.posture_ai.ergonomics.geometry import calculate_geometry
from src.posture_ai.ergonomics.indicators import evaluate_directional_geometry, evaluate_indicators
from src.posture_ai.ergonomics.risk_engine import PostureEngineConfig, TemporalRiskEngine

from .schema import MultiPersonFrameResult, PoseSubject
from .tracking import AnonymousPoseTracker, PoseDetection


_SEAT_ID = re.compile(r"SEAT_[0-9]{2,4}\Z")
SEAT_STATES = frozenset({"EMPTY", "CANDIDATE", "OCCUPIED", "OCCLUDED", "LOST"})
INTERFERENCE_STATUSES = frozenset({
    "SEAT_OCCLUDED",
    "PASSERBY_DETECTED",
    "MULTIPLE_CANDIDATES",
    "INPUT_UNCERTAIN",
})


def _number(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalized_point(value: Iterable[float], name: str) -> tuple[float, float]:
    values = tuple(_number(item, name) for item in value)
    if len(values) != 2 or any(item < 0.0 or item > 1.0 for item in values):
        raise ValueError(f"{name} must contain two values in normalized [0, 1] coordinates")
    return values


def _normalized_polygon(value: Iterable[Iterable[float]]) -> tuple[tuple[float, float], ...]:
    polygon = tuple(_normalized_point(point, "polygon point") for point in value)
    if len(polygon) < 3:
        raise ValueError("polygon must contain at least three points")
    return polygon


def _rectangle_polygon(roi: tuple[float, float, float, float]) -> tuple[tuple[float, float], ...]:
    left, top, right, bottom = roi
    return ((left, top), (right, top), (right, bottom), (left, bottom))


def _polygon_centroid(polygon: tuple[tuple[float, float], ...]) -> tuple[float, float]:
    values = np.asarray(polygon, dtype=float)
    return float(np.mean(values[:, 0])), float(np.mean(values[:, 1]))


@dataclass
class SeatZone:
    """One fixed workstation zone in normalized image coordinates."""

    seat_id: str
    roi: tuple[float, float, float, float] | list[float] | None = None
    polygon: tuple[tuple[float, float], ...] | list[list[float]] | None = None
    anchor_x: float | None = None
    anchor_y: float | None = None
    expected_facing: str = "camera"
    enabled: bool = True
    assignment_state: str = "EMPTY"
    current_track_id: str | None = None
    last_valid_subject: str | None = None
    occupied_duration: float = 0.0

    def __post_init__(self) -> None:
        self.seat_id = str(self.seat_id)
        if not _SEAT_ID.fullmatch(self.seat_id):
            raise ValueError("seat_id must be an anonymous SEAT_XX identifier")
        if self.roi is None and self.polygon is None:
            raise ValueError("SeatZone requires roi or polygon")
        if self.roi is not None:
            values = tuple(_number(item, "roi") for item in self.roi)
            if len(values) != 4 or any(item < 0.0 or item > 1.0 for item in values):
                raise ValueError("roi must contain four normalized [0, 1] values")
            if values[2] < values[0] or values[3] < values[1]:
                raise ValueError("roi must be ordered left, top, right, bottom")
            self.roi = values
        if self.polygon is not None:
            self.polygon = _normalized_polygon(self.polygon)
        elif self.roi is not None:
            self.polygon = _rectangle_polygon(self.roi)
        if self.anchor_x is None or self.anchor_y is None:
            default_x, default_y = _polygon_centroid(self.polygon)
            self.anchor_x = default_x if self.anchor_x is None else _number(self.anchor_x, "anchor_x")
            self.anchor_y = default_y if self.anchor_y is None else _number(self.anchor_y, "anchor_y")
        else:
            self.anchor_x = _number(self.anchor_x, "anchor_x")
            self.anchor_y = _number(self.anchor_y, "anchor_y")
        if not 0.0 <= self.anchor_x <= 1.0 or not 0.0 <= self.anchor_y <= 1.0:
            raise ValueError("anchor must be normalized [0, 1] coordinates")
        self.expected_facing = str(self.expected_facing)
        self.enabled = bool(self.enabled)
        self.assignment_state = str(self.assignment_state)
        if self.assignment_state not in SEAT_STATES:
            raise ValueError(f"assignment_state must be one of {sorted(SEAT_STATES)}")
        self.occupied_duration = _number(self.occupied_duration, "occupied_duration")
        if self.occupied_duration < 0.0:
            raise ValueError("occupied_duration must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SeatZone":
        anchor = value.get("anchor")
        anchor_x = value.get("anchor_x")
        anchor_y = value.get("anchor_y")
        if anchor is not None:
            anchor_x, anchor_y = anchor
        return cls(
            seat_id=value.get("id", value.get("seat_id")),
            roi=value.get("roi"),
            polygon=value.get("polygon"),
            anchor_x=anchor_x,
            anchor_y=anchor_y,
            expected_facing=value.get("expected_facing", "camera"),
            enabled=value.get("enabled", True),
        )

    @property
    def effective_polygon(self) -> tuple[tuple[float, float], ...]:
        return tuple(self.polygon or ())

    def contains(self, point: tuple[float, float] | None) -> bool:
        if point is None:
            return False
        x, y = point
        polygon = self.effective_polygon
        inside = False
        previous_x, previous_y = polygon[-1]
        for current_x, current_y in polygon:
            if ((current_y > y) != (previous_y > y)):
                crossing_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
                if x < crossing_x:
                    inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside or any(
            min(previous_x, current_x) <= x <= max(previous_x, current_x)
            and min(previous_y, current_y) <= y <= max(previous_y, current_y)
            and abs((current_x - previous_x) * (y - previous_y) - (current_y - previous_y) * (x - previous_x)) < 1e-9
            for previous_x, previous_y, current_x, current_y in (
                (*polygon[index - 1], *polygon[index]) for index in range(len(polygon))
            )
        )

    def distance_to_anchor(self, point: tuple[float, float] | None) -> float:
        if point is None:
            return math.inf
        return float(math.hypot(point[0] - self.anchor_x, point[1] - self.anchor_y))

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat_id": self.seat_id,
            "roi": None if self.roi is None else list(self.roi),
            "polygon": [list(point) for point in self.effective_polygon],
            "anchor_x": self.anchor_x,
            "anchor_y": self.anchor_y,
            "expected_facing": self.expected_facing,
            "enabled": self.enabled,
            "assignment_state": self.assignment_state,
            "current_track_id": self.current_track_id,
            "last_valid_subject": self.last_valid_subject,
            "occupied_duration": self.occupied_duration,
        }


def load_seat_layout(path: str | Path) -> tuple[str, list[SeatZone]]:
    """Load a YAML layout without opening a camera or altering the source."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    seats = [SeatZone.from_mapping(item) for item in raw.get("seats", [])]
    if not seats:
        raise ValueError("seat layout must contain at least one seat")
    ids = [seat.seat_id for seat in seats]
    if len(ids) != len(set(ids)):
        raise ValueError("seat IDs must be unique")
    return str(raw.get("camera_id", "CAM01")), seats


def body_location(detection: PoseDetection) -> tuple[tuple[float, float] | None, str]:
    """Return hip midpoint first, then shoulder midpoint, then detector fallback."""

    for names, source in (
        (("left_hip", "right_hip"), "hip_midpoint"),
        (("left_shoulder", "right_shoulder"), "shoulder_midpoint"),
    ):
        points = [detection.landmarks.by_name(name) for name in names]
        if all(point is not None and point.present for point in points):
            values = np.asarray([[point.x, point.y] for point in points], dtype=float)
            if np.isfinite(values).all():
                return (float(np.mean(values[:, 0])), float(np.mean(values[:, 1]))), source
    if detection.center is not None:
        return tuple(float(item) for item in detection.center), "detection_center"
    if detection.bbox is not None:
        left, top, right, bottom = detection.bbox
        return ((left + right) / 2.0, (top + bottom) / 2.0), "bbox_center"
    return None, "unavailable"


@dataclass(frozen=True)
class SeatAssignment:
    seat_id: str
    detection_index: int
    track_id: str
    body_point: tuple[float, float]
    body_location_source: str
    cost: float
    in_roi: bool
    pose_quality: float
    core_valid: bool


@dataclass
class SeatAwareRuntimeConfig:
    max_people: int = 3
    max_anchor_distance: float = 0.20
    max_assignment_cost: float = 0.60
    seat_dwell_seconds: float = 1.5
    replacement_dwell_seconds: float = 1.5
    occlusion_grace_seconds: float = 1.0
    empty_after_seconds: float = 3.0
    continuity_window_seconds: float = 1.0
    continuity_distance: float = 0.12
    min_candidate_quality: float = 0.5
    camera_id: str = "CAM01"

    def __post_init__(self) -> None:
        if self.max_people < 1:
            raise ValueError("max_people must be at least 1")
        for name in ("max_anchor_distance", "max_assignment_cost", "seat_dwell_seconds", "replacement_dwell_seconds", "occlusion_grace_seconds", "empty_after_seconds", "continuity_window_seconds", "continuity_distance"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.empty_after_seconds < self.occlusion_grace_seconds:
            raise ValueError("empty_after_seconds must be >= occlusion_grace_seconds")
        if not 0.0 <= self.min_candidate_quality <= 1.0:
            raise ValueError("min_candidate_quality must be between 0 and 1")


@dataclass(frozen=True)
class SeatFrameState:
    seat_id: str
    assignment_state: str
    current_track_id: str | None
    last_valid_subject: str | None
    occupied_duration: float
    occupant_session_id: str | None
    interference_statuses: tuple[str, ...]
    candidate_track_id: str | None
    candidate_duration_seconds: float
    body_point: tuple[float, float] | None
    body_location_source: str | None
    pose_quality: float
    core_valid: bool
    v4_prediction: str | None
    v4_probabilities: Mapping[str, float]
    v4_result: Mapping[str, Any] | None
    geometry: Mapping[str, Any]
    temporal_state: Mapping[str, Any]
    expected_facing: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat_id": self.seat_id,
            "assignment_state": self.assignment_state,
            "current_track_id": self.current_track_id,
            "last_valid_subject": self.last_valid_subject,
            "occupied_duration": self.occupied_duration,
            "occupant_session_id": self.occupant_session_id,
            "interference_statuses": list(self.interference_statuses),
            "candidate_track_id": self.candidate_track_id,
            "candidate_duration_seconds": self.candidate_duration_seconds,
            "body_point": self.body_point,
            "body_location_source": self.body_location_source,
            "pose_quality": self.pose_quality,
            "core_valid": self.core_valid,
            "v4_prediction": self.v4_prediction,
            "v4_probabilities": dict(self.v4_probabilities),
            "v4_result": None if self.v4_result is None else dict(self.v4_result),
            "geometry": dict(self.geometry),
            "temporal_state": dict(self.temporal_state),
            "expected_facing": self.expected_facing,
        }


@dataclass(frozen=True)
class SeatAwareFrameResult:
    timestamp_seconds: float
    seats: tuple[SeatFrameState, ...]
    anonymous_subjects: tuple[PoseSubject, ...]
    camera_id: str
    interference_statuses: tuple[str, ...] = ()
    tracking_status: str = "SEAT_AWARE_SOFTWARE_PROTOTYPE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_seconds": self.timestamp_seconds,
            "camera_id": self.camera_id,
            "tracking_status": self.tracking_status,
            "tracking_mode": "seat",
            "seat_count": len(self.seats),
            "seats": [seat.to_dict() for seat in self.seats],
            "subjects": [subject.to_dict() for subject in self.anonymous_subjects],
            "interference_statuses": list(self.interference_statuses),
        }


@dataclass
class _SeatSlot:
    zone: SeatZone
    engine: TemporalRiskEngine
    baseline: dict[str, Any] | None = None
    candidate_track_id: str | None = None
    candidate_since: float | None = None
    replacement_track_id: str | None = None
    replacement_since: float | None = None
    last_body_point: tuple[float, float] | None = None
    last_valid_at: float | None = None
    occupied_since: float | None = None
    session_number: int = 0
    occupant_session_id: str | None = None
    v4_result: dict[str, Any] | None = None
    geometry: dict[str, Any] = field(default_factory=dict)
    temporal_state: dict[str, Any] = field(default_factory=dict)
    interference_statuses: set[str] = field(default_factory=set)
    body_point: tuple[float, float] | None = None
    body_location_source: str | None = None
    pose_quality: float = 0.0
    core_valid: bool = False


class SeatAwareRuntime:
    """Assign anonymous detections to fixed seats and run one engine per seat."""

    def __init__(
        self,
        v4_runner,
        seats: Iterable[SeatZone],
        *,
        runtime_config: SeatAwareRuntimeConfig | None = None,
        posture_config: PostureEngineConfig | None = None,
    ) -> None:
        self.config = runtime_config or SeatAwareRuntimeConfig()
        self.posture_config = posture_config or PostureEngineConfig()
        self.v4_runner = v4_runner
        self.labels = list(getattr(v4_runner, "labels", ["TLB", "TUP"]))
        if not self.labels:
            raise ValueError("V4 runner must expose at least one label")
        zones = list(seats)
        if not zones:
            raise ValueError("at least one seat is required")
        if len({zone.seat_id for zone in zones}) != len(zones):
            raise ValueError("seat IDs must be unique")
        self.zones = {zone.seat_id: zone for zone in zones}
        self._slots = {
            zone.seat_id: _SeatSlot(zone, TemporalRiskEngine(self.posture_config, labels=self.labels))
            for zone in zones
            if zone.enabled
        }
        self.tracker = AnonymousPoseTracker(
            max_people=self.config.max_people,
            max_association_distance=0.20,
            max_misses=3,
            reacquisition_window_seconds=max(
                2.0,
                self.config.occlusion_grace_seconds,
                self.config.seat_dwell_seconds,
                self.config.replacement_dwell_seconds,
            ),
            reuse_ids=False,
        )
        self._last_result: SeatAwareFrameResult | None = None

    @property
    def last_result(self) -> SeatAwareFrameResult | None:
        return self._last_result

    def reset(self) -> None:
        self.tracker.reset()
        for slot in self._slots.values():
            slot.engine.reset()
            slot.baseline = None
            slot.candidate_track_id = None
            slot.candidate_since = None
            slot.replacement_track_id = None
            slot.replacement_since = None
            slot.last_body_point = None
            slot.last_valid_at = None
            slot.occupied_since = None
            slot.session_number = 0
            slot.occupant_session_id = None
            slot.v4_result = None
            slot.geometry = {}
            slot.temporal_state = {}
            slot.interference_statuses.clear()
            slot.body_point = None
            slot.body_location_source = None
            slot.pose_quality = 0.0
            slot.core_valid = False
            slot.zone.assignment_state = "EMPTY"
            slot.zone.current_track_id = None
            slot.zone.last_valid_subject = None
            slot.zone.occupied_duration = 0.0
        self._last_result = None

    def set_neutral_baseline(self, seat_id: str, baseline: Mapping[str, Any]) -> None:
        slot = self._slots.get(str(seat_id))
        if slot is None:
            raise KeyError(f"Unknown seat: {seat_id}")
        slot.baseline = dict(baseline)

    def _eligible(self, zone: SeatZone, point: tuple[float, float] | None) -> tuple[bool, float]:
        if point is None:
            return False, math.inf
        distance = zone.distance_to_anchor(point)
        return zone.contains(point) or distance <= self.config.max_anchor_distance, distance

    def _assign(
        self,
        detections: list[PoseDetection],
        subject_by_detection: Mapping[int, PoseSubject],
    ) -> tuple[dict[str, SeatAssignment], dict[str, list[int]], list[tuple[tuple[float, float] | None, str]]]:
        enabled_zones = [slot.zone for slot in self._slots.values()]
        locations = [body_location(detection) for detection in detections]
        if not enabled_zones or not detections:
            return {}, {zone.seat_id: [] for zone in enabled_zones}, locations
        large = 1_000_000.0
        costs = np.full((len(enabled_zones), len(detections)), large, dtype=float)
        eligible_by_seat: dict[str, list[int]] = {zone.seat_id: [] for zone in enabled_zones}
        for row, zone in enumerate(enabled_zones):
            slot = self._slots[zone.seat_id]
            for column, (point, source) in enumerate(locations):
                eligible, distance = self._eligible(zone, point)
                if not eligible:
                    continue
                eligible_by_seat[zone.seat_id].append(column)
                subject = subject_by_detection.get(column)
                quality = 0.0 if subject is None else subject.pose_quality
                current_bonus = -0.25 if subject is not None and subject.subject_id == zone.current_track_id else 0.0
                previous_bonus = -0.10 if subject is not None and subject.subject_id == zone.last_valid_subject else 0.0
                quality_penalty = 0.10 if quality < self.config.min_candidate_quality else 0.0
                outside_penalty = 0.20 if not zone.contains(point) else 0.0
                costs[row, column] = distance + outside_penalty + quality_penalty + current_bonus + previous_bonus
        if costs.size == 0:
            return {}, eligible_by_seat, locations
        rows, columns = linear_sum_assignment(costs)
        assignments: dict[str, SeatAssignment] = {}
        for row, column in zip(rows, columns):
            cost = float(costs[row, column])
            if cost >= large or cost > self.config.max_assignment_cost:
                continue
            zone = enabled_zones[row]
            point, source = locations[column]
            if point is None:
                continue
            subject = subject_by_detection[column]
            detection = detections[column]
            assignments[zone.seat_id] = SeatAssignment(
                seat_id=zone.seat_id,
                detection_index=column,
                track_id=subject.subject_id,
                body_point=point,
                body_location_source=source,
                cost=cost,
                in_roi=zone.contains(point),
                pose_quality=detection.pose_quality,
                core_valid=detection.core_valid,
            )
        return assignments, eligible_by_seat, locations

    def _clear_candidate(self, slot: _SeatSlot) -> None:
        slot.candidate_track_id = None
        slot.candidate_since = None

    def _clear_replacement(self, slot: _SeatSlot) -> None:
        slot.replacement_track_id = None
        slot.replacement_since = None

    def _start_session(self, slot: _SeatSlot, assignment: SeatAssignment, timestamp: float, *, reset_history: bool) -> None:
        if reset_history:
            slot.engine.reset()
            slot.baseline = None
            slot.session_number += 1
            slot.occupant_session_id = f"SESSION_{slot.zone.seat_id}_{slot.session_number:03d}"
            slot.occupied_since = timestamp
        elif slot.occupied_since is None:
            slot.occupied_since = timestamp
        slot.zone.assignment_state = "OCCUPIED"
        slot.zone.current_track_id = assignment.track_id
        slot.zone.last_valid_subject = assignment.track_id
        slot.last_valid_at = timestamp
        slot.last_body_point = assignment.body_point
        slot.body_point = assignment.body_point
        slot.body_location_source = assignment.body_location_source
        slot.pose_quality = assignment.pose_quality
        slot.core_valid = assignment.core_valid
        self._clear_candidate(slot)
        self._clear_replacement(slot)

    def _make_empty(self, slot: _SeatSlot) -> None:
        slot.engine.reset()
        slot.baseline = None
        slot.zone.assignment_state = "EMPTY"
        slot.zone.current_track_id = None
        slot.zone.occupied_duration = 0.0
        slot.occupied_since = None
        slot.occupant_session_id = None
        slot.last_valid_at = None
        slot.last_body_point = None
        slot.body_point = None
        slot.body_location_source = None
        slot.pose_quality = 0.0
        slot.core_valid = False
        slot.v4_result = None
        slot.geometry = {}
        slot.temporal_state = {"risk_level": "NORMAL", "pose_visible": False, "status": "SEAT_EMPTY"}
        self._clear_candidate(slot)
        self._clear_replacement(slot)

    def _continuity_match(self, slot: _SeatSlot, assignment: SeatAssignment, timestamp: float) -> bool:
        if slot.last_body_point is None or slot.last_valid_at is None:
            return False
        if timestamp - slot.last_valid_at > self.config.continuity_window_seconds:
            return False
        distance = float(np.linalg.norm(np.asarray(slot.last_body_point) - np.asarray(assignment.body_point)))
        return distance <= self.config.continuity_distance and assignment.core_valid and assignment.pose_quality >= self.config.min_candidate_quality

    def _update_occupancy(
        self,
        slot: _SeatSlot,
        assignment: SeatAssignment | None,
        eligible_indices: list[int],
        detections: list[PoseDetection],
        subject_by_detection: Mapping[int, PoseSubject],
        timestamp: float,
    ) -> SeatAssignment | None:
        slot.interference_statuses.clear()
        if len(eligible_indices) > 1:
            slot.interference_statuses.add("MULTIPLE_CANDIDATES")
        if assignment is not None and (not assignment.core_valid or assignment.pose_quality < self.config.min_candidate_quality):
            slot.interference_statuses.add("INPUT_UNCERTAIN")
        zone = slot.zone
        state = zone.assignment_state
        current_id = zone.current_track_id
        current_assignment = assignment if assignment is not None and assignment.track_id == current_id else None
        if state == "EMPTY":
            if assignment is None or not assignment.core_valid or assignment.pose_quality < self.config.min_candidate_quality or len(eligible_indices) > 1:
                self._clear_candidate(slot)
                if assignment is not None:
                    slot.interference_statuses.add("PASSERBY_DETECTED")
                zone.assignment_state = "EMPTY"
                return None
            if slot.candidate_track_id != assignment.track_id:
                slot.candidate_track_id = assignment.track_id
                slot.candidate_since = timestamp
            zone.assignment_state = "CANDIDATE"
            candidate_since = timestamp if slot.candidate_since is None else slot.candidate_since
            if timestamp - candidate_since >= self.config.seat_dwell_seconds:
                self._start_session(slot, assignment, timestamp, reset_history=True)
                return assignment
            return None

        if state == "CANDIDATE":
            if assignment is None or not assignment.core_valid or assignment.pose_quality < self.config.min_candidate_quality or len(eligible_indices) > 1:
                self._clear_candidate(slot)
                zone.assignment_state = "EMPTY"
                if assignment is not None:
                    slot.interference_statuses.add("PASSERBY_DETECTED")
                return None
            if slot.candidate_track_id != assignment.track_id:
                slot.candidate_track_id = assignment.track_id
                slot.candidate_since = timestamp
            candidate_since = timestamp if slot.candidate_since is None else slot.candidate_since
            if timestamp - candidate_since >= self.config.seat_dwell_seconds:
                self._start_session(slot, assignment, timestamp, reset_history=True)
                return assignment
            return None

        if current_assignment is not None:
            zone.assignment_state = "OCCUPIED"
            slot.last_valid_at = timestamp
            slot.last_body_point = current_assignment.body_point
            slot.body_point = current_assignment.body_point
            slot.body_location_source = current_assignment.body_location_source
            slot.pose_quality = current_assignment.pose_quality
            slot.core_valid = current_assignment.core_valid
            if assignment is not None and len(eligible_indices) > 1:
                slot.interference_statuses.add("PASSERBY_DETECTED")
            return current_assignment

        gap = math.inf if slot.last_valid_at is None else max(0.0, timestamp - slot.last_valid_at)
        if assignment is None:
            if gap <= self.config.occlusion_grace_seconds:
                zone.assignment_state = "OCCLUDED"
                slot.interference_statuses.add("SEAT_OCCLUDED")
            elif gap <= self.config.empty_after_seconds:
                zone.assignment_state = "LOST"
                slot.interference_statuses.add("SEAT_OCCLUDED")
            else:
                self._make_empty(slot)
            return None

        if current_id is not None and assignment.track_id != current_id:
            if self._continuity_match(slot, assignment, timestamp) and len(eligible_indices) == 1:
                # A nearby replacement PERSON ID may be a tracker fragment.  Keep
                # the workstation session only when location/core evidence is
                # continuous; this is not biological identity recognition.
                self._start_session(slot, assignment, timestamp, reset_history=False)
                return assignment
            if gap <= self.config.occlusion_grace_seconds:
                zone.assignment_state = "OCCLUDED"
                slot.interference_statuses.add("SEAT_OCCLUDED")
            else:
                zone.assignment_state = "LOST"
            slot.interference_statuses.add("PASSERBY_DETECTED")
            if assignment.core_valid and assignment.pose_quality >= self.config.min_candidate_quality and len(eligible_indices) == 1:
                if slot.replacement_track_id != assignment.track_id:
                    slot.replacement_track_id = assignment.track_id
                    slot.replacement_since = timestamp
                elif slot.replacement_since is not None and timestamp - slot.replacement_since >= self.config.replacement_dwell_seconds:
                    self._start_session(slot, assignment, timestamp, reset_history=True)
                    return assignment
            else:
                self._clear_replacement(slot)
            return None

        return None

    def _enrich(self, slot: _SeatSlot, assignment: SeatAssignment | None, detections: list[PoseDetection], timestamp: float) -> None:
        if slot.zone.assignment_state not in {"OCCUPIED", "OCCLUDED", "LOST"}:
            slot.temporal_state = {
                "timestamp_seconds": timestamp,
                "risk_level": "INPUT_UNCERTAIN" if slot.zone.assignment_state == "CANDIDATE" else "NORMAL",
                "pose_visible": False,
                "duration_seconds": 0.0,
            }
            return
        detection = None if assignment is None else detections[assignment.detection_index]
        if detection is not None and assignment is not None and assignment.core_valid and detection.world_landmarks is not None:
            try:
                prediction = self.v4_runner.predict_world_landmarks(
                    detection.world_landmarks,
                    image_visibility=detection.landmarks,
                    visibility_threshold=self.posture_config.minimum_pose_visibility,
                )
                geometry = relative_geometry(calculate_geometry(detection.world_landmarks), slot.baseline)
                directional = evaluate_directional_geometry(geometry, thresholds=self.posture_config.indicator_thresholds)
                geometry = dict(geometry)
                geometry["directional_indicators"] = directional
                indicators = evaluate_indicators(
                    geometry,
                    prediction.get("label"),
                    confidence=prediction.get("confidence"),
                    thresholds=self.posture_config.indicator_thresholds,
                )
                probabilities = prediction.get("probabilities", {})
                vector = np.asarray([float(probabilities[label]) for label in self.labels], dtype=float)
                slot.temporal_state = slot.engine.update(timestamp, pose_visible=True, probabilities=vector, geometry=geometry, indicators=indicators)
                slot.v4_result = dict(prediction)
                slot.geometry = geometry
                return
            except (KeyError, TypeError, ValueError) as exc:
                slot.v4_result = {"status": "ERROR", "detail": str(exc)}
        slot.geometry = dict(slot.geometry or {})
        slot.geometry["status"] = "POSE_LOST_OR_CORE_INVALID"
        slot.temporal_state = slot.engine.update(timestamp, pose_visible=False, geometry=slot.geometry)

    def _snapshot(self, slot: _SeatSlot, timestamp: float) -> SeatFrameState:
        if slot.occupied_since is not None:
            slot.zone.occupied_duration = max(0.0, timestamp - slot.occupied_since)
        candidate_duration = 0.0 if slot.candidate_since is None else max(0.0, timestamp - slot.candidate_since)
        return SeatFrameState(
            seat_id=slot.zone.seat_id,
            assignment_state=slot.zone.assignment_state,
            current_track_id=slot.zone.current_track_id,
            last_valid_subject=slot.zone.last_valid_subject,
            occupied_duration=slot.zone.occupied_duration,
            occupant_session_id=slot.occupant_session_id,
            interference_statuses=tuple(sorted(slot.interference_statuses)),
            candidate_track_id=slot.candidate_track_id,
            candidate_duration_seconds=candidate_duration,
            body_point=slot.body_point,
            body_location_source=slot.body_location_source,
            pose_quality=slot.pose_quality,
            core_valid=slot.core_valid,
            v4_prediction=None if slot.v4_result is None else slot.v4_result.get("label"),
            v4_probabilities={} if slot.v4_result is None else slot.v4_result.get("probabilities", {}),
            v4_result=slot.v4_result,
            geometry=slot.geometry,
            temporal_state=slot.temporal_state,
            expected_facing=slot.zone.expected_facing,
        )

    def process_frame(self, detections: list[PoseDetection] | tuple[PoseDetection, ...], timestamp_seconds: float, *, camera_id: str | None = None) -> SeatAwareFrameResult:
        timestamp = float(timestamp_seconds)
        values = list(detections)[: self.config.max_people]
        anonymous_subjects = self.tracker.update(values, timestamp)
        anonymous_subjects_by_id = {subject.subject_id: subject for subject in anonymous_subjects}
        subject_by_detection = {
            index: anonymous_subjects_by_id[track_id]
            for index, track_id in self.tracker.last_assignments.items()
            if track_id in anonymous_subjects_by_id
        }
        assignments, eligible_by_seat, locations = self._assign(values, subject_by_detection)
        used_detection_indices = {assignment.detection_index for assignment in assignments.values()}
        global_statuses: set[str] = set()
        for index, (point, _source) in enumerate(locations):
            if index in used_detection_indices or point is None:
                continue
            if any(zone.contains(point) for zone in self.zones.values() if zone.enabled):
                global_statuses.add("PASSERBY_DETECTED")
        for seat_id, slot in self._slots.items():
            selected = self._update_occupancy(
                slot,
                assignments.get(seat_id),
                eligible_by_seat.get(seat_id, []),
                values,
                subject_by_detection,
                timestamp,
            )
            self._enrich(slot, selected, values, timestamp)
            global_statuses.update(slot.interference_statuses)
        result = SeatAwareFrameResult(
            timestamp_seconds=timestamp,
            seats=tuple(self._snapshot(slot, timestamp) for slot in self._slots.values()),
            anonymous_subjects=tuple(anonymous_subjects),
            camera_id=str(camera_id or self.config.camera_id),
            interference_statuses=tuple(sorted(global_statuses)),
        )
        self._last_result = result
        return result


def dashboard_seat_cards(result: SeatAwareFrameResult, *, include_debug_track_id: bool = False) -> list[dict[str, Any]]:
    """Create instructor-facing cards that prioritize seat context."""

    cards = []
    for seat in result.seats:
        directional = seat.geometry.get("directional_indicators", {})
        temporal = seat.temporal_state
        card = {
            "seat_id": seat.seat_id,
            "ai_posture": seat.v4_result.get("display_label", seat.v4_prediction) if seat.v4_result else None,
            "ai_label": seat.v4_prediction,
            "ai_probabilities": dict(seat.v4_probabilities),
            "geometry": {
                "forward": bool(directional.get("forward", False)),
                "left": bool(directional.get("left", False)),
                "right": bool(directional.get("right", False)),
                "shoulder_tilt": seat.geometry.get("shoulder_tilt_deg"),
            },
            "state": temporal.get("risk_level", "INPUT_UNCERTAIN"),
            "seat_state": seat.assignment_state,
            "duration_seconds": temporal.get("duration_seconds", 0.0),
            "occupied_duration": seat.occupied_duration,
            "pose_quality": seat.pose_quality,
            "interference_statuses": list(seat.interference_statuses),
        }
        if include_debug_track_id:
            card["debug_track_id"] = seat.current_track_id
        cards.append(card)
    return cards


def draw_seat_layout(frame, zones: Iterable[SeatZone], *, result: SeatAwareFrameResult | None = None, include_debug_track_id: bool = False):
    """Draw seat ROIs/cards on a supplied image; never opens a camera/window."""

    output = frame.copy()
    height, width = output.shape[:2]
    state_by_id = {seat.seat_id: seat for seat in (result.seats if result is not None else ())}
    for zone in zones:
        polygon = np.asarray([(int(x * width), int(y * height)) for x, y in zone.effective_polygon], dtype=np.int32)
        seat = state_by_id.get(zone.seat_id)
        state = zone.assignment_state if seat is None else seat.assignment_state
        color = (90, 190, 90) if state == "OCCUPIED" else (0, 190, 255) if state in {"CANDIDATE", "OCCLUDED"} else (150, 150, 150)
        cv2.polylines(output, [polygon], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        anchor = (int(zone.anchor_x * width), int(zone.anchor_y * height))
        cv2.circle(output, anchor, 5, color, -1, cv2.LINE_AA)
        lines = [f"{zone.seat_id} | {state}"]
        if seat is not None:
            geometry = seat.geometry.get("directional_indicators", {})
            ai = seat.v4_result.get("display_label", seat.v4_prediction) if seat.v4_result else "—"
            lines.append(f"AI: {ai}  F:{'Y' if geometry.get('forward') else 'N'}")
            lines.append(f"STATE: {seat.temporal_state.get('risk_level', 'INPUT_UNCERTAIN')}")
            if seat.interference_statuses:
                lines.append("STATUS: " + ",".join(seat.interference_statuses))
            if include_debug_track_id and seat.current_track_id:
                lines.append(f"DEBUG: {seat.current_track_id}")
        x, y = int(polygon[:, 0].min()), max(18, int(polygon[:, 1].min()) - 8)
        for offset, line in enumerate(lines):
            baseline = y + offset * 18
            cv2.putText(output, line, (x + 1, baseline + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(output, line, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return output
