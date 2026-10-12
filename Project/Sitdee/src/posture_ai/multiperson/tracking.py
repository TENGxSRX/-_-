"""Anonymous geometry-based multi-person tracking for generic runtime use.

The tracker is deliberately small and understandable.  It associates pose
centres between adjacent frames; it never uses faces, names, embeddings, or
any attempt at real-world identity persistence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.posture_ai.pose.landmarks import PoseLandmarks
from src.posture_ai.pose.robustness import pose_quality_score
from src.posture_ai.pose.validation import validate_pose_landmarks

from .schema import PoseSubject, anonymous_subject_id


CORE_LANDMARK_NAMES = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")
TRACKING_STATES = frozenset({"NEW", "ACTIVE", "TEMPORARILY_LOST", "EXPIRED"})


def core_quality_summary(detection: "PoseDetection", *, threshold: float = 0.5) -> dict[str, Any]:
    """Return CCTV-relevant quality flags without conflating them with AI accuracy."""

    values: dict[str, float | None] = {}
    for name in CORE_LANDMARK_NAMES:
        point = detection.landmarks.by_name(name)
        visibility = None if point is None or point.visibility is None else float(point.visibility)
        values[name] = visibility

    def visible(names: tuple[str, ...]) -> bool:
        return all(values.get(name) is not None and float(values[name]) >= float(threshold) for name in names)

    return {
        "pose_quality_score": float(detection.pose_quality),
        "core_visibility": values,
        "HEAD_VISIBLE": visible(("nose",)),
        "SHOULDERS_VISIBLE": visible(("left_shoulder", "right_shoulder")),
        "HIPS_VISIBLE": visible(("left_hip", "right_hip")),
        "CORE_OK": bool(detection.core_valid),
        "CORE_PARTIAL": not bool(detection.core_valid),
        "detector_confidence": "NOT_EXPOSED_BY_CURRENT_MEDIAPIPE_ADAPTER",
    }


def _finite_xy(pose: PoseLandmarks) -> np.ndarray:
    points = []
    for point in pose.points:
        if point is None or not point.present:
            continue
        values = np.asarray([point.x, point.y], dtype=float)
        if np.isfinite(values).all():
            points.append(values)
    return np.vstack(points) if points else np.empty((0, 2), dtype=float)


def _bbox_and_center(pose: PoseLandmarks) -> tuple[tuple[float, float, float, float] | None, tuple[float, float] | None]:
    xy = _finite_xy(pose)
    if len(xy) == 0:
        return None, None
    left, top = np.min(xy, axis=0)
    right, bottom = np.max(xy, axis=0)
    bbox = (float(left), float(top), float(right), float(bottom))
    core_points = []
    for name in CORE_LANDMARK_NAMES[1:]:
        point = pose.by_name(name)
        if point is not None and point.present and np.isfinite([point.x, point.y]).all():
            core_points.append([point.x, point.y])
    reference = np.asarray(core_points, dtype=float) if core_points else np.asarray([[(left + right) / 2.0, (top + bottom) / 2.0]])
    center = tuple(float(value) for value in np.mean(reference, axis=0))
    return bbox, center


def _core_visibility_map(pose: PoseLandmarks) -> dict[str, float | None]:
    return {
        name: None if (point := pose.by_name(name)) is None or point.visibility is None else float(point.visibility)
        for name in CORE_LANDMARK_NAMES
    }


def _validate_center(center: tuple[float, float] | list[float] | None) -> tuple[float, float] | None:
    if center is None:
        return None
    if len(center) != 2:
        raise ValueError("center must contain two values")
    result = (float(center[0]), float(center[1]))
    if not np.isfinite(result).all():
        raise ValueError("center must contain finite values")
    return result


@dataclass(frozen=True)
class PoseDetection:
    """One detector output before anonymous association."""

    landmarks: PoseLandmarks
    world_landmarks: PoseLandmarks | None = None
    pose_quality: float = 0.0
    core_valid: bool = False
    bbox: tuple[float, float, float, float] | None = None
    center: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        pose = PoseLandmarks.from_any(self.landmarks)
        quality = float(self.pose_quality)
        if not np.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("pose_quality must be between 0 and 1")
        derived_bbox, derived_center = _bbox_and_center(pose)
        center = _validate_center(self.center if self.center is not None else derived_center)
        bbox = self.bbox if self.bbox is not None else derived_bbox
        if bbox is not None:
            if len(bbox) != 4 or not np.isfinite(bbox).all():
                raise ValueError("bbox must contain four finite values")
            bbox = tuple(float(value) for value in bbox)
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                raise ValueError("bbox must be ordered left, top, right, bottom")
        object.__setattr__(self, "landmarks", pose)
        object.__setattr__(self, "world_landmarks", None if self.world_landmarks is None else PoseLandmarks.from_any(self.world_landmarks))
        object.__setattr__(self, "pose_quality", quality)
        object.__setattr__(self, "core_valid", bool(self.core_valid))
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "center", center)

    @classmethod
    def from_mediapipe(cls, image_landmarks: Any, world_landmarks: Any | None = None, *, visibility_threshold: float = 0.5) -> "PoseDetection":
        image_pose = PoseLandmarks.from_any(image_landmarks)
        world_pose = None if world_landmarks is None else PoseLandmarks.from_any(world_landmarks)
        quality = pose_quality_score(image_pose, names=CORE_LANDMARK_NAMES, coverage_threshold=visibility_threshold)
        validation = validate_pose_landmarks(
            image_pose,
            required_names=CORE_LANDMARK_NAMES,
            visibility_threshold=visibility_threshold,
        )
        return cls(
            landmarks=image_pose,
            world_landmarks=world_pose,
            pose_quality=float(quality["score"]),
            core_valid=bool(validation.valid),
        )


@dataclass
class _SubjectTrack:
    subject_id: str
    landmarks: PoseLandmarks
    world_landmarks: PoseLandmarks | None
    pose_quality: float
    core_valid: bool
    bbox: tuple[float, float, float, float] | None
    center: tuple[float, float] | None
    tracking_state: str
    last_seen_at: float
    miss_count: int = 0
    frames_seen: int = 1
    v4_result: dict[str, Any] | None = None
    geometry: dict[str, Any] = field(default_factory=dict)
    temporal_state: dict[str, Any] = field(default_factory=dict)
    v4_prediction: str | None = None
    v4_probabilities: dict[str, float] | None = None
    core_visibility: dict[str, float | None] = field(default_factory=dict)
    hierarchical_state: dict[str, Any] = field(default_factory=dict)
    activity_state: str = "UNKNOWN"
    activity_raw_state: str = "UNKNOWN"
    activity_confidence: float = 0.0
    activity_method: str = "NOT_EVALUATED"
    activity_evidence: dict[str, Any] = field(default_factory=dict)
    velocity: tuple[float, float] | None = None

    def update(self, detection: PoseDetection, timestamp_seconds: float) -> None:
        previous_center = self.center
        elapsed = float(timestamp_seconds) - float(self.last_seen_at)
        if previous_center is not None and detection.center is not None and elapsed > 1e-6:
            velocity = (np.asarray(detection.center) - np.asarray(previous_center)) / elapsed
            speed = float(np.linalg.norm(velocity))
            if speed > 2.0:
                velocity = velocity * (2.0 / speed)
            self.velocity = (float(velocity[0]), float(velocity[1]))
        self.landmarks = detection.landmarks
        self.world_landmarks = detection.world_landmarks
        self.pose_quality = detection.pose_quality
        self.core_valid = detection.core_valid
        self.core_visibility = _core_visibility_map(detection.landmarks)
        self.bbox = detection.bbox
        self.center = detection.center
        self.tracking_state = "ACTIVE"
        self.last_seen_at = float(timestamp_seconds)
        self.miss_count = 0
        self.frames_seen += 1

    def mark_missed(self, timestamp_seconds: float, *, max_misses: int, reacquisition_window_seconds: float) -> None:
        self.miss_count += 1
        elapsed = float(timestamp_seconds) - self.last_seen_at
        if self.miss_count > max_misses or elapsed > reacquisition_window_seconds:
            self.tracking_state = "EXPIRED"
            self.core_valid = False
            self.pose_quality = 0.0
        else:
            self.tracking_state = "TEMPORARILY_LOST"
            self.core_valid = False
            self.pose_quality = 0.0
        self.core_visibility = {name: None for name in CORE_LANDMARK_NAMES}
        self.activity_state = "UNKNOWN"
        self.activity_raw_state = "UNKNOWN"
        self.activity_confidence = 0.0
        self.activity_method = "GEOMETRY_EXPERIMENTAL"
        self.activity_evidence = {"state": "UNKNOWN", "reason": "POSE_LOST", "method": "GEOMETRY_EXPERIMENTAL"}
        self.hierarchical_state = {"state": "UNCERTAIN", "reason": "POSE_LOST"}

    def snapshot(self) -> PoseSubject:
        return PoseSubject(
            subject_id=self.subject_id,
            landmarks=self.landmarks,
            pose_quality=self.pose_quality,
            v4_result=self.v4_result,
            geometry=self.geometry,
            temporal_state=self.temporal_state,
            tracking_state=self.tracking_state,
            world_landmarks=self.world_landmarks,
            bbox=self.bbox,
            center=self.center,
            core_valid=self.core_valid,
            v4_prediction=self.v4_prediction,
            v4_probabilities=self.v4_probabilities,
            last_seen_at=self.last_seen_at,
            core_visibility=self.core_visibility,
            hierarchical_state=self.hierarchical_state,
            activity_state=self.activity_state,
            activity_raw_state=self.activity_raw_state,
            activity_confidence=self.activity_confidence,
            activity_method=self.activity_method,
            activity_evidence=self.activity_evidence,
        )


class AnonymousPoseTracker:
    """Associate detections using normalized centre distance and lifecycle rules."""

    def __init__(
        self,
        *,
        max_people: int = 3,
        max_association_distance: float = 0.20,
        max_misses: int = 3,
        reacquisition_window_seconds: float = 1.0,
        reuse_ids: bool = True,
    ) -> None:
        if max_people < 1:
            raise ValueError("max_people must be at least 1")
        if max_association_distance <= 0.0 or reacquisition_window_seconds <= 0.0:
            raise ValueError("association distance and reacquisition window must be positive")
        if max_misses < 0:
            raise ValueError("max_misses must be non-negative")
        self.max_people = int(max_people)
        self.max_association_distance = float(max_association_distance)
        self.max_misses = int(max_misses)
        self.reacquisition_window_seconds = float(reacquisition_window_seconds)
        self.reuse_ids = bool(reuse_ids)
        self._tracks: dict[str, _SubjectTrack] = {}
        self._next_number = 1
        self._last_timestamp: float | None = None
        self.last_assignments: dict[int, str] = {}
        self.overflow_count = 0
        self.last_detected_pose_count = 0
        self.last_dropped_due_to_capacity = 0
        self.last_events: list[dict[str, Any]] = []
        self._retired_states: dict[str, str] = {}

    @property
    def expired_subject_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._retired_states))

    def track_state(self, subject_id: str) -> str | None:
        track = self._tracks.get(str(subject_id))
        if track is not None:
            return track.tracking_state
        return self._retired_states.get(str(subject_id))

    def active_track_objects(self) -> tuple[_SubjectTrack, ...]:
        """Return current non-expired tracks for the runtime enrichment layer."""

        return tuple(
            track for track in sorted(self._tracks.values(), key=lambda item: item.subject_id)
            if track.tracking_state != "EXPIRED"
        )

    def reset(self) -> None:
        self._tracks.clear()
        self._next_number = 1
        self._last_timestamp = None
        self.last_assignments = {}
        self.overflow_count = 0
        self.last_detected_pose_count = 0
        self.last_dropped_due_to_capacity = 0
        self.last_events = []
        self._retired_states.clear()

    def _new_track(self, detection: PoseDetection, timestamp_seconds: float) -> _SubjectTrack | None:
        """Allocate a released anonymous slot, or return ``None`` at capacity.

        Generic tracking reuses PERSON_01...PERSON_NN slots.  A missing slot
        is a normal CCTV condition when extra pose detections arrive, never an
        exception path.  The optional non-reuse mode remains available for the
        isolated seat-aware prototype, whose caller already bounds inputs.
        """

        if self.reuse_ids:
            used_numbers = {
                int(subject_id.split("_", 1)[1])
                for subject_id in self._tracks
                if subject_id.startswith("PERSON_") and subject_id.split("_", 1)[1].isdigit()
            }
            number = next((index for index in range(1, self.max_people + 1) if index not in used_numbers), None)
            if number is None:
                return None
            subject_id = anonymous_subject_id(number - 1)
        else:
            if len(self._tracks) >= self.max_people:
                return None
            subject_id = anonymous_subject_id(self._next_number - 1)
            self._next_number += 1
        self._retired_states.pop(subject_id, None)
        track = _SubjectTrack(
            subject_id=subject_id,
            landmarks=detection.landmarks,
            world_landmarks=detection.world_landmarks,
            pose_quality=detection.pose_quality,
            core_valid=detection.core_valid,
            bbox=detection.bbox,
            center=detection.center,
            tracking_state="NEW",
            last_seen_at=float(timestamp_seconds),
            core_visibility=_core_visibility_map(detection.landmarks),
            hierarchical_state={"state": "UNCERTAIN", "reason": "ACTIVITY_MODEL_NOT_TRAINED"},
        )
        self._tracks[subject_id] = track
        return track

    def _release_expired_tracks(self) -> None:
        """Remove only tracks already expired by normal lifecycle rules."""

        expired_ids = [subject_id for subject_id, track in self._tracks.items() if track.tracking_state == "EXPIRED"]
        for subject_id in expired_ids:
            self._retired_states[subject_id] = "EXPIRED"
            del self._tracks[subject_id]
        if len(self._retired_states) > self.max_people * 4:
            for subject_id in sorted(self._retired_states)[:-self.max_people * 4]:
                del self._retired_states[subject_id]

    def _record_capacity_drop(self, detection_index: int, timestamp_seconds: float) -> None:
        """Expose an expected over-capacity detection without calling it a failure."""

        self.last_dropped_due_to_capacity += 1
        self.overflow_count = self.last_dropped_due_to_capacity
        self.last_events.append({
            "event": "DETECTION_DROPPED_MAX_PEOPLE",
            "timestamp_seconds": float(timestamp_seconds),
            "detection_index": int(detection_index),
            "max_people": int(self.max_people),
            "detected_pose_count": int(self.last_detected_pose_count),
            "active_track_count": int(len(self._tracks)),
            "reason": "TRACK_CAPACITY_REACHED",
        })

    def _distance(self, track: _SubjectTrack, detection: PoseDetection) -> float:
        if track.center is None or detection.center is None:
            return math.inf
        predicted = np.asarray(track.center, dtype=float)
        if track.velocity is not None:
            elapsed = max(0.0, float(self._last_timestamp or track.last_seen_at) - float(track.last_seen_at))
            predicted = predicted + np.asarray(track.velocity, dtype=float) * elapsed
        distance = float(np.linalg.norm(predicted - np.asarray(detection.center, dtype=float)))
        return distance

    def update(self, detections: list[PoseDetection] | tuple[PoseDetection, ...], timestamp_seconds: float) -> tuple[PoseSubject, ...]:
        timestamp = float(timestamp_seconds)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("timestamp_seconds must be finite and non-negative")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        self._last_timestamp = timestamp
        self.last_events = []
        values = list(detections)
        if any(not isinstance(item, PoseDetection) for item in values):
            raise TypeError("detections must contain PoseDetection values")
        self.last_detected_pose_count = len(values)
        self.last_dropped_due_to_capacity = 0
        self.overflow_count = 0

        candidates = [
            track for track in self._tracks.values()
            if track.tracking_state != "EXPIRED"
            and timestamp - track.last_seen_at <= self.reacquisition_window_seconds
        ]
        pairs = []
        for detection_index, detection in enumerate(values):
            for track in candidates:
                pairs.append((self._distance(track, detection), track.subject_id, detection_index))
        pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        matched_tracks: set[str] = set()
        matched_detections: set[int] = set()
        self.last_assignments = {}
        for distance, subject_id, detection_index in pairs:
            if distance > self.max_association_distance:
                break
            if subject_id in matched_tracks or detection_index in matched_detections:
                continue
            track = self._tracks[subject_id]
            was_lost = track.tracking_state == "TEMPORARILY_LOST"
            was_new = track.tracking_state == "NEW"
            track.update(values[detection_index], timestamp)
            matched_tracks.add(subject_id)
            matched_detections.add(detection_index)
            self.last_assignments[detection_index] = subject_id
            self.last_events.append({
                "event": "PERSON_REACQUIRED" if was_lost else ("PERSON_ACTIVE" if not was_new else "PERSON_ACTIVE"),
                "subject_id": subject_id,
                "timestamp_seconds": timestamp,
            })

        for subject_id, track in self._tracks.items():
            if subject_id not in matched_tracks and track.tracking_state != "EXPIRED":
                previous_state = track.tracking_state
                track.mark_missed(
                    timestamp,
                    max_misses=self.max_misses,
                    reacquisition_window_seconds=self.reacquisition_window_seconds,
                )
                if track.tracking_state == "TEMPORARILY_LOST" and previous_state != "TEMPORARILY_LOST":
                    self.last_events.append({"event": "PERSON_TEMPORARILY_LOST", "subject_id": subject_id, "timestamp_seconds": timestamp})
                elif track.tracking_state == "EXPIRED":
                    self.last_events.append({"event": "PERSON_EXITED", "subject_id": subject_id, "timestamp_seconds": timestamp})

        # Normal lifecycle expiry happens before considering a new person.
        # TEMPORARILY_LOST tracks remain reserved through their reacquisition
        # grace period, and healthy ACTIVE tracks are never evicted here.
        self._release_expired_tracks()
        for detection_index, detection in enumerate(values):
            if detection_index not in matched_detections:
                if len(self._tracks) >= self.max_people:
                    self._record_capacity_drop(detection_index, timestamp)
                    continue
                track = self._new_track(detection, timestamp)
                if track is None:
                    self._record_capacity_drop(detection_index, timestamp)
                    continue
                self.last_assignments[detection_index] = track.subject_id
                self.last_events.append({"event": "PERSON_APPEARED", "subject_id": track.subject_id, "timestamp_seconds": timestamp})

        visible_tracks = [track for track in self._tracks.values() if track.tracking_state != "EXPIRED"]
        return tuple(track.snapshot() for track in sorted(visible_tracks, key=lambda item: item.subject_id))
