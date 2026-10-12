"""Explicit anonymous target lock for human-labelled CCTV collection.

The lock is a data-integrity guard, not person recognition.  A human chooses a
visible anonymous tracked box.  If that box becomes missing or spatially
ambiguous, the controller pauses collection and never chooses another person
on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


TARGET_LOCK_STATES = frozenset({
    "NO_TARGET",
    "TARGET_LOCKED",
    "TARGET_UNCERTAIN",
    "TARGET_RESELECT_REQUIRED",
})
IDENTITY_MODES = frozenset({"YOLO_TRACK", "MEDIAPIPE_EPHEMERAL"})


def _normalised_box(value: tuple[float, float, float, float] | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError("target box must contain four values")
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("target box must be finite and ordered")
    return box


@dataclass(frozen=True)
class TargetCandidate:
    """One anonymous candidate visible in the current frame."""

    identity_key: str
    identity_mode: str
    runtime_subject_id: str | None
    yolo_track_id: int | None
    bbox: tuple[float, float, float, float] | None
    association_state: str = "ASSOCIATED"
    association_score: float | None = None
    pose_valid: bool = False
    core_valid: bool = False

    def __post_init__(self) -> None:
        key = str(self.identity_key).strip()
        if not key:
            raise ValueError("identity_key must be non-empty")
        mode = str(self.identity_mode).upper()
        if mode not in IDENTITY_MODES:
            raise ValueError(f"identity_mode must be one of {sorted(IDENTITY_MODES)}")
        state = str(self.association_state).upper()
        if state not in {"ASSOCIATED", "AMBIGUOUS", "UNMATCHED"}:
            raise ValueError("association_state must be ASSOCIATED, AMBIGUOUS, or UNMATCHED")
        score = None if self.association_score is None else float(self.association_score)
        if score is not None and (not math.isfinite(score) or not 0.0 <= score <= 1.0):
            raise ValueError("association_score must be finite between 0 and 1")
        track = None if self.yolo_track_id is None else int(self.yolo_track_id)
        if mode == "YOLO_TRACK" and track is None:
            raise ValueError("YOLO_TRACK candidates require a tracker ID")
        object.__setattr__(self, "identity_key", key)
        object.__setattr__(self, "identity_mode", mode)
        object.__setattr__(self, "runtime_subject_id", None if self.runtime_subject_id is None else str(self.runtime_subject_id))
        object.__setattr__(self, "yolo_track_id", track)
        object.__setattr__(self, "bbox", _normalised_box(self.bbox))
        object.__setattr__(self, "association_state", state)
        object.__setattr__(self, "association_score", score)

    @property
    def is_unambiguous(self) -> bool:
        return self.association_state == "ASSOCIATED"

    @property
    def can_supply_pose(self) -> bool:
        return self.is_unambiguous and bool(self.pose_valid) and bool(self.core_valid)

    def contains(self, x: float, y: float) -> bool:
        if self.bbox is None:
            return False
        return self.bbox[0] <= float(x) <= self.bbox[2] and self.bbox[1] <= float(y) <= self.bbox[3]


@dataclass(frozen=True)
class TargetLockSnapshot:
    state: str
    identity_key: str | None
    identity_mode: str | None
    target_yolo_track_id: int | None
    runtime_subject_id: str | None
    target_bbox: tuple[float, float, float, float] | None
    association_state: str | None
    association_score: float | None
    pause_reason: str | None
    auto_paused: bool
    can_collect: bool

    def __post_init__(self) -> None:
        if self.state not in TARGET_LOCK_STATES:
            raise ValueError("invalid target lock state")

    def to_dict(self) -> dict[str, object]:
        return {
            "target_lock_state": self.state,
            "target_identity_key": self.identity_key,
            "target_identity_mode": self.identity_mode,
            "target_yolo_track_id": self.target_yolo_track_id,
            "target_runtime_subject_id": self.runtime_subject_id,
            "target_bbox": None if self.target_bbox is None else list(self.target_bbox),
            "target_association_state": self.association_state,
            "target_association_score": self.association_score,
            "target_pause_reason": self.pause_reason,
            "target_auto_paused": self.auto_paused,
            "target_can_collect": self.can_collect,
        }


class TargetLockController:
    """Maintain one explicitly selected anonymous target without reassignment."""

    def __init__(self, *, yolo_reacquisition_window_seconds: float = 1.0) -> None:
        if yolo_reacquisition_window_seconds <= 0.0:
            raise ValueError("yolo_reacquisition_window_seconds must be positive")
        self.yolo_reacquisition_window_seconds = float(yolo_reacquisition_window_seconds)
        self._identity_key: str | None = None
        self._identity_mode: str | None = None
        self._last_confirmed_at: float | None = None
        self._generic_absence_seen = False
        self._snapshot = TargetLockSnapshot(
            state="NO_TARGET",
            identity_key=None,
            identity_mode=None,
            target_yolo_track_id=None,
            runtime_subject_id=None,
            target_bbox=None,
            association_state=None,
            association_score=None,
            pause_reason="TARGET_NOT_SELECTED",
            auto_paused=True,
            can_collect=False,
        )

    @property
    def snapshot(self) -> TargetLockSnapshot:
        return self._snapshot

    def clear(self) -> TargetLockSnapshot:
        self._identity_key = None
        self._identity_mode = None
        self._last_confirmed_at = None
        self._generic_absence_seen = False
        self._snapshot = TargetLockSnapshot(
            state="NO_TARGET",
            identity_key=None,
            identity_mode=None,
            target_yolo_track_id=None,
            runtime_subject_id=None,
            target_bbox=None,
            association_state=None,
            association_score=None,
            pause_reason="TARGET_CLEARED_BY_OPERATOR",
            auto_paused=True,
            can_collect=False,
        )
        return self._snapshot

    def select(self, candidate: TargetCandidate, *, timestamp_seconds: float) -> TargetLockSnapshot:
        """Lock an operator-selected candidate only when spatially unambiguous."""

        if not candidate.is_unambiguous:
            raise ValueError("cannot lock an ambiguous or unmatched target candidate")
        timestamp = float(timestamp_seconds)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_seconds must be finite")
        self._identity_key = candidate.identity_key
        self._identity_mode = candidate.identity_mode
        self._last_confirmed_at = timestamp
        self._generic_absence_seen = False
        self._snapshot = self._from_candidate(candidate, state="TARGET_LOCKED", pause_reason=None)
        return self._snapshot

    def select_at(self, candidates: Iterable[TargetCandidate], x: float, y: float, *, timestamp_seconds: float) -> TargetLockSnapshot:
        """Select an unambiguous box explicitly clicked by the operator."""

        containing = [candidate for candidate in candidates if candidate.contains(x, y)]
        if not containing:
            raise ValueError("no target candidate at the selected point")
        safe = [candidate for candidate in containing if candidate.is_unambiguous]
        if len(safe) != 1:
            raise ValueError("target point is ambiguous; wait for separated boxes before locking")
        return self.select(safe[0], timestamp_seconds=timestamp_seconds)

    def update(self, candidates: Iterable[TargetCandidate], *, timestamp_seconds: float) -> TargetLockSnapshot:
        """Update the lock state; never substitute another candidate automatically."""

        timestamp = float(timestamp_seconds)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp_seconds must be finite")
        values = tuple(candidates)
        if self._identity_key is None or self._identity_mode is None:
            self._snapshot = TargetLockSnapshot(
                state="NO_TARGET",
                identity_key=None,
                identity_mode=None,
                target_yolo_track_id=None,
                runtime_subject_id=None,
                target_bbox=None,
                association_state=None,
                association_score=None,
                pause_reason="TARGET_NOT_SELECTED",
                auto_paused=True,
                can_collect=False,
            )
            return self._snapshot

        matching = [candidate for candidate in values if candidate.identity_key == self._identity_key and candidate.identity_mode == self._identity_mode]
        if len(matching) == 1 and matching[0].is_unambiguous:
            candidate = matching[0]
            if self._identity_mode == "MEDIAPIPE_EPHEMERAL" and self._generic_absence_seen:
                self._snapshot = self._from_candidate(candidate, state="TARGET_RESELECT_REQUIRED", pause_reason="MEDIAPIPE_TRACK_WAS_INTERRUPTED_RESELECT_REQUIRED")
                return self._snapshot
            self._last_confirmed_at = timestamp
            self._snapshot = self._from_candidate(candidate, state="TARGET_LOCKED", pause_reason=None)
            return self._snapshot

        if len(matching) > 1:
            reason = "MULTIPLE_MATCHING_TARGET_CANDIDATES"
        elif len(matching) == 1:
            reason = "TARGET_ASSOCIATION_AMBIGUOUS"
        else:
            reason = "TARGET_NOT_VISIBLE_OR_NOT_ASSOCIATED"

        if self._identity_mode == "MEDIAPIPE_EPHEMERAL":
            self._generic_absence_seen = True
            state = "TARGET_RESELECT_REQUIRED"
            reason = "MEDIAPIPE_TRACK_INTERRUPTED_RESELECT_REQUIRED"
        elif self._last_confirmed_at is not None and timestamp - self._last_confirmed_at <= self.yolo_reacquisition_window_seconds:
            state = "TARGET_UNCERTAIN"
        else:
            state = "TARGET_RESELECT_REQUIRED"
            reason = "YOLO_TRACK_REACQUISITION_WINDOW_EXPIRED"
        self._snapshot = TargetLockSnapshot(
            state=state,
            identity_key=self._identity_key,
            identity_mode=self._identity_mode,
            target_yolo_track_id=self._track_id_from_key(),
            runtime_subject_id=None,
            target_bbox=None,
            association_state=None if not matching else matching[0].association_state,
            association_score=None if not matching else matching[0].association_score,
            pause_reason=reason,
            auto_paused=True,
            can_collect=False,
        )
        return self._snapshot

    def _track_id_from_key(self) -> int | None:
        if self._identity_mode != "YOLO_TRACK" or self._identity_key is None:
            return None
        try:
            return int(self._identity_key.removeprefix("YOLO_TRACK_"))
        except ValueError:
            return None

    def _from_candidate(self, candidate: TargetCandidate, *, state: str, pause_reason: str | None) -> TargetLockSnapshot:
        can_collect = state == "TARGET_LOCKED" and candidate.can_supply_pose
        return TargetLockSnapshot(
            state=state,
            identity_key=candidate.identity_key,
            identity_mode=candidate.identity_mode,
            target_yolo_track_id=candidate.yolo_track_id,
            runtime_subject_id=candidate.runtime_subject_id,
            target_bbox=candidate.bbox,
            association_state=candidate.association_state,
            association_score=candidate.association_score,
            pause_reason=pause_reason if pause_reason is not None else (None if can_collect else "TARGET_POSE_OR_CORE_INVALID"),
            auto_paused=not can_collect,
            can_collect=can_collect,
        )


__all__ = [
    "IDENTITY_MODES",
    "TARGET_LOCK_STATES",
    "TargetCandidate",
    "TargetLockController",
    "TargetLockSnapshot",
]
