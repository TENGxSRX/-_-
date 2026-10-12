"""Conservative full-frame association between YOLO people and MediaPipe poses.

The module contains no Ultralytics import.  It accepts anonymous YOLO person
boxes and MediaPipe pose-like objects, then performs a small one-to-one spatial
assignment.  It is intentionally stricter than a best-effort overlay: when
overlapping boxes leave a plausible alternative, the result is marked
``AMBIGUOUS`` so a ground-truth collector can pause instead of changing target.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Sequence

from src.posture_ai.detection.yolo_optional import PersonBox


NormalizedBox = tuple[float, float, float, float]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalized_item_box(item: Any) -> NormalizedBox | None:
    box = getattr(item, "bbox", None)
    if box is None or len(box) != 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        return None
    if right <= left or bottom <= top:
        return None
    return (_clamp(left), _clamp(top), _clamp(right), _clamp(bottom))


def _normalized_item_center(item: Any, box: NormalizedBox | None) -> tuple[float, float] | None:
    center = getattr(item, "center", None)
    if center is not None and len(center) == 2:
        try:
            x, y = (float(value) for value in center)
        except (TypeError, ValueError):
            x = y = math.nan
        if math.isfinite(x) and math.isfinite(y):
            return (_clamp(x), _clamp(y))
    if box is None:
        return None
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def normalized_iou(first: NormalizedBox, second: NormalizedBox) -> float:
    """Return intersection-over-union for normalized ``left, top, right, bottom`` boxes."""

    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return 0.0 if union <= 0.0 else float(intersection / union)


def _inside(point: tuple[float, float] | None, box: NormalizedBox) -> bool:
    return point is not None and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _center_distance(point: tuple[float, float] | None, box: NormalizedBox) -> float:
    if point is None:
        return math.inf
    center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
    return float(math.hypot(point[0] - center[0], point[1] - center[1]))


@dataclass(frozen=True)
class PersonPoseAssociation:
    """One spatial match; never a claim of real-world identity."""

    pose_index: int
    person_index: int
    yolo_track_id: int | None
    score: float
    pose_center_inside_person_box: bool
    bbox_iou: float
    center_distance: float
    state: str
    ambiguity_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in {"ASSOCIATED", "AMBIGUOUS"}:
            raise ValueError("association state must be ASSOCIATED or AMBIGUOUS")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("association score must be between 0 and 1")


@dataclass(frozen=True)
class AssociationResult:
    associations: tuple[PersonPoseAssociation, ...]
    unmatched_pose_indices: tuple[int, ...]
    unmatched_person_indices: tuple[int, ...]
    candidate_pair_count: int
    strategy: str = "FULL_FRAME_SPATIAL_ONE_TO_ONE"

    def for_person_index(self, index: int) -> PersonPoseAssociation | None:
        return next((item for item in self.associations if item.person_index == int(index)), None)


def associate_pose_items_to_person_boxes(
    pose_items: Sequence[Any],
    person_boxes: Sequence[PersonBox],
    *,
    frame_width: int,
    frame_height: int,
    min_score: float = 0.35,
    min_iou: float = 0.05,
    max_center_distance: float = 0.18,
    ambiguity_margin: float = 0.08,
) -> AssociationResult:
    """Associate full-frame MediaPipe poses to full-frame YOLO person boxes.

    The score combines three observable spatial signals: whether the pose torso
    center lies inside the person box, pose/box IoU, and center proximity.  The
    constants are explicit engineering gates, not learned calibration values.
    A close second match is retained as an ambiguity flag rather than silently
    discarded.
    """

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if not 0.0 <= min_score <= 1.0 or min_iou < 0.0 or max_center_distance <= 0.0 or ambiguity_margin < 0.0:
        raise ValueError("association thresholds are invalid")

    pose_boxes = [_normalized_item_box(item) for item in pose_items]
    pose_centers = [_normalized_item_center(item, box) for item, box in zip(pose_items, pose_boxes)]
    person_normalized = [box.normalized(frame_width, frame_height) for box in person_boxes]
    candidates: dict[tuple[int, int], dict[str, float | bool]] = {}
    for pose_index, (pose_box, pose_center) in enumerate(zip(pose_boxes, pose_centers)):
        if pose_center is None and pose_box is None:
            continue
        for person_index, person_box in enumerate(person_normalized):
            inside = _inside(pose_center, person_box)
            overlap = 0.0 if pose_box is None else normalized_iou(pose_box, person_box)
            distance = _center_distance(pose_center, person_box)
            distance_score = max(0.0, 1.0 - min(1.0, distance / max_center_distance)) if math.isfinite(distance) else 0.0
            # A torso center inside a detected person is the strongest signal;
            # IoU and distance only refine it.  No appearance embedding is used.
            score = (0.55 if inside else 0.0) + 0.30 * overlap + 0.15 * distance_score
            eligible = bool(inside or overlap >= min_iou or distance <= max_center_distance)
            if eligible and score >= min_score:
                candidates[(pose_index, person_index)] = {
                    "score": min(1.0, float(score)),
                    "inside": inside,
                    "iou": overlap,
                    "distance": distance,
                }

    # At most four poses/people are supported by the project, so exhaustive
    # one-to-one assignment is transparent and avoids a new scipy dependency.
    best_pairs: tuple[tuple[int, int], ...] = ()
    best_value: tuple[int, float] = (-1, -1.0)
    pose_indices = tuple(range(len(pose_items)))

    def visit(offset: int, used_people: set[int], chosen: list[tuple[int, int]], total_score: float) -> None:
        nonlocal best_pairs, best_value
        if offset >= len(pose_indices):
            candidate_value = (len(chosen), round(total_score, 12))
            if candidate_value > best_value:
                best_value = candidate_value
                best_pairs = tuple(chosen)
            return
        pose_index = pose_indices[offset]
        visit(offset + 1, used_people, chosen, total_score)
        options = sorted(
            ((pair, item) for pair, item in candidates.items() if pair[0] == pose_index and pair[1] not in used_people),
            key=lambda item: (-float(item[1]["score"]), item[0][1]),
        )
        for (pair_pose, person_index), item in options:
            used_people.add(person_index)
            chosen.append((pair_pose, person_index))
            visit(offset + 1, used_people, chosen, total_score + float(item["score"]))
            chosen.pop()
            used_people.remove(person_index)

    visit(0, set(), [], 0.0)
    selected_pairs = set(best_pairs)
    associations: list[PersonPoseAssociation] = []
    for pose_index, person_index in sorted(best_pairs):
        selected = candidates[(pose_index, person_index)]
        alternatives: list[str] = []
        selected_score = float(selected["score"])
        for (candidate_pose, candidate_person), candidate in candidates.items():
            if (candidate_pose, candidate_person) == (pose_index, person_index):
                continue
            if float(candidate["score"]) < selected_score - ambiguity_margin:
                continue
            if candidate_pose == pose_index:
                alternatives.append("POSE_HAS_OTHER_PLAUSIBLE_PERSON_BOX")
            if candidate_person == person_index:
                alternatives.append("PERSON_BOX_HAS_OTHER_PLAUSIBLE_POSE")
        reasons = tuple(sorted(set(alternatives)))
        associations.append(
            PersonPoseAssociation(
                pose_index=pose_index,
                person_index=person_index,
                yolo_track_id=person_boxes[person_index].track_id,
                score=selected_score,
                pose_center_inside_person_box=bool(selected["inside"]),
                bbox_iou=float(selected["iou"]),
                center_distance=float(selected["distance"]),
                state="AMBIGUOUS" if reasons else "ASSOCIATED",
                ambiguity_reasons=reasons,
            )
        )
    return AssociationResult(
        associations=tuple(associations),
        unmatched_pose_indices=tuple(index for index in range(len(pose_items)) if index not in {item[0] for item in selected_pairs}),
        unmatched_person_indices=tuple(index for index in range(len(person_boxes)) if index not in {item[1] for item in selected_pairs}),
        candidate_pair_count=len(candidates),
    )


__all__ = [
    "AssociationResult",
    "PersonPoseAssociation",
    "associate_pose_items_to_person_boxes",
    "normalized_iou",
]
