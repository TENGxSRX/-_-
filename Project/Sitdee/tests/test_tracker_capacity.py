from __future__ import annotations

import numpy as np

from src.posture_ai.multiperson import AnonymousPoseTracker, PoseDetection
from src.posture_ai.pose.landmarks import PoseLandmarks


def _detection(center_x: float) -> PoseDetection:
    values = np.zeros((33, 3), dtype=float)
    values[0] = [center_x, 0.20, 0.0]
    values[11] = [center_x - 0.03, 0.45, 0.0]
    values[12] = [center_x + 0.03, 0.45, 0.0]
    values[23] = [center_x - 0.03, 0.70, 0.0]
    values[24] = [center_x + 0.03, 0.70, 0.0]
    pose = PoseLandmarks.from_xyz_array(values, visibility=[0.95] * 33)
    return PoseDetection(pose, pose, pose_quality=0.95, core_valid=True, center=(center_x, 0.55))


def _four() -> list[PoseDetection]:
    return [_detection(value) for value in (0.10, 0.30, 0.50, 0.70)]


def test_capacity_drops_only_unmatched_fifth_detection_without_crashing():
    tracker = AnonymousPoseTracker(max_people=4, max_association_distance=0.05, max_misses=3, reacquisition_window_seconds=2.0)
    tracker.update(_four(), 0.0)
    subjects = tracker.update([*_four(), _detection(0.95)], 0.1)
    assert len(subjects) == 4
    assert {item.subject_id for item in subjects} == {"PERSON_01", "PERSON_02", "PERSON_03", "PERSON_04"}
    assert tracker.last_detected_pose_count == 5
    assert tracker.last_dropped_due_to_capacity == 1
    assert tracker.overflow_count == 1
    assert any(event["event"] == "DETECTION_DROPPED_MAX_PEOPLE" for event in tracker.last_events)


def test_temporarily_lost_track_keeps_slot_during_reacquisition_grace_period():
    tracker = AnonymousPoseTracker(max_people=4, max_association_distance=0.05, max_misses=3, reacquisition_window_seconds=2.0)
    tracker.update(_four(), 0.0)
    subjects = tracker.update([_detection(0.10), _detection(0.30), _detection(0.50), _detection(0.95)], 0.1)
    states = {item.subject_id: item.tracking_state for item in subjects}
    assert states["PERSON_04"] == "TEMPORARILY_LOST"
    assert len(subjects) == 4
    assert tracker.last_dropped_due_to_capacity == 1
    assert tracker.track_state("PERSON_04") == "TEMPORARILY_LOST"


def test_expired_track_releases_slot_and_reuses_person_number_safely():
    tracker = AnonymousPoseTracker(max_people=4, max_association_distance=0.05, max_misses=0, reacquisition_window_seconds=2.0)
    tracker.update(_four(), 0.0)
    subjects = tracker.update([_detection(0.10), _detection(0.30), _detection(0.50), _detection(0.95)], 0.1)
    assert {item.subject_id for item in subjects} == {"PERSON_01", "PERSON_02", "PERSON_03", "PERSON_04"}
    assert tracker.last_dropped_due_to_capacity == 0
    assert tracker.last_assignments[3] == "PERSON_04"
    assert any(event["event"] == "PERSON_EXITED" and event["subject_id"] == "PERSON_04" for event in tracker.last_events)
    assert any(event["event"] == "PERSON_APPEARED" and event["subject_id"] == "PERSON_04" for event in tracker.last_events)


def test_repeated_over_capacity_detections_never_grow_ids_or_raise():
    tracker = AnonymousPoseTracker(max_people=4, max_association_distance=0.05, max_misses=3, reacquisition_window_seconds=2.0)
    tracker.update(_four(), 0.0)
    for index in range(1, 12):
        subjects = tracker.update([*_four(), _detection(0.85), _detection(0.95)], index / 10.0)
        assert len(subjects) <= 4
        assert {item.subject_id for item in subjects} <= {"PERSON_01", "PERSON_02", "PERSON_03", "PERSON_04"}
        assert tracker.last_dropped_due_to_capacity == 2
        assert sum(event["event"] == "DETECTION_DROPPED_MAX_PEOPLE" for event in tracker.last_events) == 2


def test_normal_four_person_matching_is_unchanged_at_capacity():
    tracker = AnonymousPoseTracker(max_people=4, max_association_distance=0.05, max_misses=3, reacquisition_window_seconds=2.0)
    tracker.update(_four(), 0.0)
    subjects = tracker.update(list(reversed(_four())), 0.1)
    assert [item.subject_id for item in subjects] == ["PERSON_01", "PERSON_02", "PERSON_03", "PERSON_04"]
    assert tracker.last_dropped_due_to_capacity == 0
    assert all(event["event"] == "PERSON_ACTIVE" for event in tracker.last_events)
