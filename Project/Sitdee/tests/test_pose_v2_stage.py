from __future__ import annotations

import numpy as np
import pytest

from src.posture_ai.detection.yolo_optional import PersonBox, crop_frame
from src.posture_ai.inference.pose_v2_stage import (
    POSE_AMBIGUOUS,
    POSE_CORE_INVALID,
    POSE_OK,
    POSE_PENDING,
    POSE_UNAVAILABLE,
    PoseV2Stage,
    _padded_box,
)
from src.posture_ai.multiperson.tracking import PoseDetection
from src.posture_ai.pose.landmarks import PoseLandmarks


def _pose(visibility: float = 1.0) -> PoseLandmarks:
    xyz = np.zeros((33, 3), dtype=float)
    xyz[0] = [0.0, -0.5, 0.0]
    xyz[11] = [-0.2, 0.0, 0.0]
    xyz[12] = [0.2, 0.0, 0.0]
    xyz[23] = [-0.2, 0.6, 0.0]
    xyz[24] = [0.2, 0.6, 0.0]
    return PoseLandmarks.from_xyz_array(xyz, [visibility] * 33)


class FakeBackend:
    def __init__(self, detection=None):
        self.detection = detection
        self.calls = 0
        self.closed = False

    def detect(self, _crop):
        self.calls += 1
        self.last_crop = _crop
        return [] if self.detection is None else [self.detection]

    def close(self):
        self.closed = True


def _boxes():
    return [
        PersonBox(10, 10, 100, 200, 0.9, track_id=1),
        PersonBox(150, 10, 240, 200, 0.9, track_id=2),
    ]


def test_global_budget_emits_pending_and_fairly_services_waiters():
    pose = _pose()
    backend = FakeBackend(PoseDetection(pose, pose, pose_quality=1.0, core_valid=True))
    stage = PoseV2Stage(backend, max_pose_per_second=2.0)
    frame = np.zeros((240, 300, 3), dtype=np.uint8)

    first = {item.track_id: item for item in stage.process(frame, _boxes(), 0.0)}
    assert first[1].pose_state == POSE_OK
    assert first[2].pose_state == POSE_PENDING
    assert stage.attempts == 1

    waiting = {item.track_id: item for item in stage.process(frame, _boxes(), 0.1)}
    assert waiting[1].pose_state == POSE_PENDING
    assert waiting[2].pose_state == POSE_PENDING
    assert stage.attempts == 1

    second = {item.track_id: item for item in stage.process(frame, _boxes(), 0.5)}
    assert second[2].pose_state == POSE_OK  # never-serviced waiter wins
    assert stage.attempts == 2

    third = {item.track_id: item for item in stage.process(frame, _boxes(), 1.0)}
    assert third[1].pose_state == POSE_OK
    assert stage.attempts == 3
    assert stage.summary()["max_pose_per_second_global"] == 2.0


def test_invalid_pose_is_telemetry_only_and_never_calls_a_classifier():
    pose = _pose(visibility=0.1)
    backend = FakeBackend(PoseDetection(pose, pose, pose_quality=0.1, core_valid=False))
    stage = PoseV2Stage(backend, max_pose_per_second=10.0)
    result = stage.process(np.zeros((240, 300, 3), dtype=np.uint8), [_boxes()[0]], 0.0)[0]
    assert result.pose_state == POSE_CORE_INVALID
    assert result.pose_available is False
    assert stage.summary()["v4_called"] is False


def test_no_pose_is_unavailable_and_close_releases_backend():
    backend = FakeBackend()
    stage = PoseV2Stage(backend, max_pose_per_second=10.0)
    result = stage.process(np.zeros((240, 300, 3), dtype=np.uint8), [_boxes()[0]], 0.0)[0]
    assert result.pose_state == POSE_UNAVAILABLE
    stage.close()
    assert backend.closed is True


def test_selected_box_can_use_budget_when_no_current_neighbor_exists():
    backend = FakeBackend()
    stage = PoseV2Stage(backend, max_pose_per_second=2.0)
    result = stage.process(np.zeros((240, 300, 3), dtype=np.uint8), [_boxes()[0]], 0.0, all_boxes=_boxes()[:1])[0]
    assert result.pose_state == POSE_UNAVAILABLE
    assert stage.attempts == 1
    assert backend.calls == 1


def test_unselected_overlapping_person_blocks_selected_pose_without_using_budget():
    backend = FakeBackend()
    stage = PoseV2Stage(backend, max_pose_per_second=2.0)
    selected = PersonBox(10, 10, 180, 220, 0.9, track_id=1)
    unselected_neighbor = PersonBox(100, 20, 260, 225, 0.9, track_id=2)
    result = stage.process(
        np.zeros((260, 320, 3), dtype=np.uint8),
        [selected], 0.0,
        all_boxes=[selected, unselected_neighbor],
    )[0]
    assert result.pose_state == "POSE_AMBIGUOUS"
    assert stage.attempts == 0
    assert backend.calls == 0


def test_unselected_nonoverlapping_person_is_not_processed_or_returned():
    backend = FakeBackend()
    stage = PoseV2Stage(backend, max_pose_per_second=2.0)
    selected, unselected = _boxes()
    results = stage.process(
        np.zeros((240, 300, 3), dtype=np.uint8),
        [selected], 0.0,
        all_boxes=[selected, unselected],
    )
    assert [item.track_id for item in results] == [1]
    assert stage.attempts == 1
    assert backend.calls == 1


@pytest.mark.parametrize("neighbor_id", [2, None])
def test_disjoint_raw_boxes_with_contaminated_padding_reject_before_budget(neighbor_id):
    backend = FakeBackend()
    stage = PoseV2Stage(backend)
    frame = np.zeros((260, 400, 3), dtype=np.uint8)
    selected = PersonBox(20, 20, 120, 220, 0.9, track_id=1)
    neighbor = PersonBox(125, 20, 230, 220, 0.9, track_id=neighbor_id)
    assert selected.x2 < neighbor.x1
    result = stage.process(frame, [selected], 0, all_boxes=[selected, neighbor])[0]
    assert result.pose_state == POSE_AMBIGUOUS
    assert result.sampling_reason == "PADDED_CROP_REJECTED"
    assert result.detail == "PADDED_CROP_PERSON_CONTAMINATION"
    assert result.bbox == (10, 0, 130, 240)  # actual pixel crop, not raw bbox
    assert not result.pose_available and result.world_landmarks_xyz is None
    record = result.to_record(session_id="TEST", timestamp_seconds=0, timestamp_utc="TEST", frame_index=1)
    assert record["detail"] == "PADDED_CROP_PERSON_CONTAMINATION"
    assert backend.calls == stage.attempts == stage.successes == 0
    assert set(stage._tracks) == {1}  # safety context never becomes eligible
    assert stage._tracks[1].last_attempt_at is None
    # A rejected crop leaves the same global budget unit available immediately.
    stage.process(frame, [selected], 0, all_boxes=[selected])
    assert backend.calls == stage.attempts == 1


def test_clean_padding_excludes_self_by_track_id_and_preserves_exact_crop():
    pose = _pose()
    backend = FakeBackend(PoseDetection(pose, pose, pose_quality=1, core_valid=True))
    stage = PoseV2Stage(backend)
    frame = np.arange(260 * 400 * 3, dtype=np.uint8).reshape(260, 400, 3)
    selected = PersonBox(2.3, 4.7, 102.4, 204.8, 0.9, track_id=1)
    self_copy = PersonBox(*[selected.x1, selected.y1, selected.x2, selected.y2], 0.9, track_id=1)
    neighbor = PersonBox(160, 20, 260, 220, 0.9, track_id=2)
    result = stage.process(frame, [selected], 0, all_boxes=[self_copy, neighbor])[0]
    assert result.pose_state == POSE_OK
    expected, _ = crop_frame(frame, _padded_box(selected, 0.10))
    np.testing.assert_array_equal(backend.last_crop, expected)
    assert result.bbox == (0, 0, 113, 225)
    assert backend.calls == stage.attempts == stage.successes == 1
    assert stage.crop_padding == 0.10 and stage.ambiguity_iou == 0.25


def test_integer_crop_rounding_catches_neighbor_outside_float_padding():
    backend = FakeBackend()
    stage = PoseV2Stage(backend)
    selected = PersonBox(20.2, 20, 120.2, 220, 0.9, track_id=1)
    neighbor = PersonBox(130.5, 20, 240, 220, 0.9, track_id=2)
    assert _padded_box(selected, stage.crop_padding).x2 < neighbor.x1
    result = stage.process(np.zeros((260, 400, 3), dtype=np.uint8), [selected], 0, all_boxes=[selected, neighbor])[0]
    assert result.pose_state == POSE_AMBIGUOUS
    assert result.bbox[2] == 131
    assert backend.calls == stage.attempts == 0


def test_touching_crop_edge_is_not_contamination():
    backend = FakeBackend()
    stage = PoseV2Stage(backend)
    selected = PersonBox(20, 20, 120, 220, 0.9, track_id=1)
    neighbor = PersonBox(130, 20, 240, 220, 0.9, track_id=2)
    result = stage.process(np.zeros((260, 400, 3), dtype=np.uint8), [selected], 0, all_boxes=[selected, neighbor])[0]
    assert result.pose_state == POSE_UNAVAILABLE  # backend ran, returned no pose
    assert backend.calls == stage.attempts == 1


@pytest.mark.parametrize("reverse_neighbors", [False, True])
def test_all_neighbors_are_checked_and_only_clean_selected_track_uses_budget(reverse_neighbors):
    backend = FakeBackend()
    stage = PoseV2Stage(backend)
    selected = PersonBox(20, 20, 120, 220, 0.9, track_id=1)
    clean_selected = PersonBox(350, 20, 450, 220, 0.9, track_id=4)
    neighbors = [PersonBox(200, 20, 300, 220, 0.9, track_id=2),
                 PersonBox(125, 20, 180, 220, 0.9, track_id=3)]
    if reverse_neighbors:
        neighbors.reverse()
    results = stage.process(np.zeros((260, 500, 3), dtype=np.uint8), [selected, clean_selected], 0,
                            all_boxes=[selected, clean_selected, *neighbors])
    assert [item.track_id for item in results] == [1, 4]
    assert [item.pose_state for item in results] == [POSE_AMBIGUOUS, POSE_UNAVAILABLE]
    assert backend.calls == stage.attempts == 1
    assert set(stage._tracks) == {1, 4}


def test_padding_safety_preserves_global_budget_and_waiter_fairness():
    pose = _pose()
    backend = FakeBackend(PoseDetection(pose, pose, pose_quality=1, core_valid=True))
    stage = PoseV2Stage(backend)
    frame = np.zeros((260, 700, 3), dtype=np.uint8)
    selected = [PersonBox(x, 20, x + 100, 220, 0.9, track_id=i)
                for i, x in enumerate((20, 180, 350, 520), start=1)]
    neighbor = PersonBox(125, 20, 170, 220, 0.9, track_id=99)
    first = stage.process(frame, selected, 0, all_boxes=[*selected, neighbor])
    assert [item.pose_state for item in first] == [POSE_AMBIGUOUS, POSE_OK, POSE_PENDING, POSE_PENDING]
    assert backend.calls == stage.attempts == 1
    served = [2]
    for now in (0.1, 0.5, 0.6, 1.0, 1.1, 1.5):
        results = stage.process(frame, selected, now, all_boxes=selected)
        ok = [item.track_id for item in results if item.pose_state == POSE_OK]
        if now in (0.1, 0.6, 1.1):
            assert not ok and all(item.pose_state == POSE_PENDING for item in results)
        served.extend(ok)
    assert served == [2, 1, 3, 4]
    assert backend.calls == stage.attempts == stage.successes == 4
    assert stage.summary()["max_pose_per_second_global"] == 2.0
