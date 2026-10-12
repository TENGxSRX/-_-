from __future__ import annotations

import json

import numpy as np

from src.posture_ai.multiperson import (
    SeatAwareRuntime,
    SeatAwareRuntimeConfig,
    SeatZone,
    dashboard_seat_cards,
    load_seat_layout,
)
from src.posture_ai.multiperson.seating import draw_seat_layout
from src.posture_ai.multiperson.tracking import PoseDetection
from src.posture_ai.pose.landmarks import PoseLandmarks


class _FixtureV4:
    labels = ["TLB", "TUP"]

    def predict_world_landmarks(self, world_landmarks, *, image_visibility=None, visibility_threshold=0.5):
        x = float((world_landmarks.by_name("left_shoulder").x + world_landmarks.by_name("right_shoulder").x) / 2.0)
        label = "TLB" if x < 0.5 else "TUP"
        return {
            "label": label,
            "display_label": "Lean backward" if label == "TLB" else "Upright",
            "confidence": 0.9,
            "probabilities": {"TLB": 0.9 if label == "TLB" else 0.1, "TUP": 0.1 if label == "TLB" else 0.9},
            "feature_count": 15,
        }


def _pose(x: float, *, core_valid: bool = True) -> PoseLandmarks:
    values = np.zeros((33, 3), dtype=float)
    values[0] = [x, 0.25, 0.0]
    values[11] = [x - 0.10, 0.45, 0.0]
    values[12] = [x + 0.10, 0.45, 0.0]
    values[23] = [x - 0.08, 0.72, 0.0]
    values[24] = [x + 0.08, 0.72, 0.0]
    if not core_valid:
        values[23] = [np.nan, np.nan, np.nan]
        values[24] = [np.nan, np.nan, np.nan]
    return PoseLandmarks.from_xyz_array(values, visibility=[0.95] * 33)


def _detection(x: float, *, core_valid: bool = True) -> PoseDetection:
    pose = _pose(x, core_valid=core_valid)
    return PoseDetection(pose, pose, pose_quality=0.95 if core_valid else 0.2, core_valid=core_valid, center=(x, 0.55))


def _zones(count: int = 3) -> list[SeatZone]:
    if count == 1:
        return [SeatZone("SEAT_02", roi=(0.35, 0.20, 0.65, 0.90), anchor_x=0.50, anchor_y=0.62)]
    return [
        SeatZone("SEAT_01", roi=(0.05, 0.20, 0.30, 0.90), anchor_x=0.175, anchor_y=0.62),
        SeatZone("SEAT_02", roi=(0.35, 0.20, 0.65, 0.90), anchor_x=0.50, anchor_y=0.62),
        SeatZone("SEAT_03", roi=(0.70, 0.20, 0.95, 0.90), anchor_x=0.825, anchor_y=0.62),
    ]


def _runtime(*, count: int = 3, **kwargs) -> SeatAwareRuntime:
    config_values = {"max_people": 3, **kwargs}
    config = SeatAwareRuntimeConfig(**config_values)
    return SeatAwareRuntime(_FixtureV4(), _zones(count), runtime_config=config)


def _occupy_three(runtime: SeatAwareRuntime) -> None:
    runtime.process_frame([_detection(0.20), _detection(0.50), _detection(0.80)], 0.0)
    result = runtime.process_frame([_detection(0.20), _detection(0.50), _detection(0.80)], 1.6)
    assert all(seat.assignment_state == "OCCUPIED" for seat in result.seats)


def test_two_seated_people_are_assigned_to_distinct_seats():
    runtime = _runtime()
    runtime.process_frame([_detection(0.20), _detection(0.50)], 0.0)
    result = runtime.process_frame([_detection(0.20), _detection(0.50)], 1.6)
    assert [(seat.seat_id, seat.assignment_state) for seat in result.seats] == [
        ("SEAT_01", "OCCUPIED"), ("SEAT_02", "OCCUPIED"), ("SEAT_03", "EMPTY")
    ]


def test_three_seated_people_are_supported_without_changing_v4():
    runtime = _runtime()
    _occupy_three(runtime)
    assert [seat.v4_prediction for seat in runtime.last_result.seats] == ["TLB", "TUP", "TUP"]


def test_passerby_does_not_replace_an_occupied_seat_after_one_frame():
    runtime = _runtime(count=1, max_people=2)
    runtime.process_frame([_detection(0.50)], 0.0)
    occupied = runtime.process_frame([_detection(0.50)], 1.6).seats[0]
    old_session = occupied.occupant_session_id
    for timestamp in (2.0, 2.2, 2.4, 2.6):
        runtime.process_frame([], timestamp)
    result = runtime.process_frame([_detection(0.50)], 2.7).seats[0]
    assert result.assignment_state in {"LOST", "OCCLUDED"}
    assert result.occupant_session_id == old_session
    assert "PASSERBY_DETECTED" in result.interference_statuses


def test_detection_order_swap_does_not_swap_seat_context():
    runtime = _runtime()
    runtime.process_frame([_detection(0.20), _detection(0.50)], 0.0)
    result = runtime.process_frame([_detection(0.50), _detection(0.20)], 0.1)
    assert result.seats[0].candidate_track_id == "PERSON_01"
    assert result.seats[1].candidate_track_id == "PERSON_02"


def test_short_occlusion_keeps_seat_session_and_uses_conservative_pose_lost():
    runtime = _runtime(count=1)
    runtime.process_frame([_detection(0.50)], 0.0)
    occupied = runtime.process_frame([_detection(0.50)], 1.6).seats[0]
    result = runtime.process_frame([], 2.0).seats[0]
    assert result.assignment_state == "OCCLUDED"
    assert result.occupant_session_id == occupied.occupant_session_id
    assert result.temporal_state["risk_level"] == "POSE_LOST"
    assert "SEAT_OCCLUDED" in result.interference_statuses


def test_seat_becomes_lost_then_empty_after_configured_timeout():
    runtime = _runtime(count=1)
    runtime.process_frame([_detection(0.50)], 0.0)
    runtime.process_frame([_detection(0.50)], 1.6)
    assert runtime.process_frame([], 2.8).seats[0].assignment_state == "LOST"
    assert runtime.process_frame([], 4.7).seats[0].assignment_state == "EMPTY"


def test_new_person_occupies_empty_seat_with_new_session_boundary():
    runtime = _runtime(count=1)
    runtime.process_frame([_detection(0.50)], 0.0)
    first = runtime.process_frame([_detection(0.50)], 1.6).seats[0]
    runtime.process_frame([], 2.5)
    runtime.process_frame([], 4.7)
    candidate = runtime.process_frame([_detection(0.50)], 5.0).seats[0]
    replacement = runtime.process_frame([_detection(0.50)], 6.6).seats[0]
    assert candidate.assignment_state == "CANDIDATE"
    assert replacement.assignment_state == "OCCUPIED"
    assert replacement.occupant_session_id != first.occupant_session_id


def test_two_detections_overlapping_one_seat_are_not_double_assigned():
    runtime = _runtime(count=1, max_people=2)
    result = runtime.process_frame([_detection(0.48), _detection(0.52)], 0.0).seats[0]
    assert result.assignment_state == "EMPTY"
    assert "MULTIPLE_CANDIDATES" in result.interference_statuses
    assert result.current_track_id is None


def test_core_landmark_failure_does_not_make_an_occupied_seat_empty():
    runtime = _runtime(count=1)
    runtime.process_frame([_detection(0.50)], 0.0)
    runtime.process_frame([_detection(0.50)], 1.6)
    result = runtime.process_frame([_detection(0.50, core_valid=False)], 1.8).seats[0]
    assert result.assignment_state == "OCCUPIED"
    assert result.current_track_id == "PERSON_01"
    assert result.core_valid is False
    assert "INPUT_UNCERTAIN" in result.interference_statuses
    assert result.temporal_state["risk_level"] == "POSE_LOST"


def test_tracker_id_fragment_can_keep_seat_session_when_location_is_continuous():
    runtime = _runtime(count=1)
    runtime.process_frame([_detection(0.50)], 0.0)
    first = runtime.process_frame([_detection(0.50)], 1.6).seats[0]
    for timestamp in (1.8, 2.0, 2.2, 2.4):
        runtime.process_frame([], timestamp)
    result = runtime.process_frame([_detection(0.50)], 2.5).seats[0]
    assert result.current_track_id != first.current_track_id
    assert result.occupant_session_id == first.occupant_session_id
    assert result.assignment_state == "OCCUPIED"


def test_normalized_layout_loads_and_visualizer_preserves_resolution_independence():
    camera_id, zones = load_seat_layout("configs/seat_layout_example.yaml")
    assert camera_id == "CAM01"
    assert [zone.seat_id for zone in zones] == ["SEAT_01", "SEAT_02", "SEAT_03"]
    assert zones[0].contains((0.20, 0.72))
    assert not zones[0].contains((0.50, 0.72))
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    rendered = draw_seat_layout(image, zones)
    assert rendered.shape == image.shape
    result = _runtime().process_frame([_detection(0.20)], 0.0)
    cards = dashboard_seat_cards(result)
    json.dumps(cards, ensure_ascii=False, allow_nan=False)
    assert cards[0]["seat_id"] == "SEAT_01"


def test_seat_aware_runtime_accepts_three_plus_seats_and_keeps_generic_module_optional():
    runtime = _runtime()
    result = runtime.process_frame([], 0.0)
    assert len(result.seats) == 3
    assert result.tracking_status == "SEAT_AWARE_SOFTWARE_PROTOTYPE"
    assert result.to_dict()["tracking_mode"] == "seat"
