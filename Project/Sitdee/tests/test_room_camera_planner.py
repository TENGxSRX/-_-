from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.posture_ai.planning.room_camera import (
    angle_difference_deg,
    camera_view_direction,
    horizontal_coverage_width,
    plan_room,
    render_plan_svg,
    target_in_camera_coverage,
    validate_room_spec,
)


FIXTURE = Path("fixtures/test_room_spec.json")


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_room_schema_requires_real_measurements_and_lists():
    validated = validate_room_spec(_fixture())
    assert validated["spec_status"] == "TEST_ROOM_ONLY"
    assert validated["planning_estimate"] == "GEOMETRIC PLANNING ESTIMATE"
    with pytest.raises(ValueError, match="room.width_m"):
        validate_room_spec({"room": {"length_m": 4}, "cameras": [], "targets": []})


def test_fov_direction_distance_and_target_coverage_are_explicit():
    assert horizontal_coverage_width(5.0, 90.0) == pytest.approx(10.0)
    assert camera_view_direction(0.0)["x"] == pytest.approx(1.0)
    assert camera_view_direction(90.0)["y"] == pytest.approx(1.0)
    assert angle_difference_deg(-10.0, 350.0) == pytest.approx(0.0)
    camera = {"id": "CAM", "x_m": 0, "y_m": 0, "height_m": 2, "direction_deg": 0, "hfov_deg": 90}
    target = {"id": "SEAT", "x_m": 2, "y_m": 0}
    result = target_in_camera_coverage(camera, target)
    assert result["visible_estimate"] is True
    assert result["distance_m"] == pytest.approx(2.0)


def test_obstacle_creates_a_blind_spot_candidate():
    camera = {"id": "CAM", "x_m": 0, "y_m": 0, "height_m": 2, "direction_deg": 0, "hfov_deg": 90}
    target = {"id": "SEAT", "x_m": 4, "y_m": 0}
    obstacle = {"id": "MONITOR", "polygon": [[1, -1], [2, -1], [2, 1], [1, 1]]}
    result = target_in_camera_coverage(camera, target, [obstacle])
    assert result["visible_estimate"] is False
    assert result["blocked_by"] == ["MONITOR"]


def test_plan_and_svg_inputs_are_test_fixture_only():
    plan = plan_room(_fixture())
    assert plan["spec_status"] == "TEST_ROOM_ONLY"
    assert plan["camera_count_decision"] == "PENDING_REAL_FLOOR_PLAN_REVIEW"
    assert len(plan["target_coverage"]) == 2
    assert all("planning_estimate" in pair for camera in plan["cameras"] for pair in camera["target_results"])
    svg = render_plan_svg(plan)
    assert "GEOMETRIC PLANNING ESTIMATE" in svg
    assert "CAM_TEST_01" in svg
