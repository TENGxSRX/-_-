"""Room and camera planning utilities based on explicit measurements."""

from .room_camera import (
    angle_difference_deg,
    camera_view_direction,
    distance_camera_to_target,
    floor_coverage_wedge,
    horizontal_coverage_width,
    plan_room,
    target_in_camera_coverage,
    validate_room_spec,
    vertical_coverage_height,
)

__all__ = [
    "angle_difference_deg",
    "camera_view_direction",
    "distance_camera_to_target",
    "floor_coverage_wedge",
    "horizontal_coverage_width",
    "plan_room",
    "target_in_camera_coverage",
    "validate_room_spec",
    "vertical_coverage_height",
]
