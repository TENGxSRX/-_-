"""Measurement-driven room/camera planning estimates.

This module intentionally does not choose real room dimensions or claim exact
optical coverage. Its outputs are geometric planning estimates that must be
reviewed against the real floor plan, camera model, occlusions, and privacy
constraints.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from html import escape
from typing import Any


PLANNING_ESTIMATE = "GEOMETRIC PLANNING ESTIMATE"


def _number(value: Any, name: str, *, positive: bool = False, non_negative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    if non_negative and result < 0:
        raise ValueError(f"{name} must not be negative")
    return result


def _optional_number(value: Any, name: str, *, positive: bool = False, non_negative: bool = False) -> float | None:
    return None if value is None else _number(value, name, positive=positive, non_negative=non_negative)


def _point(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must be [x_m, y_m]")
    return _number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]")


def angle_difference_deg(first: float, second: float) -> float:
    """Signed shortest difference ``first - second`` in [-180, 180)."""

    return (float(first) - float(second) + 180.0) % 360.0 - 180.0


def camera_view_direction(direction_deg: float) -> dict[str, float]:
    """Return a unit vector; 0 degrees points along +x and 90 along +y."""

    angle = math.radians(float(direction_deg))
    return {"x": math.cos(angle), "y": math.sin(angle), "direction_deg": float(direction_deg) % 360.0}


def distance_camera_to_target(camera: Mapping[str, Any], target: Mapping[str, Any]) -> float:
    dx = _number(target.get("x_m"), "target.x_m") - _number(camera.get("x_m"), "camera.x_m")
    dy = _number(target.get("y_m"), "target.y_m") - _number(camera.get("y_m"), "camera.y_m")
    return math.hypot(dx, dy)


def horizontal_coverage_width(distance_m: float, hfov_deg: float) -> float:
    distance = _number(distance_m, "distance_m", non_negative=True)
    hfov = _number(hfov_deg, "hfov_deg", positive=True)
    if hfov >= 180.0:
        raise ValueError("hfov_deg must be less than 180")
    return 2.0 * distance * math.tan(math.radians(hfov) / 2.0)


def vertical_coverage_height(distance_m: float, vfov_deg: float) -> float:
    distance = _number(distance_m, "distance_m", non_negative=True)
    vfov = _number(vfov_deg, "vfov_deg", positive=True)
    if vfov >= 180.0:
        raise ValueError("vfov_deg must be less than 180")
    return 2.0 * distance * math.tan(math.radians(vfov) / 2.0)


def _validate_polygon(value: Any, name: str) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        raise ValueError(f"{name} must contain at least three [x_m, y_m] points")
    return [[float(x), float(y)] for x, y in (_point(point, f"{name}[{index}]") for index, point in enumerate(value))]


def _normalise_obstacle(obstacle: Mapping[str, Any], index: int) -> dict[str, Any]:
    name = f"obstacles[{index}]"
    if not isinstance(obstacle, Mapping):
        raise ValueError(f"{name} must be an object")
    obstacle_id = str(obstacle.get("id", f"OBSTACLE_{index + 1:02d}"))
    if "polygon" in obstacle:
        polygon = _validate_polygon(obstacle["polygon"], f"{name}.polygon")
    elif all(key in obstacle for key in ("x_m", "y_m", "width_m", "length_m")):
        x = _number(obstacle["x_m"], f"{name}.x_m")
        y = _number(obstacle["y_m"], f"{name}.y_m")
        width = _number(obstacle["width_m"], f"{name}.width_m", positive=True)
        length = _number(obstacle["length_m"], f"{name}.length_m", positive=True)
        polygon = [[x, y], [x + width, y], [x + width, y + length], [x, y + length]]
    else:
        raise ValueError(f"{name} needs polygon or x_m/y_m/width_m/length_m")
    return {"id": obstacle_id, "type": str(obstacle.get("type", "unspecified")), "polygon": polygon}


def validate_room_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a JSON/YAML room specification without inventing values."""

    if not isinstance(spec, Mapping):
        raise ValueError("room specification must be an object")
    room = spec.get("room")
    if not isinstance(room, Mapping):
        raise ValueError("room object is required")
    width = _number(room.get("width_m"), "room.width_m", positive=True)
    length = _number(room.get("length_m"), "room.length_m", positive=True)
    ceiling = _optional_number(room.get("ceiling_height_m"), "room.ceiling_height_m", positive=True)

    cameras = spec.get("cameras", [])
    targets = spec.get("targets", [])
    obstacles = spec.get("obstacles", [])
    if not isinstance(cameras, Sequence) or isinstance(cameras, (str, bytes)):
        raise ValueError("cameras must be a list")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise ValueError("targets must be a list")
    if not isinstance(obstacles, Sequence) or isinstance(obstacles, (str, bytes)):
        raise ValueError("obstacles must be a list")
    if not cameras:
        raise ValueError("at least one candidate camera is required")
    if not targets:
        raise ValueError("at least one target seat/workstation is required")

    normalized_cameras: list[dict[str, Any]] = []
    for index, camera in enumerate(cameras):
        name = f"cameras[{index}]"
        if not isinstance(camera, Mapping):
            raise ValueError(f"{name} must be an object")
        item = dict(camera)
        if not str(item.get("id", "")).strip():
            raise ValueError(f"{name}.id is required")
        item["id"] = str(item["id"])
        for key in ("x_m", "y_m", "height_m", "direction_deg"):
            item[key] = _number(item.get(key), f"{name}.{key}", positive=key == "height_m")
        item["hfov_deg"] = _optional_number(item.get("hfov_deg"), f"{name}.hfov_deg", positive=True)
        item["vfov_deg"] = _optional_number(item.get("vfov_deg"), f"{name}.vfov_deg", positive=True)
        if item["hfov_deg"] is not None and item["hfov_deg"] >= 180.0:
            raise ValueError(f"{name}.hfov_deg must be less than 180")
        if item["vfov_deg"] is not None and item["vfov_deg"] >= 180.0:
            raise ValueError(f"{name}.vfov_deg must be less than 180")
        item["min_useful_distance_m"] = _optional_number(item.get("min_useful_distance_m"), f"{name}.min_useful_distance_m", non_negative=True)
        item["max_useful_distance_m"] = _optional_number(item.get("max_useful_distance_m"), f"{name}.max_useful_distance_m", positive=True)
        resolution = item.get("resolution")
        if resolution is not None:
            if not isinstance(resolution, Mapping):
                raise ValueError(f"{name}.resolution must be an object")
            item["resolution"] = {
                "width_px": int(_number(resolution.get("width_px"), f"{name}.resolution.width_px", positive=True)),
                "height_px": int(_number(resolution.get("height_px"), f"{name}.resolution.height_px", positive=True)),
            }
        normalized_cameras.append(item)

    normalized_targets: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        name = f"targets[{index}]"
        if not isinstance(target, Mapping):
            raise ValueError(f"{name} must be an object")
        item = dict(target)
        if not str(item.get("id", "")).strip():
            raise ValueError(f"{name}.id is required")
        item["id"] = str(item["id"])
        item["x_m"] = _number(item.get("x_m"), f"{name}.x_m")
        item["y_m"] = _number(item.get("y_m"), f"{name}.y_m")
        item["facing_deg"] = _optional_number(item.get("facing_deg"), f"{name}.facing_deg")
        item["person_width_m"] = _optional_number(item.get("person_width_m"), f"{name}.person_width_m", positive=True)
        normalized_targets.append(item)

    normalized = dict(spec)
    normalized["room"] = {**dict(room), "width_m": width, "length_m": length, "ceiling_height_m": ceiling}
    normalized["cameras"] = normalized_cameras
    normalized["targets"] = normalized_targets
    normalized["obstacles"] = [_normalise_obstacle(item, index) for index, item in enumerate(obstacles)]
    normalized.setdefault("spec_status", "UNSPECIFIED_MEASUREMENTS")
    normalized["planning_estimate"] = PLANNING_ESTIMATE
    return normalized


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = current
        x2, y2 = previous
        crosses = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1)
        if crosses:
            inside = not inside
        previous = current
    return inside


def _orientation(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    eps = 1e-9
    if abs(o1) < eps and min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps:
        return True
    if abs(o2) < eps and min(a[0], b[0]) - eps <= d[0] <= max(a[0], b[0]) + eps and min(a[1], b[1]) - eps <= d[1] <= max(a[1], b[1]) + eps:
        return True
    if abs(o3) < eps and min(c[0], d[0]) - eps <= a[0] <= max(c[0], d[0]) + eps and min(c[1], d[1]) - eps <= a[1] <= max(c[1], d[1]) + eps:
        return True
    if abs(o4) < eps and min(c[0], d[0]) - eps <= b[0] <= max(c[0], d[0]) + eps and min(c[1], d[1]) - eps <= b[1] <= max(c[1], d[1]) + eps:
        return True
    return ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0))


def _is_occluded(camera: Mapping[str, Any], target: Mapping[str, Any], obstacles: Sequence[Mapping[str, Any]]) -> list[str]:
    start = (_number(camera["x_m"], "camera.x_m"), _number(camera["y_m"], "camera.y_m"))
    end = (_number(target["x_m"], "target.x_m"), _number(target["y_m"], "target.y_m"))
    blocked: list[str] = []
    for obstacle in obstacles:
        polygon = obstacle["polygon"]
        if _point_in_polygon(start, polygon) or _point_in_polygon(end, polygon):
            blocked.append(str(obstacle["id"]))
            continue
        edges = zip(polygon, polygon[1:] + polygon[:1])
        if any(_segments_intersect(start, end, first, second) for first, second in edges):
            blocked.append(str(obstacle["id"]))
    return blocked


def floor_coverage_wedge(camera: Mapping[str, Any], room: Mapping[str, Any]) -> list[list[float]]:
    """Return a schematic floor wedge, not a ray-traced optical footprint."""

    hfov = camera.get("hfov_deg")
    if hfov is None:
        return []
    room_diagonal = math.hypot(_number(room["width_m"], "room.width_m"), _number(room["length_m"], "room.length_m"))
    max_distance = camera.get("max_useful_distance_m") or room_diagonal
    max_distance = _number(max_distance, "max_useful_distance_m", positive=True)
    direction = float(camera["direction_deg"])
    points = []
    for angle in (direction - float(hfov) / 2.0, direction + float(hfov) / 2.0):
        vector = camera_view_direction(angle)
        points.append([float(camera["x_m"]) + max_distance * vector["x"], float(camera["y_m"]) + max_distance * vector["y"]])
    return [[float(camera["x_m"]), float(camera["y_m"])], points[0], points[1]]


def target_in_camera_coverage(camera: Mapping[str, Any], target: Mapping[str, Any], obstacles: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    distance = distance_camera_to_target(camera, target)
    dx = float(target["x_m"]) - float(camera["x_m"])
    dy = float(target["y_m"]) - float(camera["y_m"])
    bearing = math.degrees(math.atan2(dy, dx)) % 360.0
    direction_offset = angle_difference_deg(bearing, float(camera["direction_deg"]))
    hfov = camera.get("hfov_deg")
    in_horizontal = None if hfov is None else abs(direction_offset) <= float(hfov) / 2.0
    min_distance = camera.get("min_useful_distance_m")
    max_distance = camera.get("max_useful_distance_m")
    distance_ok = (min_distance is None or distance >= float(min_distance)) and (max_distance is None or distance <= float(max_distance))
    blocked_by = _is_occluded(camera, target, obstacles)
    coverage_width = None if hfov is None else horizontal_coverage_width(distance, float(hfov))
    vfov = camera.get("vfov_deg")
    coverage_height = None if vfov is None else vertical_coverage_height(distance, float(vfov))
    facing = target.get("facing_deg")
    front_angle = None if facing is None else abs(angle_difference_deg((bearing + 180.0) % 360.0, float(facing)))
    resolution = camera.get("resolution") or {}
    person_width = target.get("person_width_m")
    estimated_width_px = None
    if person_width is not None and coverage_width and resolution.get("width_px"):
        estimated_width_px = float(resolution["width_px"]) * float(person_width) / coverage_width
    visible = bool(in_horizontal is True and distance_ok and not blocked_by)
    score = 0.0
    score += 50.0 if in_horizontal is True else 0.0
    score += 15.0 if distance_ok else 0.0
    score += 15.0 if not blocked_by else 0.0
    if front_angle is not None:
        score += max(0.0, 20.0 * (1.0 - min(front_angle, 90.0) / 90.0))
    if estimated_width_px is not None:
        score += min(10.0, max(0.0, estimated_width_px / 20.0))
    return {
        "camera_id": str(camera["id"]),
        "target_id": str(target["id"]),
        "distance_m": distance,
        "bearing_deg": bearing,
        "camera_direction_offset_deg": direction_offset,
        "front_view_angle_deg": front_angle,
        "within_horizontal_fov": in_horizontal,
        "distance_ok": distance_ok,
        "blocked_by": blocked_by,
        "horizontal_coverage_width_m": coverage_width,
        "vertical_coverage_height_m": coverage_height,
        "estimated_person_width_px": estimated_width_px,
        "visible_estimate": visible,
        "placement_score": round(score, 3),
        "planning_estimate": PLANNING_ESTIMATE,
    }


def plan_room(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate all camera/target pairs and provide a heuristic candidate order."""

    validated = validate_room_spec(spec)
    room = validated["room"]
    cameras = validated["cameras"]
    targets = validated["targets"]
    obstacles = validated["obstacles"]
    camera_results = []
    for camera in cameras:
        target_results = [target_in_camera_coverage(camera, target, obstacles) for target in targets]
        visible_targets = [item["target_id"] for item in target_results if item["visible_estimate"]]
        camera_results.append({
            "id": camera["id"],
            "position_m": {"x": camera["x_m"], "y": camera["y_m"], "height": camera["height_m"]},
            "direction": camera_view_direction(camera["direction_deg"]),
            "floor_wedge": floor_coverage_wedge(camera, room),
            "target_results": target_results,
            "visible_target_ids": visible_targets,
            "blind_spot_target_ids": [item["target_id"] for item in target_results if not item["visible_estimate"]],
        })
    target_coverage = []
    for target in targets:
        matches = [
            item
            for result in camera_results
            for item in result["target_results"]
            if item["target_id"] == target["id"] and item["visible_estimate"]
        ]
        target_coverage.append({"target_id": target["id"], "camera_ids": [item["camera_id"] for item in matches], "covered": bool(matches)})

    remaining = {item["target_id"] for item in target_coverage if not item["covered"]}
    greedy: list[str] = []
    unused = list(camera_results)
    while remaining and unused:
        best = max(unused, key=lambda item: len(remaining.intersection(item["visible_target_ids"])))
        covered = remaining.intersection(best["visible_target_ids"])
        if not covered:
            break
        greedy.append(str(best["id"]))
        remaining -= covered
        unused.remove(best)
    score_order = [str(item["id"]) for item in sorted(camera_results, key=lambda item: max((pair["placement_score"] for pair in item["target_results"]), default=0.0), reverse=True)]
    return {
        "spec_status": validated.get("spec_status"),
        "planning_estimate": PLANNING_ESTIMATE,
        "room": room,
        "cameras": camera_results,
        "targets": targets,
        "obstacles": obstacles,
        "target_coverage": target_coverage,
        "uncovered_target_ids": [item["target_id"] for item in target_coverage if not item["covered"]],
        "blind_spot_candidates": [
            {"target_id": item["target_id"], "reason": "No candidate camera currently provides an unblocked in-FOV estimate."}
            for item in target_coverage if not item["covered"]
        ],
        "greedy_camera_set_estimate": greedy,
        "camera_score_order": score_order,
        "camera_count_decision": "PENDING_REAL_FLOOR_PLAN_REVIEW",
        "limitations": [
            "FOV math is a geometric planning estimate, not a calibrated optical simulation.",
            "Furniture/people/monitors can create occlusions not represented by the supplied obstacles.",
            "Head/shoulder/hip visibility must be checked with real images and camera tests.",
            "The camera count is not a deployment decision until the real room plan is reviewed.",
        ],
    }


def render_plan_svg(plan: Mapping[str, Any]) -> str:
    """Render a simple top-down schematic for a supplied room specification."""

    room = plan["room"]
    width_m = float(room["width_m"])
    length_m = float(room["length_m"])
    margin = 80.0
    scale = min(900.0 / width_m, 620.0 / length_m)
    svg_width = width_m * scale + 2 * margin
    svg_height = length_m * scale + 2 * margin

    def xy(x: float, y: float) -> tuple[float, float]:
        return margin + float(x) * scale, svg_height - margin - float(y) * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width:.1f} {svg_height:.1f}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Room camera planning estimate</title>",
        f'<desc id="desc">{escape(PLANNING_ESTIMATE)} for a supplied room specification.</desc>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<rect x="{margin:.1f}" y="{margin:.1f}" width="{width_m * scale:.1f}" height="{length_m * scale:.1f}" fill="#ffffff" stroke="#334155" stroke-width="2"/>',
        f'<text x="{margin:.1f}" y="{margin - 24:.1f}" font-family="Arial" font-size="16" fill="#0f172a">{escape(PLANNING_ESTIMATE)}</text>',
    ]
    for obstacle in plan.get("obstacles", []):
        points = " ".join(f"{xy(point[0], point[1])[0]:.1f},{xy(point[0], point[1])[1]:.1f}" for point in obstacle["polygon"])
        parts.append(f'<polygon points="{points}" fill="#94a3b8" fill-opacity="0.55" stroke="#475569" stroke-width="1"/>')
        center_x = sum(point[0] for point in obstacle["polygon"]) / len(obstacle["polygon"])
        center_y = sum(point[1] for point in obstacle["polygon"]) / len(obstacle["polygon"])
        tx, ty = xy(center_x, center_y)
        parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#0f172a">{escape(str(obstacle["id"]))}</text>')
    target_map = {item["target_id"]: item for item in plan.get("target_coverage", [])}
    for target in plan.get("targets", []):
        tx, ty = xy(target["x_m"], target["y_m"])
        covered = bool(target_map.get(target["id"], {}).get("covered"))
        color = "#16a34a" if covered else "#dc2626"
        parts.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="8" fill="{color}"/><text x="{tx + 12:.1f}" y="{ty + 4:.1f}" font-family="Arial" font-size="12" fill="#0f172a">{escape(str(target["id"]))}</text>')
    for camera in plan.get("cameras", []):
        cx, cy = xy(camera["position_m"]["x"], camera["position_m"]["y"])
        wedge = camera.get("floor_wedge") or []
        if len(wedge) == 3:
            points = " ".join(f"{xy(point[0], point[1])[0]:.1f},{xy(point[0], point[1])[1]:.1f}" for point in wedge)
            parts.append(f'<polygon points="{points}" fill="#2563eb" fill-opacity="0.12" stroke="#2563eb" stroke-width="1"/>')
        direction = camera["direction"]
        end = xy(camera["position_m"]["x"] + 0.7 * direction["x"], camera["position_m"]["y"] + 0.7 * direction["y"])
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="#2563eb"/><line x1="{cx:.1f}" y1="{cy:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" stroke="#1d4ed8" stroke-width="3"/><text x="{cx + 10:.1f}" y="{cy - 10:.1f}" font-family="Arial" font-size="12" fill="#0f172a">{escape(str(camera["id"]))}</text>')
    parts.extend([
        f'<text x="{margin:.1f}" y="{svg_height - 22:.1f}" font-family="Arial" font-size="12" fill="#0f172a">green = estimated covered target · red = uncovered target · blue = camera/FOV estimate</text>',
        "</svg>",
    ])
    return "\n".join(parts)
