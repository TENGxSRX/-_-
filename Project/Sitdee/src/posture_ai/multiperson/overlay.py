"""Readable anonymous multi-person overlay and dashboard-card projection."""

from __future__ import annotations

from typing import Any

import cv2

from .schema import MultiPersonFrameResult, PoseSubject


SKELETON_CONNECTIONS = (
    (0, 11), (0, 12), (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
)


def dashboard_cards(result: MultiPersonFrameResult) -> list[dict[str, Any]]:
    """Return one clearly separated card model per anonymous subject."""

    cards = []
    for subject in result.subjects:
        directional = subject.geometry.get("directional_indicators", {})
        temporal = subject.temporal_state
        activity = subject.activity_state
        if temporal.get("risk_level") == "POSE_LOST":
            display_state = "POSE_LOST"
        elif activity in {"STANDING", "UNKNOWN"}:
            display_state = "MONITOR_PAUSED"
        else:
            display_state = temporal.get("risk_level", "POSE_LOST")
        cards.append({
            "subject_id": subject.subject_id,
            "activity_state": activity,
            "activity_raw_state": subject.activity_raw_state,
            "activity_confidence": subject.activity_confidence,
            "activity_method": subject.activity_method,
            "ai_posture": subject.v4_prediction if activity == "SITTING" else None,
            "ai_probabilities": subject.v4_probabilities or {},
            "geometry_indicators": {
                "forward": bool(directional.get("forward", False)),
                "left": bool(directional.get("left", False)),
                "right": bool(directional.get("right", False)),
                "shoulder_tilt": subject.geometry.get("shoulder_tilt_deg"),
            },
            "state": display_state,
            "duration_seconds": temporal.get("duration_seconds", 0.0),
            "pause_reason": temporal.get("pause_reason"),
            "pose_quality": subject.pose_quality,
            "tracking_state": subject.tracking_state,
        })
    return cards


def _pixel(point, width: int, height: int) -> tuple[int, int] | None:
    if point is None or not point.present:
        return None
    x = int(max(0.0, min(1.0, float(point.x))) * width)
    y = int(max(0.0, min(1.0, float(point.y))) * height)
    return x, y


def draw_multiperson_overlay(frame, result: MultiPersonFrameResult, *, debug: bool = False):
    """Draw anonymous skeletons/cards without opening windows or cameras."""

    output = frame.copy()
    height, width = output.shape[:2]
    for subject in result.subjects:
        card = next(item for item in dashboard_cards(result) if item["subject_id"] == subject.subject_id)
        state = str(card["state"])
        color = (60, 190, 60) if state == "NORMAL" else (0, 180, 255)
        if state in {"WARNING", "HIGH_DURATION_RISK"}:
            color = (0, 0, 255)
        points = [_pixel(point, width, height) for point in subject.landmarks.points]
        for start, end in SKELETON_CONNECTIONS:
            first, second = points[start], points[end]
            if first is not None and second is not None:
                cv2.line(output, first, second, color, 2, cv2.LINE_AA)
        for point in points:
            if point is not None:
                cv2.circle(output, point, 3, color, -1, cv2.LINE_AA)
        if subject.bbox is not None:
            left, top, right, bottom = subject.bbox
            x1, y1 = int(max(0.0, min(1.0, left)) * width), int(max(0.0, min(1.0, top)) * height)
            x2, y2 = int(max(0.0, min(1.0, right)) * width), int(max(0.0, min(1.0, bottom)) * height)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label_x, label_y = x1, max(22, y1 - 8)
        elif subject.center is not None:
            label_x, label_y = int(subject.center[0] * width), int(subject.center[1] * height)
        else:
            label_x, label_y = 10, 30
        activity = card["activity_state"]
        if activity == "STANDING":
            lines = (f"{subject.subject_id}  STANDING", "POSTURE MONITOR PAUSED")
        elif activity == "UNKNOWN":
            lines = (f"{subject.subject_id}  UNKNOWN", "ACTIVITY UNCERTAIN | MONITOR PAUSED")
        else:
            probabilities = card["ai_probabilities"]
            confidence = float(probabilities.get(card["ai_posture"], 0.0)) if card["ai_posture"] else 0.0
            lines = (
                f"{subject.subject_id}  SITTING | AI: {card['ai_posture'] or '—'} {confidence:.0%}",
                f"{state}  POOR_DURATION: {float(card['duration_seconds'] or 0.0):.1f}s",
            )
        if debug:
            directional = card["geometry_indicators"]
            geometry_text = "F:{} L:{} R:{}".format(
                "Y" if directional["forward"] else "N",
                "Y" if directional["left"] else "N",
                "Y" if directional["right"] else "N",
            )
            lines = (*lines, f"DEBUG G:{geometry_text} Q:{subject.pose_quality:.2f} TRACK:{subject.tracking_state}")
        for offset, line in enumerate(lines):
            y = label_y + offset * 19
            cv2.putText(output, line, (label_x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(output, line, (label_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return output
