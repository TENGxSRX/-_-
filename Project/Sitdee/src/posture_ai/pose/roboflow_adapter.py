"""Parse Roboflow keypoint responses without coupling the core to its SDK."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .landmarks import LANDMARK_NAMES, PoseLandmarks


DEFAULT_KEYPOINT_NAMES = (
    "nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"
)


def _keypoint_items(prediction: Mapping[str, Any]) -> Sequence[Any]:
    value = prediction.get("keypoints", ())
    if isinstance(value, Mapping):
        return [{"name": name, **(point if isinstance(point, Mapping) else {})}
                for name, point in value.items()]
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def parse_roboflow_pose(
    payload: Mapping[str, Any], *, confidence: float = 0.35,
    keypoint_names: Sequence[str] = DEFAULT_KEYPOINT_NAMES,
) -> PoseLandmarks | None:
    """Return the highest-confidence person as normalized 2D canonical points.

    Roboflow image coordinates are converted to 0..1 and z is deliberately set
    to 0.  This representation is for a Roboflow-specific 2D model only; it must
    never be fed to a classifier trained on MediaPipe world coordinates.
    """
    image = payload.get("image") or {}
    width, height = float(image.get("width") or 0), float(image.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("ROBOFLOW_IMAGE_SIZE_MISSING")
    predictions = payload.get("predictions") or ()
    candidates = [p for p in predictions if isinstance(p, Mapping)]
    if not candidates:
        return None
    prediction = max(candidates, key=lambda p: float(p.get("confidence") or 0))
    mapping: dict[str, dict[str, float]] = {}
    for index, point in enumerate(_keypoint_items(prediction)):
        if not isinstance(point, Mapping):
            continue
        name = point.get("name") or point.get("class")
        if name is None and index < len(keypoint_names):
            name = keypoint_names[index]
        if name not in LANDMARK_NAMES:
            continue
        score = float(point.get("confidence", point.get("score", 1.0)) or 0)
        x, y = point.get("x"), point.get("y")
        if x is None or y is None or score < confidence:
            continue
        mapping[str(name)] = {
            "x": float(x) / width, "y": float(y) / height,
            "z": 0.0, "visibility": score,
        }
    return PoseLandmarks.from_mapping(mapping)

