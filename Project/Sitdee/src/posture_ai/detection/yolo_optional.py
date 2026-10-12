"""Optional Ultralytics YOLO adapters for controlled benchmark work.

The project does not import Ultralytics at module import time.  The existing
MediaPipe runtime remains the default.  YOLO person detection can provide
boxes for a crop-then-MediaPipe candidate, while YOLO pose keypoints remain a
separate representation until an explicit landmark adapter is proven.
"""

from __future__ import annotations

import importlib.util
import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.posture_ai.pose.landmarks import PoseLandmarks


COCO17_KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


class YOLOUnavailableError(RuntimeError):
    """Raised when an optional YOLO backend cannot be constructed."""


@dataclass(frozen=True)
class YoloEnvironment:
    available: bool
    package: str = "ultralytics"
    version: str | None = None
    reason: str | None = None


def inspect_yolo_dependency() -> YoloEnvironment:
    """Inspect the optional dependency without importing or installing it."""

    if importlib.util.find_spec("ultralytics") is None:
        return YoloEnvironment(False, reason="ULTRALYTICS_NOT_INSTALLED")
    try:
        version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        return YoloEnvironment(False, reason="ULTRALYTICS_METADATA_NOT_FOUND")
    return YoloEnvironment(True, version=version)


@dataclass(frozen=True)
class PersonBox:
    """One detector box in full-frame pixel coordinates.

    ``track_id`` is an anonymous, runtime-only tracker value supplied by
    Ultralytics/ByteTrack when tracking is used.  It is deliberately not a
    person identity and is never persisted as a real-world identity.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    track_id: int | None = None

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in (self.x1, self.y1, self.x2, self.y2, self.confidence))
        if not np.isfinite(values).all():
            raise ValueError("PersonBox values must be finite")
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ValueError("PersonBox must have positive width and height")
        if not 0.0 <= values[4] <= 1.0:
            raise ValueError("PersonBox confidence must be between 0 and 1")
        object.__setattr__(self, "x1", values[0])
        object.__setattr__(self, "y1", values[1])
        object.__setattr__(self, "x2", values[2])
        object.__setattr__(self, "y2", values[3])
        object.__setattr__(self, "confidence", values[4])
        object.__setattr__(self, "class_id", int(self.class_id))
        track = None if self.track_id is None else int(self.track_id)
        object.__setattr__(self, "track_id", track)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def clipped(self, frame_width: int, frame_height: int) -> "PersonBox":
        x1 = max(0.0, min(float(frame_width - 1), self.x1))
        y1 = max(0.0, min(float(frame_height - 1), self.y1))
        x2 = max(x1 + 1.0, min(float(frame_width), self.x2))
        y2 = max(y1 + 1.0, min(float(frame_height), self.y2))
        return PersonBox(x1, y1, x2, y2, self.confidence, self.class_id, self.track_id)

    def normalized(self, frame_width: int, frame_height: int) -> tuple[float, float, float, float]:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        return (
            self.x1 / frame_width,
            self.y1 / frame_height,
            self.x2 / frame_width,
            self.y2 / frame_height,
        )


def map_crop_landmarks_to_full_frame(
    crop_landmarks: PoseLandmarks,
    crop_box: PersonBox,
    *,
    frame_width: int,
    frame_height: int,
) -> PoseLandmarks:
    """Map normalized crop image landmarks back without aspect-ratio distortion.

    Image x/y coordinates are mapped into the full-frame normalized space.  The
    crop-relative z value is deliberately preserved but marked by the caller as
    crop-relative; it must not be sent to the frozen V4 world-landmark feature
    builder without a separately validated semantic adapter.
    """

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    points = []
    for point in crop_landmarks.points:
        if point is None or not point.present:
            points.append(point)
            continue
        x_pixel = crop_box.x1 + float(point.x) * crop_box.width
        y_pixel = crop_box.y1 + float(point.y) * crop_box.height
        points.append(type(point)(
            x_pixel / frame_width,
            y_pixel / frame_height,
            point.z,
            point.visibility,
        ))
    return PoseLandmarks(tuple(points))


class UltralyticsPersonDetector:
    """Lazy Ultralytics person detector for candidate Stack B.

    It requires an explicitly supplied local weight file.  It never downloads
    weights implicitly, and it only returns COCO class 0 (person) boxes.
    """

    def __init__(self, weights: str | Path, *, device: str | None = None, confidence: float = 0.25, iou: float = 0.7):
        environment = inspect_yolo_dependency()
        if not environment.available:
            raise YOLOUnavailableError(environment.reason or "ULTRALYTICS_UNAVAILABLE")
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(f"YOLO weight file does not exist: {self.weights}")
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency branch
            raise YOLOUnavailableError(f"ULTRALYTICS_IMPORT_FAILED:{type(exc).__name__}") from exc
        self.model = YOLO(str(self.weights))
        self.device = device
        self.confidence = float(confidence)
        self.iou = float(iou)
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.iou <= 1.0:
            raise ValueError("confidence and iou must be between 0 and 1")

    def detect(self, frame_bgr: Any) -> list[PersonBox]:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise ValueError("frame must be a non-empty image")
        results = self.model.predict(
            source=frame_bgr,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidence = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        output = []
        for coordinates, score, class_id in zip(xyxy, confidence, classes):
            if int(class_id) != 0:
                continue
            output.append(PersonBox(*coordinates.tolist(), float(score), int(class_id)))
        return output


class UltralyticsPersonTracker(UltralyticsPersonDetector):
    """Optional COCO-person tracker backed by Ultralytics tracking.

    This class is intentionally lazy and requires a pre-existing local weight
    file, matching :class:`UltralyticsPersonDetector`.  It does not download
    weights while a collection session is running.  Returned ``track_id``
    values are only short-lived anonymous association hints; a collector must
    still pause when target association is ambiguous.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str | None = None,
        confidence: float = 0.25,
        iou: float = 0.7,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        super().__init__(weights, device=device, confidence=confidence, iou=iou)
        tracker_name = str(tracker).strip()
        if not tracker_name:
            raise ValueError("tracker must be a non-empty Ultralytics tracker configuration")
        self.tracker = tracker_name

    def track(self, frame_bgr: Any) -> list[PersonBox]:
        """Return only COCO ``person`` boxes, with tracker IDs when exposed."""

        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise ValueError("frame must be a non-empty image")
        results = self.model.track(
            source=frame_bgr,
            classes=[0],
            conf=self.confidence,
            iou=self.iou,
            tracker=self.tracker,
            persist=True,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidence = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        raw_ids = getattr(boxes, "id", None)
        track_ids = [None] * len(xyxy) if raw_ids is None else raw_ids.detach().cpu().numpy().astype(int).tolist()
        output: list[PersonBox] = []
        for coordinates, score, class_id, track_id in zip(xyxy, confidence, classes, track_ids):
            if int(class_id) != 0:
                continue
            output.append(PersonBox(*coordinates.tolist(), float(score), int(class_id), None if track_id is None else int(track_id)))
        return output


def crop_frame(frame_bgr: Any, box: PersonBox) -> tuple[Any, PersonBox]:
    """Return an aspect-preserving pixel crop and its clipped box."""

    if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3:
        raise ValueError("frame must be a color image")
    height, width = frame_bgr.shape[:2]
    clipped = box.clipped(width, height)
    x1, y1 = int(np.floor(clipped.x1)), int(np.floor(clipped.y1))
    x2, y2 = int(np.ceil(clipped.x2)), int(np.ceil(clipped.y2))
    return frame_bgr[y1:y2, x1:x2].copy(), clipped


__all__ = [
    "COCO17_KEYPOINT_NAMES",
    "PersonBox",
    "YOLOUnavailableError",
    "YoloEnvironment",
    "UltralyticsPersonDetector",
    "UltralyticsPersonTracker",
    "crop_frame",
    "inspect_yolo_dependency",
    "map_crop_landmarks_to_full_frame",
]
