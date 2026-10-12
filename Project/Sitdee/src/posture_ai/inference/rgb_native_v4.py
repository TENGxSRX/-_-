"""Offline inference helper for the frozen RGB-native V4 contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.posture_ai.features.rgb_native_v4 import build_rgb_native_v4_features
from src.posture_ai.pose.landmarks import LANDMARK_INDEX


class RGBNativeV4Runner:
    """Load a V4 artifact and reject mismatched coordinates/schema loudly."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "model.joblib")
        self.feature_config = json.loads((self.model_dir / "feature_config.json").read_text(encoding="utf-8"))
        self.labels = json.loads((self.model_dir / "labels.json").read_text(encoding="utf-8"))["source_labels"]
        self.expected_names = list(self.feature_config["columns"])
        self.family = str(self.feature_config["name"])

    def predict_world_landmarks(self, world_landmarks: Any, *, image_visibility: Any | None = None, visibility_threshold: float = 0.5) -> dict[str, Any]:
        if image_visibility is not None:
            visibility = np.asarray([float(getattr(point, "visibility", point)) for point in image_visibility], dtype=float)
            required = [LANDMARK_INDEX[name] for name in self.feature_config["core_landmarks"]]
            if visibility.shape != (33,) or not np.isfinite(visibility[required]).all() or np.any(visibility[required] < visibility_threshold):
                raise ValueError("RGB-native V4 core landmark visibility is below the configured threshold")
        try:
            raw = np.asarray(world_landmarks, dtype=float)
        except (TypeError, ValueError):
            raw = np.empty((0,))
        if raw.shape in {(33, 3), (99,)}:
            xyz = raw
        else:
            xyz = np.asarray([[float(point.x), float(point.y), float(point.z)] for point in world_landmarks], dtype=float)
        features, names = build_rgb_native_v4_features(xyz, self.family)
        if names != self.expected_names:
            raise ValueError("RGB-native V4 feature schema/order mismatch")
        prediction = str(self.model.predict(features)[0])
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(features)[0], dtype=float)
            classes = list(self.model.classes_)
        else:
            probabilities = np.zeros(len(self.labels), dtype=float)
            classes = self.labels
            probabilities[self.labels.index(prediction)] = 1.0
        probability_by_label = {str(label): float(probabilities[classes.index(label)]) for label in self.labels if label in classes}
        return {
            "label": prediction,
            "display_label": {"TUP": "Upright", "TLB": "Lean backward"}.get(prediction, prediction),
            "confidence": float(np.max(probabilities)),
            "probabilities": probability_by_label,
            "feature_values": [float(value) for value in features[0]],
            "feature_names": list(names),
            "feature_family": self.family,
            "feature_count": int(features.shape[1]),
            "adapter_version": self.feature_config["adapter_version"],
            "adapter_status": "RGB_NATIVE_PIPELINE_MATCHED_FOR_TRAINING_SOURCE",
        }
