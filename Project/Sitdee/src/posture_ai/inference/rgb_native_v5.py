"""Offline RGB-native V5 model runner.

The runner is deliberately separate from V4.  It accepts only the frozen
World Landmark representation and refuses a missing or mismatched artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.posture_ai.features.rgb_native_v5 import (
    V5_CORE_LANDMARKS,
    build_rgb_native_v5_features,
)


class RGBNativeV5Runner:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "model.joblib")
        self.feature_config = json.loads((self.model_dir / "feature_config.json").read_text(encoding="utf-8"))
        self.labels = json.loads((self.model_dir / "labels.json").read_text(encoding="utf-8"))["source_labels"]
        self.family = str(self.feature_config["name"])
        self.expected_feature_names = list(self.feature_config["columns"])
        if list(self.feature_config.get("core_landmarks", [])) != list(V5_CORE_LANDMARKS):
            raise ValueError("RGB-native V5 artifact has an incompatible core landmark contract")

    def predict_world_landmarks(self, world_landmarks, *, image_visibility=None, visibility_threshold: float = 0.5) -> dict:
        if image_visibility is not None:
            visibility = np.asarray(image_visibility, dtype=float)
            if visibility.shape != (33,):
                raise ValueError("image_visibility must contain 33 MediaPipe landmarks")
            core_indices = [self._landmark_index(name) for name in V5_CORE_LANDMARKS]
            if np.any(visibility[core_indices] < visibility_threshold):
                raise ValueError("One or more V5 core landmarks are below the visibility threshold")
        features, names = build_rgb_native_v5_features(world_landmarks, self.family)
        if names != self.expected_feature_names:
            raise ValueError("V5 feature adapter output order does not match the exported model schema")
        prediction = self.model.predict(features)[0]
        if hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(features)[0], dtype=float)
        else:
            decision = np.atleast_1d(np.asarray(self.model.decision_function(features))[0])
            probabilities = np.exp(decision - np.max(decision))
            probabilities = probabilities / probabilities.sum()
        by_label = {str(label): float(probabilities[i]) for i, label in enumerate(self.model.classes_)}
        return {
            "label": str(prediction),
            "display_label": {"TUP": "Upright", "TLF": "Lean forward", "TLB": "Lean backward", "TLL": "Lean left", "TLR": "Lean right", "SLOUCH": "Slouch"}.get(str(prediction), str(prediction)),
            "confidence": float(np.max(probabilities)),
            "probabilities": by_label,
            "feature_config": self.family,
            "feature_count": int(features.shape[1]),
            "adapter_status": "RGB_NATIVE_V5_FEATURE_PIPELINE",
            "adapter_source_convention": "MediaPipe Pose World Landmarks -> hip-centered V4-compatible features",
        }

    @staticmethod
    def _landmark_index(name: str) -> int:
        from src.posture_ai.pose.landmarks import LANDMARK_INDEX

        return LANDMARK_INDEX[name]


__all__ = ["RGBNativeV5Runner"]
