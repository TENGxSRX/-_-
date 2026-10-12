"""Isolated inference contract for the P001 five-class webcam experiment."""
from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np

from src.posture_ai.features.upper_body_posture_v2 import FEATURES, build_features
from src.posture_ai.inference.upper_body_posture_v2 import evidence_xyz

MODEL_ID = "webcam_p001_fiveclass_exp_v1"
FEATURE_SCHEMA = "upper_body_posture_v2:NO_HIP"
LABELS = ("TUP", "TLF", "TLB", "TLL", "TLR")
LABELS_TH = {
    "TUP": "นั่งตรง",
    "TLF": "เอนไปข้างหน้า",
    "TLB": "เอนไปข้างหลัง",
    "TLL": "เอียงซ้าย",
    "TLR": "เอียงขวา",
    "UNKNOWN": "ยังไม่มั่นใจ",
}


class WebcamFiveClassExperimentalRunner:
    """Load the versioned artifact without touching V4 or the default V2 model."""

    def __init__(self, model_dir: str | Path = "outputs/models/webcam_p001_fiveclass_exp_v1"):
        path = Path(model_dir)
        required = ("model.joblib", "metadata.json", "feature_config.json", "labels.json")
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"FIVECLASS_ARTIFACT_MISSING:{','.join(missing)}")
        self.bundle = joblib.load(path / "model.joblib")
        self.metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        feature_config = json.loads((path / "feature_config.json").read_text(encoding="utf-8"))
        labels_config = json.loads((path / "labels.json").read_text(encoding="utf-8"))
        expected_features = tuple(FEATURES["NO_HIP"])
        if (
            self.bundle.get("experiment") != MODEL_ID
            or self.bundle.get("feature_schema") != FEATURE_SCHEMA
            or tuple(self.bundle.get("feature_names", ())) != expected_features
            or feature_config.get("schema") != FEATURE_SCHEMA
            or tuple(feature_config.get("features", ())) != expected_features
            or tuple(self.bundle.get("labels", ())) != LABELS
            or tuple(labels_config.get("labels", ())) != LABELS
            or labels_config.get("unknown") != "UNKNOWN"
            or self.metadata.get("status") != "EXPERIMENTAL_ONLY"
        ):
            raise ValueError("FIVECLASS_ARTIFACT_SCHEMA_MISMATCH")
        self.model = self.bundle.get("model")
        self.threshold = float(self.bundle.get("unknown_threshold", math.nan))
        classes = tuple(str(value) for value in getattr(self.model, "classes_", ()))
        if not 0.0 < self.threshold < 1.0 or set(classes) != set(LABELS) or not hasattr(self.model, "predict_proba"):
            raise ValueError("FIVECLASS_ARTIFACT_MODEL_MISMATCH")
        self.classes = np.asarray(classes, dtype=object)

    @staticmethod
    def _unknown(reason: str, confidence: float | None = None) -> dict:
        return {"output": "UNKNOWN", "confidence": confidence, "reason": reason, "model_id": MODEL_ID}

    def predict(self, evidence) -> dict:
        if evidence.pose_state != "POSE_OK" or not evidence.core_valid:
            return self._unknown(str(evidence.pose_state))
        try:
            features = build_features(evidence_xyz(evidence), "NO_HIP")
        except (TypeError, ValueError):
            return self._unknown("INVALID_NOSE_OR_SHOULDERS")
        if features.shape != (1, len(FEATURES["NO_HIP"])) or not np.isfinite(features).all():
            return self._unknown("INVALID_FEATURES")
        probabilities = np.asarray(self.model.predict_proba(features), dtype=float)
        if probabilities.shape != (1, len(self.classes)) or not np.isfinite(probabilities).all():
            return self._unknown("INVALID_PROBABILITIES")
        best = int(probabilities[0].argmax())
        label, confidence = str(self.classes[best]), float(probabilities[0, best])
        if confidence < self.threshold:
            return self._unknown("LOW_CONFIDENCE_EXPERIMENTAL", confidence)
        return {"output": label, "confidence": confidence, "reason": "EXPERIMENTAL_P001_CALIBRATION", "model_id": MODEL_ID}


__all__ = ["FEATURE_SCHEMA", "LABELS", "LABELS_TH", "MODEL_ID", "WebcamFiveClassExperimentalRunner"]
