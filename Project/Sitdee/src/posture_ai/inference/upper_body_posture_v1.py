"""Experimental refusal-capable posture inference, independent of frozen V4."""
from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np

from src.posture_ai.features.upper_body_posture_v1 import SCHEMA, FEATURE_NAMES, CORE_NAMES, CORE_INDICES, SUBTYPES, build_features

V4_PATH = Path("outputs/models/rgb_native_v4/model.joblib")
V4_SHA256 = "af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd"


def verify_v4() -> str:
    digest = hashlib.sha256(V4_PATH.read_bytes()).hexdigest()
    if digest != V4_SHA256:
        raise ValueError("FROZEN_V4_HASH_MISMATCH")
    return digest


def classify_features(bundle, features) -> dict:
    x = np.asarray(features, dtype=float)
    if x.shape != (1, len(FEATURE_NAMES)) or not np.isfinite(x).all():
        return {"output": "UNKNOWN", "subtype": None, "reason": "INVALID_FEATURES"}
    # ponytail: marginal range box misses joint OOD; replace after paired CCTV GT validation.
    if np.any(x[0] < bundle["minimum"]) or np.any(x[0] > bundle["maximum"]):
        return {"output": "UNKNOWN", "subtype": None, "reason": "POSSIBLE_OOD_EXPERIMENTAL"}
    votes = [str(model.predict(x)[0]) for model in [bundle["model"], *bundle["committee"]]]
    if len(set(votes)) != 1:
        return {"output": "UNKNOWN", "subtype": None, "reason": "MODEL_DISAGREEMENT_EXPERIMENTAL"}
    label = votes[0]
    if label not in SUBTYPES or label not in bundle["labels"]:
        return {"output": "UNKNOWN", "subtype": None, "reason": "UNSUPPORTED_LABEL"}
    return {"output": "CORRECT_SITTING" if label == "TUP" else "INCORRECT_SITTING",
            "subtype": SUBTYPES[label], "reason": "EXPERIMENTAL_SOURCE_DOMAIN"}


class UpperBodyPostureRunner:
    def __init__(self, model_dir="outputs/models/upper_body_posture_v1_drivenet_exp"):
        verify_v4()
        self.bundle = joblib.load(Path(model_dir) / "candidate.joblib")
        if self.bundle["schema"] != SCHEMA or tuple(self.bundle["features"]) != FEATURE_NAMES:
            raise ValueError("UPPER_BODY_SCHEMA_MISMATCH")

    def predict(self, evidence) -> dict:
        unknown = {"output": "UNKNOWN", "subtype": None, "reason": evidence.pose_state}
        if evidence.pose_state != "POSE_OK" or not evidence.core_valid:
            return unknown
        visibility = [evidence.landmark_visibility.get(name) for name in CORE_NAMES]
        if any(v is None or not np.isfinite(v) or v < 0.5 for v in visibility):
            return {**unknown, "reason": "LOW_CORE_VISIBILITY"}
        try:
            core = [evidence.world_landmarks_xyz[i] for i in CORE_INDICES]
            return classify_features(self.bundle, build_features(core))
        except (IndexError, TypeError, ValueError):
            return {**unknown, "reason": "INVALID_SHOULDERS_OR_HIPS"}
