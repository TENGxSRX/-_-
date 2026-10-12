"""Hip-optional experimental inference; frozen V1/V4 are never rewritten."""
from pathlib import Path

import joblib
import numpy as np

from src.posture_ai.features.upper_body_posture_v1 import SUBTYPES
from src.posture_ai.features.upper_body_posture_v2 import CORE_NAMES, FEATURES, INDICES, SCHEMA, build_features
from src.posture_ai.inference.upper_body_posture_v1 import verify_v4


def classify_batch(bundle, features):
    x = np.asarray(features, dtype=float)
    unknown = {"output": "UNKNOWN", "subtype": None}
    if x.ndim != 2 or x.shape[1] != len(bundle["features"]):
        raise ValueError("INVALID_FEATURE_SHAPE")
    records = [{**unknown, "reason": "INVALID_FEATURES"} for _ in x]
    active = np.ones(x.shape, dtype=bool)
    if bundle["condition"] == "HIP_DROPOUT_ROBUST":
        active[x[:, -1] == 0, 6:-1] = False
    # ponytail: marginal box misses joint OOD; replace only with validated CCTV evidence.
    finite = np.isfinite(x).all(axis=1)
    ood = (((x < bundle["minimum"]) | (x > bundle["maximum"])) & active).any(axis=1)
    for i in np.flatnonzero(finite & ood):
        records[i] = {**unknown, "reason": "POSSIBLE_OOD_EXPERIMENTAL"}
    indices = np.flatnonzero(finite & ~ood)
    if len(indices):
        votes = np.column_stack([m.predict(x[indices]) for m in [bundle["model"], *bundle["committee"]]])
        for i, row in zip(indices, votes):
            label = str(row[0])
            if np.any(row != row[0]):
                records[i] = {**unknown, "reason": "MODEL_DISAGREEMENT_EXPERIMENTAL"}
            elif label not in SUBTYPES or label not in bundle["labels"]:
                records[i] = {**unknown, "reason": "UNSUPPORTED_LABEL"}
            else:
                records[i] = {"output": "CORRECT_SITTING" if label == "TUP" else "INCORRECT_SITTING",
                              "subtype": SUBTYPES[label], "reason": "EXPERIMENTAL_SOURCE_DOMAIN"}
    return records


def classify_features(bundle, features):
    if np.asarray(features).shape != (1, len(bundle["features"])):
        return {"output": "UNKNOWN", "subtype": None, "reason": "INVALID_FEATURES"}
    return classify_batch(bundle, features)[0]


def evidence_xyz(evidence):
    """Read only five anchors; missing/non-finite/low-visibility hips are optional."""
    a = np.full((5, 3), np.nan)
    names = (*CORE_NAMES, "left_hip", "right_hip")
    for j, (index, name) in enumerate(zip(INDICES, names)):
        v = evidence.landmark_visibility.get(name)
        if v is None or not np.isfinite(v) or v < 0.5:
            continue
        try:
            point = np.asarray(evidence.world_landmarks_xyz[index], dtype=float)
            if point.shape == (3,):
                a[j] = point
        except (IndexError, TypeError, ValueError):
            pass
    return a


class UpperBodyPostureV2Runner:
    def __init__(self, model_dir="outputs/models/upper_body_posture_v2"):
        verify_v4()
        self.bundle = joblib.load(Path(model_dir) / "candidate.joblib")
        if (self.bundle["schema"] != SCHEMA or self.bundle["condition"] not in ("NO_HIP", "HIP_DROPOUT_ROBUST")
                or tuple(self.bundle["features"]) != FEATURES[self.bundle["condition"]]):
            raise ValueError("UPPER_BODY_V2_SCHEMA_MISMATCH")

    def predict(self, evidence):
        unknown = {"output": "UNKNOWN", "subtype": None, "reason": evidence.pose_state}
        # The live Pose backend MUST use nose/shoulders as its core contract.
        if evidence.pose_state != "POSE_OK" or not evidence.core_valid:
            return unknown
        try:
            return classify_features(self.bundle, build_features(evidence_xyz(evidence), self.bundle["condition"]))
        except (TypeError, ValueError):
            return {**unknown, "reason": "INVALID_NOSE_OR_SHOULDERS"}
