import json
from pathlib import Path

import joblib
import numpy as np

from scripts.train_webcam_p001_fiveclass_exp_v1 import (
    LABELS, FEATURE_SCHEMA, apply_unknown, feature_parity, load_human_gt,
    threshold_metrics,
)


def test_human_gt_loader_requires_explicit_provenance_and_exact_schema():
    records, audit = load_human_gt()
    assert len(records) == 257
    assert audit["label_source"] == ["HUMAN"]
    assert set(audit["labels"]) == set(LABELS)
    assert audit["duplicate_feature_rows"] == 0
    assert all(record["feature_schema_version"] == FEATURE_SCHEMA for record in records)


def test_stored_features_reproduce_runtime_feature_builder():
    records, _ = load_human_gt()
    parity = feature_parity(records)
    assert parity["decision"] == "RUNTIME_FEATURE_PARITY_CONFIRMED"
    assert parity["max_abs_difference"] <= 1e-9


def test_unknown_threshold_is_explicit_and_does_not_change_labels():
    probabilities = np.array([[0.90, 0.05, 0.02, 0.02, 0.01], [0.40, 0.30, 0.10, 0.10, 0.10]])
    prediction = apply_unknown(probabilities, np.asarray(LABELS), 0.50)
    assert prediction.tolist() == ["TUP", "UNKNOWN"]
    result = threshold_metrics(np.asarray(["TUP", "TLF"]), probabilities, np.asarray(LABELS), 0.50)
    assert result["unknown_count"] == 1
    assert result["coverage"] == 0.5


def test_no_segment_overlap_in_available_two_session_split():
    records, _ = load_human_gt()
    sessions = sorted({record["session_id"] for record in records})
    holdout = "SESSION_20260919T035749Z_P001"
    dev = {record["segment_id"] for record in records if record["session_id"] != holdout}
    test = {record["segment_id"] for record in records if record["session_id"] == holdout}
    assert len(sessions) == 2
    assert dev.isdisjoint(test)


def test_versioned_artifact_reloads_with_explicit_unknown_policy():
    bundle = joblib.load(Path("outputs/models/webcam_p001_fiveclass_exp_v1/model.joblib"))
    assert bundle["feature_schema"] == FEATURE_SCHEMA
    assert tuple(bundle["feature_names"]) == tuple(__import__(
        "src.posture_ai.features.upper_body_posture_v2", fromlist=["FEATURES"]
    ).FEATURES["NO_HIP"])
    assert bundle["labels"] == list(LABELS)
    assert bundle["unknown_threshold"] == 0.5
