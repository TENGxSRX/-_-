import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.posture_ai.data.anya import (
    detect_anya_grouping,
    detect_anya_schema,
    detect_anya_target,
    iter_anya_csv,
    sha256_file,
)
from src.posture_ai.features.pose_features import expected_feature_columns
from src.posture_ai.inference.pipeline import InferenceRunner


ROOT = Path(__file__).resolve().parents[1]
ANYA_CSV = ROOT / "datasets" / "raw" / "anya" / "official-download" / "ANN.csv"


def test_anya_chunked_loading_and_schema_detection():
    assert ANYA_CSV.is_file()
    header = pd.read_csv(ANYA_CSV, nrows=0, sep=",", encoding="utf-8").columns.tolist()
    schema = detect_anya_schema(header)
    assert schema["column_count"] == 176
    assert schema["raw_coordinate_complete"] is True
    assert schema["raw_landmark_count"] == 33
    assert schema["visibility_indices"] == list(range(33))
    assert schema["normalized_landmark_count"] == 13
    first = next(iter_anya_csv(ANYA_CSV, chunksize=7, usecols=["frame", "filename", "Predicted_Label"]))
    assert len(first) == 7
    assert list(first.columns) == ["frame", "filename", "Predicted_Label"]


def test_anya_target_detection_does_not_promote_predictions():
    columns = ["frame", "filename", "clusterid", "normclusterid", "Predicted_Label"]
    target = detect_anya_target(columns)
    assert target["verified_ground_truth_columns"] == []
    assert target["prediction_output_columns"] == ["Predicted_Label"]
    assert target["cluster_id_columns"] == ["clusterid", "normclusterid"]
    assert target["status"] == "no_verified_ground_truth_column"


def test_anya_grouping_detection_is_conservative():
    grouping = detect_anya_grouping(["frame", "filename", "clusterid"])
    assert grouping["participant_columns"] == []
    assert grouping["video_or_recording_columns"] == []
    assert grouping["recommended_group_column"] == "filename"
    assert grouping["participant_independence_proven"] is False


def test_raw_hash_helper_and_manifest_style_verification(tmp_path):
    path = tmp_path / "tiny.csv"
    path.write_bytes(b"a,b\n1,2\n")
    assert sha256_file(path) == "492d5ea496056f1a6a6592241032fab764c321596317930b4fa0e1e8bc3b7470"


def test_model_reload_and_feature_schema_mismatch_rejection(tmp_path):
    model = DummyClassifier(strategy="prior").fit(np.zeros((2, 99)), [0, 1])
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "feature_config.json").write_text(json.dumps({"name": "raw", "columns": expected_feature_columns()[:-1]}), encoding="utf-8")
    (tmp_path / "labels.json").write_text(json.dumps({"source_labels": [0, 1]}), encoding="utf-8")
    runner = InferenceRunner(tmp_path)
    assert hasattr(runner.model, "predict")

    class Landmark:
        x = 0.5
        y = 0.5
        z = 0.0
        visibility = 1.0

    with pytest.raises(ValueError, match="Feature adapter output order"):
        runner.predict_landmarks([Landmark() for _ in range(33)])
