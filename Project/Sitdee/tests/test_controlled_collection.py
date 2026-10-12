from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from src.posture_ai.data.controlled_collection import (
    CONTROLLED_LABELS,
    CONTROLLED_SCHEMA_VERSION,
    CORE_LANDMARKS,
    ControlledSampleWriter,
    evaluate_collection_quality,
    make_session_id,
    validate_subject_id,
)


def _core_visibility(value: float = 0.9) -> dict[str, dict[str, object]]:
    return {name: {"visible": True, "visibility": value} for name in CORE_LANDMARKS}


def test_manual_label_and_anonymous_subject_contract():
    assert CONTROLLED_LABELS == ("TUP", "TLB")
    assert validate_subject_id("p001") == "P001"
    assert make_session_id("P001", "P001_TUP_01").endswith("_P001_P001_TUP_01")
    with pytest.raises(ValueError, match="anonymous"):
        validate_subject_id("Alice")


def test_quality_gate_requires_core_pose_and_finite_features():
    names = ("f0", "f1", "f2")
    accepted = evaluate_collection_quality(
        pose_detected=True,
        core_visibility=_core_visibility(),
        feature_values=(0.1, 0.2, 0.3),
        feature_names=names,
        frame_width=640,
        frame_height=480,
    )
    assert accepted["accepted"] is True
    rejected = evaluate_collection_quality(
        pose_detected=True,
        core_visibility={**_core_visibility(), "left_hip": {"visible": False, "visibility": 0.1}},
        feature_values=(0.1, np.nan, 0.3),
        feature_names=names,
        frame_width=640,
        frame_height=480,
    )
    assert rejected["accepted"] is False
    assert any(reason.startswith("CORE_INVALID") for reason in rejected["reasons"])
    assert "NONFINITE_V4_FEATURES" in rejected["reasons"]


def test_writer_preserves_feature_order_and_separates_human_label(tmp_path):
    feature_names = ("v4_a", "v4_b")
    writer = ControlledSampleWriter(tmp_path, feature_names=feature_names)
    writer.append_sample(
        sample_id="sample_001",
        session_id="session_001",
        subject_id="P001",
        clip_id="P001_TUP_01",
        human_label="TUP",
        timestamp_seconds=1.25,
        frame_index=10,
        source_type="RTSP",
        camera_id="CAM01",
        camera_profile="HIKVISION_MAIN",
        frame_width=640,
        frame_height=480,
        core_visibility=_core_visibility(),
        quality_flag="ACCEPTED",
        quality_warnings=(),
        v4_prediction="TLB",
        v4_confidence=0.91,
        v4_probabilities={"TUP": 0.09, "TLB": 0.91},
        geometry={"forward": False},
        calibration_status="RUNTIME_GEOMETRY_ONLY",
        feature_values=(1.0, 2.0),
    )
    with writer.samples_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["schema_version"] == CONTROLLED_SCHEMA_VERSION
    assert rows[0]["human_label"] == "TUP"
    assert rows[0]["v4_prediction"] == "TLB"
    assert rows[0]["frame_age_ms"] == ""
    assert rows[0]["v4_a"] == "1"
    assert rows[0]["v4_b"] == "2"
    writer.append_rejection({"reason": "POSE_LOST", "raw_video_recorded": False})
    assert json.loads(writer.rejections_path.read_text(encoding="utf-8"))["reason"] == "POSE_LOST"
    with pytest.raises(ValueError, match="human_label"):
        writer.append_sample(
            sample_id="sample_002", session_id="session_001", subject_id="P001", clip_id="P001_TUP_01",
            human_label="TLL", timestamp_seconds=2, frame_index=11, source_type="RTSP", camera_id=None,
            camera_profile=None, frame_width=640, frame_height=480, core_visibility=_core_visibility(),
            quality_flag="ACCEPTED", quality_warnings=(), v4_prediction=None, v4_confidence=None,
            v4_probabilities={}, geometry={}, calibration_status="RUNTIME_GEOMETRY_ONLY", feature_values=(1, 2),
        )
