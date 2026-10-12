from __future__ import annotations

import numpy as np
import pytest

from src.posture_ai.data import MultiPostureAdapter, UnsupportedLabelError, canonical_label, get_dataset_record
from src.posture_ai.data.cctv_collection import ManualPersonAssignment, label_for_key, validate_person_id
from src.posture_ai.data.controlled_collection import ControlledSampleWriter, evaluate_collection_quality
from src.posture_ai.multiperson import MultiPersonRuntime, MultiPersonRuntimeConfig, PoseDetection, core_quality_summary, resolve_performance_preset
from src.posture_ai.pose.landmarks import PoseLandmarks


def _pose(center_x: float) -> PoseLandmarks:
    values = np.zeros((33, 3), dtype=float)
    values[0] = [center_x, 0.20, 0.0]
    values[11] = [center_x - 0.08, 0.45, 0.0]
    values[12] = [center_x + 0.08, 0.45, 0.0]
    values[23] = [center_x - 0.07, 0.72, 0.0]
    values[24] = [center_x + 0.07, 0.72, 0.0]
    values[25] = [center_x - 0.18, 0.76, 0.0]
    values[26] = [center_x + 0.18, 0.76, 0.0]
    values[27] = [center_x - 0.24, 0.60, 0.0]
    values[28] = [center_x + 0.24, 0.60, 0.0]
    return PoseLandmarks.from_xyz_array(values, visibility=[0.95] * 33)


def _detection(center_x: float) -> PoseDetection:
    pose = _pose(center_x)
    return PoseDetection(pose, pose, pose_quality=0.95, core_valid=True, center=(center_x, 0.55))


class _V4:
    labels = ["TLB", "TUP"]

    def predict_world_landmarks(self, world_landmarks, *, image_visibility=None, visibility_threshold=0.5):
        return {"label": "TUP", "confidence": 0.9, "probabilities": {"TLB": 0.1, "TUP": 0.9}}


def test_cctv_runtime_supports_configured_one_to_four_people_without_shared_state():
    detections = [_detection(0.12), _detection(0.36), _detection(0.60), _detection(0.84)]
    for max_people in range(1, 5):
        runtime = MultiPersonRuntime(_V4(), runtime_config=MultiPersonRuntimeConfig(max_people=max_people))
        result = runtime.process_frame(detections, 0.0)
        assert len(result.subjects) == max_people
        assert len({subject.subject_id for subject in result.subjects}) == max_people


def test_rtsp_cctv_preset_is_latest_frame_sized_for_desktop_inference():
    preset = resolve_performance_preset("RTSP_CCTV")
    assert preset["process_every"] == 1
    assert preset["capture_width"] == 960
    assert preset["capture_height"] == 540


def test_core_quality_keeps_detection_and_landmark_quality_separate():
    detection = _detection(0.5)
    summary = core_quality_summary(detection)
    assert summary["HEAD_VISIBLE"] is True
    assert summary["SHOULDERS_VISIBLE"] is True
    assert summary["HIPS_VISIBLE"] is True
    assert summary["CORE_OK"] is True
    assert summary["detector_confidence"] == "NOT_EXPOSED_BY_CURRENT_MEDIAPIPE_ADAPTER"


def test_telemetry_exposes_core_visibility_and_clears_it_when_track_is_lost():
    runtime = MultiPersonRuntime(_V4(), runtime_config=MultiPersonRuntimeConfig(max_people=1, max_misses=1))
    visible = runtime.process_frame([_detection(0.5)], 0.0).subjects[0]
    assert visible.core_visibility["nose"] == 0.95
    lost = runtime.process_frame([], 0.1).subjects[0]
    assert lost.core_visibility["nose"] is None


def test_runtime_exposes_hierarchical_uncertain_until_models_are_trained():
    runtime = MultiPersonRuntime(_V4(), runtime_config=MultiPersonRuntimeConfig(max_people=1))
    subject = runtime.process_frame([_detection(0.5)], 0.0).subjects[0]
    hierarchy = subject.hierarchical_state
    assert hierarchy["state"] == "UNCERTAIN"
    assert hierarchy["activity_model_status"] == "NOT_TRAINED"
    assert hierarchy["sitting_quality_model_status"] == "NOT_TRAINED"
    assert hierarchy["subtype_model_status"] == "NOT_TRAINED"


def test_canonical_adapter_rejects_unverified_labels_and_pseudo_label_dataset():
    adapter = MultiPostureAdapter()
    assert adapter.map_label("TUP") == "NORMAL"
    assert canonical_label("MultiPosture", "TLR") == "NON_NORMAL"
    with pytest.raises(UnsupportedLabelError):
        adapter.map_label("UNKNOWN")
    with pytest.raises(UnsupportedLabelError):
        canonical_label("ANYA", "1")


def test_registry_preserves_anyas_pseudo_label_only_status():
    record = get_dataset_record("ANYA")
    assert record.trust_level == "PSEUDO_LABEL_ONLY"
    assert record.supervised_training_allowed is False
    assert get_dataset_record("MultiPosture").supervised_training_allowed is True


def test_controlled_cctv_assignments_are_manual_and_anonymous():
    assignment = ManualPersonAssignment("person_03", "tll")
    assert assignment.person_id == "PERSON_03"
    assert assignment.recordable is True
    assert assignment.to_dict()["label_source"] == "MANUAL_OPERATOR_ONLY"
    assert label_for_key("5") == "TLR"
    assert label_for_key("u") == "UNCERTAIN"
    assert validate_person_id("PERSON_04") == "PERSON_04"
    with pytest.raises(ValueError, match="anonymous"):
        validate_person_id("Alice")


def test_controlled_cctv_quality_accepts_compact_runtime_visibility_map():
    visibility = {name: 0.91 for name in ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")}
    result = evaluate_collection_quality(
        pose_detected=True,
        core_visibility=visibility,
        feature_values=(0.1, 0.2),
        feature_names=("f0", "f1"),
        frame_width=960,
        frame_height=540,
    )
    assert result["accepted"] is True


def test_controlled_cctv_writer_can_store_verified_five_class_manual_labels(tmp_path):
    writer = ControlledSampleWriter(
        tmp_path,
        feature_names=("f0", "f1"),
        allowed_labels=("TUP", "TLB", "TLF", "TLL", "TLR"),
    )
    visibility = {name: 0.95 for name in ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")}
    writer.append_sample(
        sample_id="sample_001",
        session_id="session_001",
        subject_id="P001",
        clip_id="CCTV_01",
        human_label="TLL",
        timestamp_seconds=1.0,
        frame_index=1,
        source_type="RTSP",
        camera_id="CAM01",
        camera_profile="HIKVISION_MAIN",
        frame_width=960,
        frame_height=540,
        core_visibility=visibility,
        quality_flag="ACCEPTED",
        quality_warnings=(),
        v4_prediction="TUP",
        v4_confidence=0.8,
        v4_probabilities={"TUP": 0.8, "TLB": 0.2},
        geometry={},
        calibration_status="MULTI_PERSON_RUNTIME_GEOMETRY_ONLY",
        feature_values=(1.0, 2.0),
    )
    assert writer.sample_count == 1


def test_controlled_cctv_writer_preserves_v2_ground_truth_metadata_columns(tmp_path):
    writer = ControlledSampleWriter(
        tmp_path,
        feature_names=("f0",),
        allowed_labels=("TUP", "TLB"),
        schema_version="controlled_cctv_ground_truth_v2",
        extra_columns=("expected_people_count", "target_subject_id", "label_source"),
    )
    visibility = {name: 0.95 for name in ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")}
    writer.append_sample(
        sample_id="sample_001",
        session_id="session_001",
        subject_id="P001",
        clip_id="CCTV_01",
        human_label="TUP",
        timestamp_seconds=1.0,
        frame_index=1,
        source_type="RTSP",
        camera_id="CAM01",
        camera_profile="HIKVISION_MAIN",
        frame_width=960,
        frame_height=540,
        core_visibility=visibility,
        quality_flag="ACCEPTED",
        quality_warnings=(),
        v4_prediction="TUP",
        v4_confidence=0.8,
        v4_probabilities={"TUP": 0.8, "TLB": 0.2},
        geometry={},
        calibration_status="MULTI_PERSON_RUNTIME_GEOMETRY_ONLY",
        feature_values=(1.0,),
        extra_values={"expected_people_count": 2, "target_subject_id": "PERSON_01", "label_source": "MANUAL_OPERATOR_ONLY"},
    )
    row = (tmp_path / "samples.csv").read_text(encoding="utf-8")
    assert "expected_people_count" in row
    assert "MANUAL_OPERATOR_ONLY" in row
