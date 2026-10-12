from __future__ import annotations

from src.posture_ai.data.cctv_metrics import (
    aggregate_detection_metrics,
    detection_frame_metrics,
    detection_metrics_by_scenario,
    frozen_v4_metrics,
    per_person_core_metrics,
    temporal_stability_metrics,
    tracking_metrics,
)


def test_detection_count_metrics_separate_exact_under_over_and_core():
    exact = detection_frame_metrics(3, 3, 2)
    assert exact["exact_count"] is True
    assert exact["all_people_core_valid"] is False
    aggregate = aggregate_detection_metrics([
        {"expected_people_count": 3, "detected_people_count": 3, "core_valid_people_count": 3},
        {"expected_people_count": 3, "detected_people_count": 2, "core_valid_people_count": 2},
        {"expected_people_count": 3, "detected_people_count": 4, "core_valid_people_count": 4},
    ])
    assert aggregate["exact_count_rate"] == 1 / 3
    assert aggregate["undercount_rate"] == 1 / 3
    assert aggregate["overcount_rate"] == 1 / 3
    assert aggregate["all_people_core_valid_rate"] == 1 / 3


def test_detection_metrics_group_by_expected_people_count():
    grouped = detection_metrics_by_scenario([
        {"expected_people_count": 1, "detected_people_count": 1, "core_valid_people_count": 1},
        {"expected_people_count": 2, "detected_people_count": 1, "core_valid_people_count": 1},
    ])
    assert grouped["1"]["exact_count_rate"] == 1.0
    assert grouped["2"]["undercount_rate"] == 1.0


def test_core_metrics_group_by_participant_and_v4_requires_human_labels():
    rows = [
        {"subject_id": "P001", "pose_detected": 1, "core_valid": 1, "nose_visibility": 0.9, "left_hip_visibility": 0.8},
        {"subject_id": "P001", "pose_detected": 1, "core_valid": 0, "nose_visibility": 0.4, "left_hip_visibility": 0.2},
    ]
    core = per_person_core_metrics(rows)
    assert core["groups"]["P001"]["pose_detected_rate"] == 1.0
    assert core["groups"]["P001"]["core_valid_rate"] == 0.5
    assert frozen_v4_metrics(["TUP", "TLB"], ["TUP", "TUP"])["sample_count"] == 2


def test_temporal_and_tracking_metrics_are_diagnostic_not_identity_metrics():
    temporal = temporal_stability_metrics([
        {"timestamp_seconds": 0, "human_label": "TUP", "v4_prediction": "TUP", "v4_confidence": 0.9},
        {"timestamp_seconds": 1, "human_label": "TUP", "v4_prediction": "TLB", "v4_confidence": 0.6},
        {"timestamp_seconds": 2, "human_label": "TUP", "v4_prediction": "TUP", "v4_confidence": 0.8},
    ])
    assert temporal["raw_prediction_switches"] == 2
    tracking = tracking_metrics([
        {"timestamp_seconds": 0, "subjects": [{"subject_id": "PERSON_01", "tracking_state": "NEW"}]},
        {"timestamp_seconds": 1, "subjects": [{"subject_id": "PERSON_01", "tracking_state": "TEMPORARILY_LOST"}]},
    ])
    assert tracking["new_track_count"] == 1
    assert tracking["id_switch_evidence"] == "NOT_MEASURED_NO_MANUAL_TRACK_ID_GROUND_TRUTH"
