from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from scripts.collect_webcam_human_gt_v1 import (
    FEATURES, LABELS, accepted_record, build_sequence, hip_visibility, rejected_record,
    validate_dataset, validate_participant,
)


def evidence(*, state="POSE_OK", core=True, hips=0.1):
    xyz = np.zeros((33, 3), dtype=float)
    xyz[0] = [0, -0.8, 0.1]
    xyz[11] = [-0.2, -0.5, 0]
    xyz[12] = [0.2, -0.5, 0]
    xyz[23] = [-0.2, 0, 0]
    xyz[24] = [0.2, 0, 0]
    return SimpleNamespace(pose_state=state, core_valid=core, track_id=7,
        world_landmarks_xyz=xyz.tolist(), landmark_visibility={"nose": 1, "left_shoulder": 1,
        "right_shoulder": 1, "left_hip": hips, "right_hip": hips})


def make_record(**changes):
    values = dict(evidence=evidence(), participant="P001", session_id="SESSION_1", segment_id="SESSION_1_TUP_R01",
        round_index=1, human_label="TUP", timestamp_utc="2026-09-19T00:00:00Z", elapsed_seconds=3,
        frame_index=8, camera_index=0, resolution=(960, 540), human_confirmed=True)
    values.update(changes)
    return accepted_record(**values)


def test_anonymous_participant_and_explicit_human_label_are_required():
    assert validate_participant("p001") == "P001"
    with pytest.raises(ValueError): validate_participant("Alice")
    with pytest.raises(ValueError, match="explicit human"): make_record(human_confirmed=False)
    with pytest.raises(ValueError): make_record(human_label="MODEL_TUP")


def test_targeted_tlf_rounds_are_unique_and_default_protocol_is_unchanged():
    default = build_sequence(LABELS, 5)
    targeted = build_sequence(["TLF"], 2, 6)
    assert len(default) == 25 and default[:2] == [("TUP", 1), ("TUP", 2)]
    assert default[-1] == ("TLR", 5)
    assert targeted == [("TLF", 6), ("TLF", 7)]
    assert not set(targeted) & set(default)


def test_record_is_blind_segmented_webcam_gt_with_hidden_hips():
    row = make_record()
    assert row["segment_id"].endswith("TUP_R01") and row["round_index"] == 1
    assert row["label_source"] == "HUMAN" and row["label_status"] == "HUMAN_GROUND_TRUTH"
    assert row["model_prediction"] is None and not row["v4_called"]
    assert row["source_type"] == "USB" and row["camera_index"] == 0
    assert row["hip_visibility_proxy"] == "BOTH_HIDDEN"
    assert len(row["features"]) == len(FEATURES["NO_HIP"])
    assert not row["raw_video_recorded"] and not row["raw_image_saved"]


def test_unsafe_pose_is_rejected_and_never_becomes_accepted():
    with pytest.raises(ValueError, match="unsafe or invalid"):
        make_record(evidence=evidence(state="POSE_AMBIGUOUS"))
    row = rejected_record(participant="P001", session_id="S", segment_id="S_TUP_R01", round_index=1,
        human_label="TUP", reason="PADDED_CROP_PERSON_CONTAMINATION", elapsed_seconds=1,
        frame_index=2, track_id=7, pose_state="POSE_AMBIGUOUS")
    assert row["crop_safety_rejected"] and row["label_status"] == "REJECTED_NOT_TRAINING_DATA"


def test_manifest_audit_parses_rows_rejects_duplicates_and_leakage(tmp_path: Path):
    row = make_record()
    accepted = tmp_path / "accepted_observations.jsonl"
    accepted.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "rejected_evidence.jsonl").write_text("", encoding="utf-8")
    assert validate_dataset(tmp_path) == {"status": "PASS", "accepted_observations": 1,
        "rejected_observations": 0, "segment_count": 1, "feature_dimensions": [6]}
    accepted.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"): validate_dataset(tmp_path)
    row["model_prediction"] = "TUP"
    accepted.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"): validate_dataset(tmp_path)


def test_hip_visibility_states():
    assert hip_visibility(evidence(hips=0.9)) == "VISIBLE"
    item = evidence(hips=0.9); item.landmark_visibility["right_hip"] = 0.1
    assert hip_visibility(item) == "PARTLY_HIDDEN"
