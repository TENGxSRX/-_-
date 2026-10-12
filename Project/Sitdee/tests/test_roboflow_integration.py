import json

import pytest

from scripts.audit_roboflow_split import audit
from src.posture_ai.pose.roboflow_adapter import parse_roboflow_pose


def test_parse_roboflow_named_keypoints():
    payload = {"image": {"width": 640, "height": 480}, "predictions": [{"confidence": .9, "keypoints": [
        {"name": "nose", "x": 320, "y": 100, "confidence": .99},
        {"name": "left_shoulder", "x": 250, "y": 200, "confidence": .9},
        {"name": "right_shoulder", "x": 390, "y": 200, "confidence": .9},
        {"name": "left_hip", "x": 270, "y": 400, "confidence": .2},
    ]}]}
    pose = parse_roboflow_pose(payload, confidence=.35)
    assert pose.by_name("nose").x == .5
    assert pose.by_name("left_shoulder").y == pytest.approx(200 / 480)
    assert pose.by_name("left_hip") is None


def test_split_audit_rejects_same_person(tmp_path):
    for split in ("train", "test"):
        directory = tmp_path / split / "images"; directory.mkdir(parents=True)
        (directory / f"P001__S1__TUP__000001.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="FAIL_PARTICIPANT_LEAKAGE"):
        audit(tmp_path)


def test_split_audit_accepts_disjoint_people(tmp_path):
    for split, person in (("train", "P001"), ("valid", "P002"), ("test", "P003")):
        directory = tmp_path / split / "images"; directory.mkdir(parents=True)
        (directory / f"{person}__S1__TUP__000001.jpg").write_bytes(b"x")
    assert audit(tmp_path)["status"] == "PASS"

