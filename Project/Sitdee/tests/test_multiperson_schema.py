from __future__ import annotations

import numpy as np
import pytest

from src.posture_ai.multiperson import MultiPersonFrameResult, PoseSubject, anonymous_subject_id
from src.posture_ai.pose.landmarks import PoseLandmarks


def _landmarks() -> PoseLandmarks:
    return PoseLandmarks.from_xyz_array(np.zeros((33, 3), dtype=float))


def test_anonymous_ids_are_stable_and_non_biometric():
    assert anonymous_subject_id(0) == "PERSON_01"
    assert anonymous_subject_id(1) == "PERSON_02"
    with pytest.raises(ValueError):
        anonymous_subject_id(-1)


def test_multi_person_result_keeps_subject_state_separate():
    first = PoseSubject("PERSON_01", _landmarks(), 0.9, v4_result={"label": "TUP"})
    second = PoseSubject("PERSON_02", _landmarks(), 0.8, geometry={"left": True})
    result = MultiPersonFrameResult(1.5, (first, second), camera_id="CAM01")
    payload = result.to_dict()
    assert payload["subject_count"] == 2
    assert [item["subject_id"] for item in payload["subjects"]] == ["PERSON_01", "PERSON_02"]
    assert "face" not in str(payload).lower()


def test_duplicate_or_non_anonymous_ids_are_rejected():
    landmarks = _landmarks()
    with pytest.raises(ValueError):
        PoseSubject("student_1", landmarks, 0.8)
    subject = PoseSubject("PERSON_01", landmarks, 0.8)
    with pytest.raises(ValueError, match="unique"):
        MultiPersonFrameResult(0.0, (subject, subject))
