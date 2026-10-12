from pathlib import Path
import json

import pytest

from src.posture_ai.data.local_rgb_v1 import (
    LOCAL_MANIFEST_COLUMNS,
    LOCAL_PARTICIPANTS,
    LOCAL_POSTURES,
    LOCAL_SESSIONS,
    expected_manifest_rows,
    expected_sample_id,
    expected_video_relative_path,
    parse_local_video_filename,
    validate_manifest_schema,
    validate_video_path,
)


def test_expected_local_collection_has_100_slots_and_all_classes():
    rows = expected_manifest_rows()
    assert len(rows) == 100
    assert {row["participant_id"] for row in rows} == set(LOCAL_PARTICIPANTS)
    assert {row["session_id"] for row in rows} == set(LOCAL_SESSIONS)
    assert {row["project_label"] for row in rows} == set(LOCAL_POSTURES)
    assert all(row["status"] == "WAITING_FOR_VIDEO" for row in rows)
    validate_manifest_schema(rows)


def test_controlled_filename_parser_accepts_expected_name():
    parsed = parse_local_video_filename("P10_S02_TLR_01.mp4")
    assert parsed == {
        "participant": "P10",
        "session": "S02",
        "posture": "TLR",
        "take": "01",
        "extension": ".MP4",
    }
    assert expected_sample_id("P01", "S01", "TUP") == "P01_S01_TUP_01"
    assert expected_video_relative_path("P01", "S01", "TUP").endswith("participant_01/session_01/TUP/P01_S01_TUP_01.mp4")


@pytest.mark.parametrize("filename", ["P01_S01_TUP.mp4", "participant_01.mp4", "P11_S01_TUP_01.mp4", "P01_S03_TUP_01.mp4", "P01_S01_TUP_02.mp4"])
def test_filename_parser_rejects_uncontrolled_names(filename):
    with pytest.raises(ValueError):
        parse_local_video_filename(filename)


def test_folder_and_filename_must_agree(tmp_path):
    raw_root = tmp_path / "local_rgb_v1"
    good = raw_root / "participant_01" / "session_01" / "TLF" / "P01_S01_TLF_01.mp4"
    good.parent.mkdir(parents=True)
    good.touch()
    parsed, errors = validate_video_path(good, raw_root)
    assert parsed is not None
    assert errors == []

    bad = raw_root / "participant_01" / "session_01" / "TUP" / "P01_S01_TLF_01.mp4"
    bad.parent.mkdir(parents=True)
    bad.touch()
    _, errors = validate_video_path(bad, raw_root)
    assert any("Folder does not match filename" in error for error in errors)


def test_manifest_does_not_claim_media_facts_before_video_exists():
    rows = expected_manifest_rows()
    for row in rows:
        assert row["duration_seconds"] == ""
        assert row["fps"] == ""
        assert row["total_frames"] == ""
        assert row["pose_detected_frames"] == ""
        assert row["core_accepted_frames"] == ""
        assert row["status"] == "WAITING_FOR_VIDEO"
        assert set(row) == set(LOCAL_MANIFEST_COLUMNS)


def test_current_local_split_plan_has_no_participant_overlap_and_does_not_train():
    plan_path = Path(__file__).resolve().parents[1] / "outputs" / "metrics" / "local_v5_split_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["participant_overlap"] == []
    assert plan["training_performed"] is False
