from __future__ import annotations

import json

import numpy as np

from src.posture_ai.monitoring.shadow import (
    RotatingJsonlWriter,
    ShadowReviewQueue,
    analyze_shadow_session,
    build_subject_observation,
    derive_unknown_reasons,
)
from src.posture_ai.multiperson.schema import PoseSubject
from src.posture_ai.pose.landmarks import PoseLandmarks


def _subject(*, lower_visibility: float = 0.1, activity_state: str = "UNKNOWN", raw_state: str = "UNKNOWN") -> PoseSubject:
    xyz = np.zeros((33, 3), dtype=float)
    visibility = [0.95] * 33
    # Keep hips visible while knees/ankles fail: this can support only a
    # *suspected* desk-occlusion pattern, never a certainty claim.
    for index in (25, 26, 27, 28):
        visibility[index] = lower_visibility
    pose = PoseLandmarks.from_xyz_array(xyz, visibility=visibility)
    return PoseSubject(
        subject_id="PERSON_01",
        landmarks=pose,
        pose_quality=0.95,
        core_valid=True,
        activity_state=activity_state,
        activity_raw_state=raw_state,
        activity_method="GEOMETRY_EXPERIMENTAL",
        activity_evidence={"reason": "LOW_VISIBILITY_OR_MISSING_LOWER_BODY"},
        temporal_state={"risk_level": "MONITOR_PAUSED", "duration_seconds": 0.0},
    )


def _record(subject: PoseSubject, timestamp: float) -> dict:
    return build_subject_observation(
        subject,
        session_id="test-shadow",
        timestamp_seconds=timestamp,
        timestamp_utc="2026-09-10T00:00:00+00:00",
        frame_index=int(timestamp * 10),
        source_health={"source": "rtsp://user:secret@example.invalid/stream", "source_status": "SOURCE_READY"},
        inference_fps=3.0,
        frame_age_ms=12.0,
        events=(),
    )


def test_unknown_reasons_are_explicit_and_occlusion_is_only_suspected():
    reasons = derive_unknown_reasons(_subject())
    assert "LOW_LOWER_BODY_VISIBILITY" in reasons["reason_codes"]
    assert "DESK_OCCLUSION_SUSPECTED" in reasons["reason_codes"]
    assert reasons["desk_occlusion_suspected"] is True
    assert "POSE_LOST" not in reasons["reason_codes"]


def test_shadow_record_redacts_source_and_never_creates_label():
    record = _record(_subject(), 1.0)
    encoded = json.dumps(record)
    assert "secret" not in encoded
    assert record["labels"] == {"status": "NOT_A_LABEL", "human_label": None}
    assert record["raw_video_recorded"] is False


def test_review_queue_marks_items_for_human_label_only():
    queue = ShadowReviewQueue(unknown_review_seconds=0.0, repeated_lower_body_threshold=1, cooldown_seconds=0.0)
    items = queue.inspect(_record(_subject(), 1.0))
    assert items
    assert {item["review_status"] for item in items} == {"NEEDS_HUMAN_LABEL"}
    assert all(item["automatic_label"] is None for item in items)


def test_rotating_writer_is_bounded_without_deleting_old_files(tmp_path):
    writer = RotatingJsonlWriter(tmp_path, "observations", max_bytes=1024, max_files=1, flush_seconds=0.1)
    assert writer.write({"value": "a" * 700}, now_seconds=0.0)
    assert not writer.write({"value": "b" * 700}, now_seconds=1.0)
    writer.close()
    summary = writer.summary()
    assert summary["capacity_reached"] is True
    assert summary["retention_behavior"] == "ROTATE_THEN_STOP_NO_AUTOMATIC_DELETION"
    assert (tmp_path / "observations_000.jsonl").exists()


def test_analyzer_ranks_lower_body_visibility_from_shadow_records(tmp_path):
    session = tmp_path / "shadow_test"
    session.mkdir()
    record = _record(_subject(), 1.0)
    (session / "observations_000.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    frame = {
        "record_type": "frame_observation", "timestamp_seconds": 1.0, "active_track_count": 1,
        "frame_age_ms": 12.0, "inference_fps": 3.0, "source": {"reconnect_count": 0, "decode_failures": 0}, "events": [],
    }
    (session / "frames_000.jsonl").write_text(json.dumps(frame) + "\n", encoding="utf-8")
    (session / "session_manifest.json").write_text(json.dumps({"session_id": "test-shadow", "final_source_health": {"reconnect_count": 0, "decode_failures": 0}}), encoding="utf-8")
    analysis = analyze_shadow_session(session)
    assert analysis["bottlenecks"][0]["category"] == "LOWER_BODY_VISIBILITY"
    assert analysis["safety"]["automatic_labels_created"] is False
