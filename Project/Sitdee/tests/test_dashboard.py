from __future__ import annotations

import json
from pathlib import Path

from src.posture_ai.demo.dashboard import MISSING_TELEMETRY, load_event_log, load_session_history, load_summary, render_dashboard


def test_fixture_dashboard_loads_and_does_not_call_fixture_results_scientific():
    summary = load_summary(Path("fixtures/test_session_summary.json"))
    html = render_dashboard(summary, history=[{"session_id": "fixture", "duration_seconds": 12.0}])
    assert "TEST_FIXTURE" in html
    assert "AI POSTURE" in html
    assert "GEOMETRY INDICATORS" in html
    assert MISSING_TELEMETRY in html


def test_dashboard_keeps_source_health_separate_from_posture():
    summary = {
        "camera_id": "CAM01",
        "source_type": "RTSP",
        "source_status": "SOURCE_RECONNECTING",
        "source_health": {
            "resolution": {"width": 1920, "height": 1080},
            "effective_received_fps": 12.5,
            "reconnect_count": 2,
        },
    }
    html = render_dashboard(summary)
    assert "SOURCE HEALTH" in html
    assert "CAM01" in html
    assert "SOURCE_RECONNECTING" in html
    assert "1920 x 1080" in html
    assert "AI POSTURE" in html


def test_dashboard_event_loader_and_missing_fields(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(json.dumps({"timestamp": 0.0, "pose_detected": True}) + "\n", encoding="utf-8")
    events, sampled = load_event_log(events_path)
    assert sampled is False
    html = render_dashboard({}, events=events)
    assert MISSING_TELEMETRY in html


def test_dashboard_renders_multi_person_and_seat_event_fields_without_guessing():
    html = render_dashboard(
        {},
        events=[
            {
                "timestamp_seconds": 1.0,
                "tracking_mode": "generic",
                "subject_count": 2,
                "subjects": [
                    {"subject_id": "PERSON_01", "v4_prediction": "TUP", "temporal_state": {"risk_level": "NORMAL"}},
                    {"subject_id": "PERSON_02", "v4_prediction": "TLB", "temporal_state": {"risk_level": "OBSERVE"}},
                ],
                "source_health": {"source_status": "SOURCE_READY"},
            }
        ],
    )
    assert "PERSON_01: TUP" in html
    assert "PERSON_02: TLB" in html
    assert "SOURCE_READY" in html


def test_session_history_reads_saved_summaries_without_guessing(tmp_path: Path):
    (tmp_path / "b.json").write_text(json.dumps({"session_id": "B", "warning_count": 2}), encoding="utf-8")
    rows = load_session_history(tmp_path)
    assert rows[0]["session_id"] == "B"
    assert rows[0]["warning_count"] == 2
    assert rows[0]["duration_seconds"] is None
