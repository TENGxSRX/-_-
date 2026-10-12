from __future__ import annotations

import json

import pytest

from src.posture_ai.multiperson.demo import run_multiperson_demo
from src.posture_ai.multiperson.seating import load_seat_layout
from src.posture_ai.multiperson.telemetry import frame_to_event


class _Result:
    def to_dict(self):
        return {"timestamp_seconds": 1.0, "tracking_status": "TEST"}


def test_multiperson_source_runtime_uses_explicit_seat_layout_contract():
    with pytest.raises(ValueError, match="seat tracking requires"):
        run_multiperson_demo("rtsp", tracking_mode="seat", display=False)


def test_multiperson_max_people_contract_is_bounded_before_source_open():
    with pytest.raises(ValueError, match="between 1 and 4"):
        run_multiperson_demo("rtsp", max_people=5, display=False)


def test_source_telemetry_redacts_url_and_drops_unknown_secrets():
    event = frame_to_event(
        _Result(),
        source_metadata={
            "source": "rtsp://alice:super-secret@example.invalid/Streaming/Channels/101",
            "source_status": "SOURCE_READY",
            "password": "super-secret",
            "resolved_source": "rtsp://alice:super-secret@example.invalid/Streaming/Channels/101",
        },
    )
    encoded = json.dumps(event, ensure_ascii=False)
    assert "super-secret" not in encoded
    assert "alice" not in encoded
    assert event["source_health"]["source_status"] == "SOURCE_READY"


def test_hikvision_layout_template_does_not_invent_real_seat_coordinates():
    with pytest.raises(ValueError, match="at least one seat"):
        load_seat_layout("configs/seat_layout_hikvision.example.yaml")
