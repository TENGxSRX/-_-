from __future__ import annotations

from pathlib import Path

import pytest

from src.posture_ai.inference.webcam_v1 import (
    resolve_performance_preset,
    should_process_frame,
)


def test_performance_presets_are_explicit_and_non_semantic():
    assert resolve_performance_preset("DESKTOP")["process_every"] == 1
    assert resolve_performance_preset("BALANCED")["process_every"] == 2
    low = resolve_performance_preset("LOW_SPEC")
    assert low["process_every"] == 3
    assert low["capture_width"] == 640
    assert low["capture_height"] == 480


def test_process_every_override_is_validated():
    assert resolve_performance_preset("LOW_SPEC", 2)["process_every"] == 2
    with pytest.raises(ValueError):
        resolve_performance_preset("LOW_SPEC", 0)
    with pytest.raises(ValueError):
        resolve_performance_preset("unknown")


def test_first_frame_is_always_processed_and_skips_are_explicit():
    assert should_process_frame(0, 3, False)
    assert not should_process_frame(1, 3, True)
    assert not should_process_frame(2, 3, True)
    assert should_process_frame(3, 3, True)


def test_laptop_fixture_is_marked_as_non_scientific():
    fixture = Path("fixtures/test_session_summary.json")
    assert fixture.exists()
    assert '"fixture_status": "TEST_FIXTURE"' in fixture.read_text(encoding="utf-8")
