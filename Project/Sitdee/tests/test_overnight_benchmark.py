from __future__ import annotations

import numpy as np

from scripts.overnight_benchmark import sampling_step, stats
from scripts.run_hybrid_v4_geometry import run as run_hybrid


def test_overnight_sampling_is_deterministic_and_rate_ordered():
    assert sampling_step(30.0, 5.0) == 6
    assert sampling_step(29.998, 2.0) == 15
    assert sampling_step(29.998, 1.0) == 30


def test_overnight_stats_ignores_nonfinite_values():
    result = stats([1.0, 2.0, np.nan, np.inf, -np.inf])
    assert result["min"] == 1.0
    assert result["max"] == 2.0
    assert result["mean"] == 1.5


def test_overnight_stats_empty_is_explicit():
    result = stats([np.nan, np.inf])
    assert result["mean"] is None
    assert result["p90"] is None


def test_hybrid_demo_rejects_webcam_source_zero(tmp_path):
    try:
        run_hybrid("0", tmp_path / "out.json", max_frames=1)
    except ValueError as exc:
        assert "webcam source 0" in str(exc)
    else:
        raise AssertionError("webcam source 0 must never be opened by the offline demo")
