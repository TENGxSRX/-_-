from __future__ import annotations

import numpy as np

from src.posture_ai.inference.pipeline import TemporalSmoother


def test_temporal_smoother_requires_persistence_before_switching():
    smoother = TemporalSmoother(["TUP", "TLF"], window_size=3, persistence_frames=2)
    first = smoother.update(np.array([0.9, 0.1]))
    assert first["label"] == "TUP"
    second = smoother.update(np.array([0.1, 0.9]))
    assert second["candidate_label"] in {"TUP", "TLF"}
    third = smoother.update(np.array([0.1, 0.9]))
    assert third["label"] == "TUP"
    fourth = smoother.update(np.array([0.1, 0.9]))
    assert fourth["label"] == "TLF"
    assert abs(sum(fourth["probabilities"].values()) - 1.0) < 1e-9
