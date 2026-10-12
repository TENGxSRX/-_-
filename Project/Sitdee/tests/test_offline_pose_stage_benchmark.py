"""Lightweight offline benchmark guards; no network, model loading or camera."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.benchmark_offline_pose_stages import sample_due, stats, synthetic_crops


class OfflinePoseBenchmarkTests(unittest.TestCase):
    def test_synthetic_compute_load_is_explicit_and_bounded(self):
        crop = np.zeros((120, 80, 3), dtype=np.uint8)
        for count in (1, 2, 4, 8):
            crops = synthetic_crops(crop, count)
            self.assertEqual(len(crops), count)
            self.assertTrue(all(item is crop for item in crops))
        with self.assertRaises(ValueError):
            synthetic_crops(crop, 0)

    def test_video_time_sampling_rates(self):
        fps = 30.0
        self.assertTrue(sample_due(0, fps, -1e12, 0.5))
        self.assertFalse(sample_due(30, fps, 0.0, 0.5))
        self.assertTrue(sample_due(60, fps, 0.0, 0.5))
        self.assertTrue(sample_due(30, fps, 0.0, 1.0))
        self.assertTrue(sample_due(15, fps, 0.0, 2.0))

    def test_summary_requires_measured_values(self):
        self.assertIsNone(stats([])["median"])
        observed = stats([10.0, 20.0, 30.0])
        self.assertEqual(observed["median"], 20.0)
        self.assertGreaterEqual(observed["p95"], observed["median"])


if __name__ == "__main__":
    unittest.main()
