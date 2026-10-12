"""Generate a small, clearly marked synthetic multi-person telemetry fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.multiperson.overlay import dashboard_cards  # noqa: E402
from src.posture_ai.multiperson.runtime import MultiPersonRuntime  # noqa: E402
from src.posture_ai.multiperson.telemetry import MultiPersonTelemetry  # noqa: E402
from src.posture_ai.multiperson.tracking import PoseDetection  # noqa: E402
from src.posture_ai.pose.landmarks import PoseLandmarks  # noqa: E402


class FixtureV4Runner:
    labels = ["TLB", "TUP"]

    def predict_world_landmarks(self, world_landmarks, *, image_visibility=None, visibility_threshold=0.5):
        center_x = float(world_landmarks.by_name("left_shoulder").x + world_landmarks.by_name("right_shoulder").x) / 2.0
        label = "TLB" if center_x < 0.5 else "TUP"
        return {
            "label": label,
            "display_label": "Upright" if label == "TUP" else "Lean backward",
            "confidence": 0.9,
            "probabilities": {"TLB": 0.1 if label == "TUP" else 0.9, "TUP": 0.9 if label == "TUP" else 0.1},
            "feature_count": 15,
            "fixture_only": True,
        }


def _pose(center_x: float) -> PoseLandmarks:
    values = np.zeros((33, 3), dtype=float)
    values[0] = [center_x, 0.25, 0.0]
    values[11] = [center_x - 0.10, 0.45, 0.0]
    values[12] = [center_x + 0.10, 0.45, 0.0]
    values[23] = [center_x - 0.08, 0.72, 0.0]
    values[24] = [center_x + 0.08, 0.72, 0.0]
    return PoseLandmarks.from_xyz_array(values, visibility=[0.95] * 33)


def _detection(center_x: float, label_quality: float = 0.9) -> PoseDetection:
    pose = _pose(center_x)
    return PoseDetection(pose, pose, pose_quality=label_quality, core_valid=True)


def build_fixture() -> dict:
    runtime = MultiPersonRuntime(FixtureV4Runner())
    telemetry = MultiPersonTelemetry(max_events=20)
    result = runtime.process_frame([_detection(0.30), _detection(0.70)], 0.0)
    event = telemetry.add_frame(result, frame_index=0)
    return {
        "fixture_status": "TEST_FIXTURE — synthetic architecture demonstration only",
        "scientific_use": "Not real experimental evidence; no camera or participant data",
        "frame_event": event,
        "dashboard_cards": dashboard_cards(result),
        "expected_properties": {
            "anonymous_ids": ["PERSON_01", "PERSON_02"],
            "independent_v4_results": True,
            "independent_temporal_state": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "fixtures" / "multiperson_prototype_v1.json")
    args = parser.parse_args()
    payload = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "fixture_status": payload["fixture_status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
