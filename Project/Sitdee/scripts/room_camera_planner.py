"""Run the measurement-driven room camera planning estimate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.planning.room_camera import plan_room, render_plan_svg  # noqa: E402


def load_spec(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        import yaml

        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("room specification must contain an object at its root")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a geometric room/camera planning estimate from explicit measurements")
    parser.add_argument("--spec", required=True, help="Path to a JSON or YAML room specification; no real dimensions are invented")
    parser.add_argument("--output-json", help="Optional path for the calculated JSON report")
    parser.add_argument("--output-svg", help="Optional path for a top-down schematic SVG")
    args = parser.parse_args()
    try:
        result = plan_room(load_spec(args.spec))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Room planning could not run: {exc}", file=sys.stderr)
        return 2

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_svg:
        output = Path(args.output_svg)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_plan_svg(result), encoding="utf-8")

    covered = sum(bool(item["covered"]) for item in result["target_coverage"])
    total = len(result["target_coverage"])
    print(json.dumps({
        "spec_status": result["spec_status"],
        "planning_estimate": result["planning_estimate"],
        "targets_covered_estimate": covered,
        "targets_total": total,
        "uncovered_target_ids": result["uncovered_target_ids"],
        "camera_score_order": result["camera_score_order"],
        "greedy_camera_set_estimate": result["greedy_camera_set_estimate"],
        "camera_count_decision": result["camera_count_decision"],
        "output_json": str(Path(args.output_json).resolve()) if args.output_json else None,
        "output_svg": str(Path(args.output_svg).resolve()) if args.output_svg else None,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
