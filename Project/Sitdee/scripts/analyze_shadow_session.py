"""Analyze one closed privacy-aware shadow session without opening a source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.monitoring.shadow import analyze_shadow_session, render_analysis_th  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze anonymous no-video shadow telemetry")
    parser.add_argument("--session", required=True, help="Path to one outputs/shadow_sessions/<session_id> directory")
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--report", default=None, help="Defaults to reports/shadow_session_<id>_th.md")
    args = parser.parse_args()
    if not 0.0 <= args.visibility_threshold <= 1.0:
        parser.error("--visibility-threshold must be between 0 and 1")
    try:
        analysis = analyze_shadow_session(args.session, visibility_threshold=args.visibility_threshold)
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "SHADOW_ANALYSIS_FAILED", "detail": str(exc)}, ensure_ascii=False))
        return 2
    session_dir = Path(args.session)
    json_path = session_dir / "analysis.json"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = Path(args.report) if args.report else ROOT / "reports" / f"shadow_session_{session_dir.name}_th.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_analysis_th(analysis), encoding="utf-8")
    print(json.dumps({"status": "SHADOW_ANALYSIS_COMPLETED", "analysis": str(json_path), "report": str(report_path), "bottlenecks": analysis["bottlenecks"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
