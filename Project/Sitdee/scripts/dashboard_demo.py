"""Generate an offline Thai-friendly demo dashboard from saved files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.demo.dashboard import load_event_log, load_session_history, load_summary, render_dashboard  # noqa: E402


def _latest_summary(directory: Path) -> Path | None:
    paths = sorted(directory.glob("*.json"), key=lambda path: path.name, reverse=True)
    return paths[0] if paths else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a dashboard without opening a webcam")
    parser.add_argument("--summary", help="Saved session summary JSON; defaults to latest report or TEST_FIXTURE")
    parser.add_argument("--events", help="Optional saved JSONL event log")
    parser.add_argument("--reports-dir", default="outputs/live/session_reports")
    parser.add_argument("--fixture", default="fixtures/test_session_summary.json", help="TEST_FIXTURE summary path")
    parser.add_argument("--use-fixture", action="store_true", help="Use the TEST_FIXTURE instead of the latest saved session")
    parser.add_argument("--output", default="outputs/demo/session_dashboard.html")
    parser.add_argument("--max-events", type=int, default=3000)
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    summary_path = Path(args.summary) if args.summary else (Path(args.fixture) if args.use_fixture else _latest_summary(reports_dir))
    if summary_path is None:
        summary_path = Path(args.fixture)
    if not summary_path.exists():
        print(f"No summary found: {summary_path}", file=sys.stderr)
        return 2
    try:
        summary = load_summary(summary_path)
        events = []
        events_sampled = False
        events_path = Path(args.events) if args.events else None
        if events_path is None:
            session_id = summary.get("session_id")
            if session_id:
                candidate = Path("outputs/live/logs") / f"webcam_{session_id}.jsonl"
                if candidate.exists():
                    events_path = candidate
        if events_path is not None and events_path.exists():
            events, events_sampled = load_event_log(events_path, max_events=args.max_events)
        history = load_session_history(reports_dir) if reports_dir.exists() else []
        html = render_dashboard(
            summary,
            events=events,
            history=history,
            summary_source=str(summary_path),
            events_source=str(events_path) if events_path is not None and events_path.exists() else None,
            events_sampled=events_sampled,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Dashboard could not be generated: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(json.dumps({
        "output": str(output.resolve()),
        "summary": str(summary_path.resolve()),
        "events": str(events_path.resolve()) if events_path is not None and events_path.exists() else "NOT_RECORDED",
        "timeline_rows": len(events),
        "history_rows": len(history),
        "fixture_status": summary.get("fixture_status", "SAVED_SESSION"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
