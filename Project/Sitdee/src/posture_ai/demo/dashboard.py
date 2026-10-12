"""Small offline HTML dashboard for saved session data.

The dashboard is deliberately file-based and read-only. It separates AI
posture fields from experimental geometry indicators and never fills missing
telemetry with guessed values.
"""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any, Mapping


MISSING_TELEMETRY = "NOT RECORDED IN THIS SESSION"


def load_summary(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"session summary must be a JSON object: {source}")
    return value


def load_event_log(path: str | Path, *, max_events: int = 3000) -> tuple[list[dict[str, Any]], bool]:
    """Load JSONL events and return (timeline, was_sampled_or_truncated)."""

    if max_events < 1:
        raise ValueError("max_events must be positive")
    source = Path(path)
    events: list[dict[str, Any]] = []
    sampled = False
    hard_limit = max_events * 4
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(events) >= hard_limit:
                sampled = True
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {source}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object at {source}:{line_number}")
            events.append(value)
    if len(events) > max_events:
        step = int(math.ceil(len(events) / max_events))
        events = events[::step]
        sampled = True
    return events, sampled


def load_session_history(report_dir: str | Path, *, limit: int = 50) -> list[dict[str, Any]]:
    """Read saved summaries only; malformed files are surfaced to the caller."""

    if limit < 1:
        raise ValueError("limit must be positive")
    directory = Path(report_dir)
    paths = sorted(directory.glob("*.json"), key=lambda path: path.name, reverse=True)[:limit]
    rows: list[dict[str, Any]] = []
    for path in paths:
        summary = load_summary(path)
        rows.append({
            "source_path": str(path),
            "session_id": summary.get("session_id", path.stem),
            "duration_seconds": summary.get("total_session_duration_seconds"),
            "input_quality_status": summary.get("input_quality_status"),
            "warning_count": summary.get("warning_count"),
            "valid_pose_percentage": summary.get("valid_pose_percentage"),
            "average_fps": summary.get("average_fps"),
            "source_type": summary.get("source_type"),
            "camera_id": summary.get("camera_id"),
            "source_status": summary.get("source_status"),
            "source_effective_fps": summary.get("source_effective_fps") or (summary.get("source_health") or {}).get("effective_received_fps"),
        })
    return rows


def _value(data: Mapping[str, Any], key: str) -> Any:
    value = data.get(key)
    return MISSING_TELEMETRY if value is None else value


def _number(data: Mapping[str, Any], key: str, suffix: str = "") -> str:
    value = data.get(key)
    if value is None:
        return MISSING_TELEMETRY
    try:
        return f"{float(value):.1f}{suffix}"
    except (TypeError, ValueError):
        return escape(str(value))


def _event_value(event: Mapping[str, Any], key: str) -> str:
    if key not in event or event.get(key) is None:
        return MISSING_TELEMETRY
    value = event[key]
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, (int, float)):
        return f"{value:.3f}" if isinstance(value, float) else str(value)
    return str(value)


def _source_resolution(summary: Mapping[str, Any]) -> str:
    health = summary.get("source_health") or {}
    value = summary.get("source_resolution") or health.get("resolution")
    if isinstance(value, Mapping):
        width, height = value.get("width"), value.get("height")
        if width is not None and height is not None:
            return f"{width} x {height}"
    return MISSING_TELEMETRY if value is None else str(value)


def _source_value(summary: Mapping[str, Any], key: str) -> Any:
    health = summary.get("source_health") or {}
    value = summary.get(key)
    if value is None:
        health_key = {
            "source_effective_fps": "effective_received_fps",
            "source_reconnect_count": "reconnect_count",
        }.get(key, key)
        value = health.get(health_key)
    return MISSING_TELEMETRY if value is None else value


def _multi_person_timeline_values(event: Mapping[str, Any]) -> tuple[str, str, str]:
    """Summarize optional multi-person/seat fields without inventing values."""

    seats = event.get("seats")
    if isinstance(seats, list):
        posture = []
        risks = []
        states = []
        for seat in seats:
            if not isinstance(seat, Mapping):
                continue
            seat_id = str(seat.get("seat_id", "SEAT"))
            prediction = seat.get("v4_prediction")
            if prediction is None:
                result = seat.get("v4_result")
                prediction = result.get("label") if isinstance(result, Mapping) else None
            posture.append(f"{seat_id}: {prediction or MISSING_TELEMETRY}")
            temporal = seat.get("temporal_state")
            risk = temporal.get("risk_level") if isinstance(temporal, Mapping) else None
            risks.append(f"{seat_id}: {risk or MISSING_TELEMETRY}")
            states.append(f"{seat_id}: {seat.get('assignment_state', MISSING_TELEMETRY)}")
        return "; ".join(posture) or MISSING_TELEMETRY, "; ".join(risks) or MISSING_TELEMETRY, "; ".join(states) or MISSING_TELEMETRY

    subjects = event.get("subjects")
    if isinstance(subjects, list):
        posture = []
        risks = []
        for subject in subjects:
            if not isinstance(subject, Mapping):
                continue
            subject_id = str(subject.get("subject_id", "PERSON"))
            posture.append(f"{subject_id}: {subject.get('v4_prediction') or MISSING_TELEMETRY}")
            temporal = subject.get("temporal_state")
            risk = temporal.get("risk_level") if isinstance(temporal, Mapping) else None
            risks.append(f"{subject_id}: {risk or MISSING_TELEMETRY}")
        count = event.get("subject_count", len(subjects))
        return "; ".join(posture) or MISSING_TELEMETRY, "; ".join(risks) or MISSING_TELEMETRY, f"SUBJECTS: {count}"

    return MISSING_TELEMETRY, MISSING_TELEMETRY, MISSING_TELEMETRY


def _timeline_rows(events: list[dict[str, Any]]) -> str:
    if not events:
        return f'<tr><td colspan="8">{escape(MISSING_TELEMETRY)}</td></tr>'
    rows = []
    for event in events:
        multi_posture, multi_risk, multi_pose = _multi_person_timeline_values(event)
        if multi_posture != MISSING_TELEMETRY:
            posture = multi_posture
            risk = multi_risk
            pose = multi_pose
        else:
            posture = _event_value(event, "ai_posture")
            risk = _event_value(event, "risk_state")
            if "pose_detected" not in event:
                pose = MISSING_TELEMETRY
            else:
                pose = "VALID" if bool(event.get("pose_detected")) and bool(event.get("core_landmark_accepted", False)) else "LOST / REJECTED"
        geometry = []
        for key, label in (("forward_indicator", "Forward"), ("left_indicator", "Left"), ("right_indicator", "Right")):
            if key in event and event.get(key) is not None:
                geometry.append(f"{label}: {'YES' if event[key] else 'NO'}")
        geometry_text = "; ".join(geometry) if geometry else MISSING_TELEMETRY
        source_health = event.get("source_health")
        source_status = source_health.get("source_status") if isinstance(source_health, Mapping) else event.get("source_status")
        rows.append(
            "<tr>"
            f"<td>{escape(_event_value(event, 'timestamp') if 'timestamp' in event else _event_value(event, 'timestamp_seconds'))}</td>"
            f"<td>{escape(posture)}</td>"
            f"<td>{escape(risk)}</td>"
            f"<td>{escape(pose)}</td>"
            f"<td>{escape(_event_value(event, 'ai_confidence'))}</td>"
            f"<td>{escape(geometry_text)}</td>"
            f"<td>{escape(_event_value(event, 'warning_event'))}</td>"
            f"<td>{escape(str(source_status) if source_status is not None else MISSING_TELEMETRY)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _summary_metric(label: str, value: str, *, group: str) -> str:
    return f'<div class="metric" data-group="{escape(group)}"><span>{escape(label)}</span><strong>{value}</strong></div>'


def render_dashboard(
    summary: Mapping[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
    summary_source: str | None = None,
    events_source: str | None = None,
    events_sampled: bool = False,
) -> str:
    """Render a standalone Thai-friendly HTML dashboard."""

    events = list(events or [])
    history = list(history or [])
    ai_metrics = "".join([
        _summary_metric("Upright", _number(summary, "upright_percentage", "%"), group="AI POSTURE"),
        _summary_metric("Lean Backward", _number(summary, "lean_backward_percentage", "%"), group="AI POSTURE"),
    ])
    geometry_metrics = "".join([
        _summary_metric("Forward indicator", _number(summary, "forward_indicator_percentage", "%"), group="GEOMETRY INDICATORS"),
        _summary_metric("Left indicator", _number(summary, "left_indicator_percentage", "%"), group="GEOMETRY INDICATORS"),
        _summary_metric("Right indicator", _number(summary, "right_indicator_percentage", "%"), group="GEOMETRY INDICATORS"),
    ])
    general_metrics = "".join([
        _summary_metric("Session duration", _number(summary, "total_session_duration_seconds", " s"), group="SESSION"),
        _summary_metric("Valid pose", _number(summary, "valid_pose_percentage", "%"), group="SESSION"),
        _summary_metric("Pose lost", _number(summary, "pose_lost_percentage", "%"), group="SESSION"),
        _summary_metric("Warning count", _number(summary, "warning_count"), group="SESSION"),
        _summary_metric("Longest poor period", _number(summary, "longest_continuous_poor_posture_seconds", " s"), group="SESSION"),
        _summary_metric("Average FPS", _number(summary, "average_fps"), group="SESSION"),
    ])
    source_metrics = "".join([
        _summary_metric("Camera", str(_source_value(summary, "camera_id")), group="SOURCE HEALTH"),
        _summary_metric("Source type", str(_source_value(summary, "source_type")), group="SOURCE HEALTH"),
        _summary_metric("Source status", str(_source_value(summary, "source_status")), group="SOURCE HEALTH"),
        _summary_metric("Resolution", _source_resolution(summary), group="SOURCE HEALTH"),
        _summary_metric("Received FPS", _number({"value": _source_value(summary, "source_effective_fps")}, "value"), group="SOURCE HEALTH"),
        _summary_metric("Reconnects", _number({"value": _source_value(summary, "source_reconnect_count")}, "value"), group="SOURCE HEALTH"),
    ])
    history_rows = []
    for row in history:
        history_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('session_id', MISSING_TELEMETRY)))}</td>"
            f"<td>{escape(_number(row, 'duration_seconds', ' s'))}</td>"
            f"<td>{escape(str(_value(row, 'input_quality_status')))}</td>"
            f"<td>{escape(_number(row, 'warning_count'))}</td>"
            f"<td>{escape(_number(row, 'valid_pose_percentage', '%'))}</td>"
            f"<td>{escape(_number(row, 'average_fps'))}</td>"
            f"<td>{escape(str(row.get('source_status') or MISSING_TELEMETRY))}</td>"
            f"<td>{escape(_number(row, 'source_effective_fps'))}</td>"
            "</tr>"
        )
    if not history_rows:
        history_rows.append(f'<tr><td colspan="8">{escape(MISSING_TELEMETRY)}</td></tr>')
    event_note = "Timeline is sampled for display." if events_sampled else ""
    source_note = " · ".join(filter(None, [summary_source, events_source, event_note])) or "TEST_FIXTURE or saved session"
    return f'''<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Office Posture - Offline Demo Dashboard</title>
<style>
:root {{ color-scheme: light; font-family: Segoe UI, Tahoma, sans-serif; background: #f4f7fb; color: #172033; }}
body {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
h1 {{ margin-bottom: 4px; }} h2 {{ margin-top: 28px; border-bottom: 1px solid #ccd5e2; padding-bottom: 6px; }}
.note {{ color: #52627a; font-size: .92rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 10px; }}
.metric {{ background: white; border: 1px solid #d8e0eb; border-radius: 8px; padding: 12px; min-height: 58px; }}
.metric span {{ display:block; color:#52627a; font-size:.82rem; }} .metric strong {{ display:block; margin-top:6px; font-size:1.15rem; }}
.group-ai {{ border-left: 5px solid #2563eb; }} .group-geo {{ border-left: 5px solid #d97706; }} .group-session {{ border-left: 5px solid #64748b; }}
.table-wrap {{ overflow-x:auto; background:white; border:1px solid #d8e0eb; border-radius:8px; }}
table {{ border-collapse: collapse; width:100%; min-width:680px; }} th,td {{ padding:9px 10px; border-bottom:1px solid #e5eaf1; text-align:left; font-size:.88rem; }} th {{ background:#edf2f8; }}
.missing {{ color:#8a4b08; }} footer {{ margin-top:24px; color:#52627a; font-size:.8rem; }}
@media (max-width: 560px) {{ body {{ padding:12px; }} h1 {{ font-size:1.45rem; }} }}
</style>
</head>
<body>
<h1>Office Posture — Offline Demo</h1>
<p class="note">แดชบอร์ดจากไฟล์ที่บันทึกไว้ ไม่ได้เปิดกล้อง และไม่ใช่คะแนนสุขภาพหรือการวินิจฉัย</p>
<p class="note">แหล่งข้อมูล: {escape(source_note)}</p>
<h2>Session overview</h2>
<div class="grid">{general_metrics}</div>
<h2>SOURCE HEALTH</h2>
<p class="note">สถานะกล้อง/แหล่งภาพแยกจาก AI posture และ risk state</p>
<div class="grid">{source_metrics}</div>
<h2>AI POSTURE</h2>
<p class="note">ค่าจาก V4: TUP / TLB เท่านั้น</p>
<div class="grid">{ai_metrics}</div>
<h2>GEOMETRY INDICATORS</h2>
<p class="note">ค่าทดลองจาก geometry engine แยกจาก AI posture</p>
<div class="grid">{geometry_metrics}</div>
<h2>Timeline</h2>
<p class="note">แสดง AI posture, risk state, ความครบของ input, confidence และ geometry ที่มีอยู่จริง หากไม่มี field จะแสดง {escape(MISSING_TELEMETRY)}</p>
<div class="table-wrap"><table><thead><tr><th>เวลา</th><th>AI posture</th><th>Risk state</th><th>Input / subjects</th><th>Confidence</th><th>Geometry</th><th>Warning</th><th>Source</th></tr></thead><tbody>{_timeline_rows(events)}</tbody></table></div>
<h2>Session history</h2>
<div class="table-wrap"><table><thead><tr><th>Session</th><th>Duration</th><th>Input quality</th><th>Warnings</th><th>Valid pose</th><th>Average FPS</th><th>Source</th><th>Source FPS</th></tr></thead><tbody>{''.join(history_rows)}</tbody></table></div>
<footer>AI posture และ geometry เป็นข้อมูลเชิงวิศวกรรมของ prototype · source files remain unchanged</footer>
</body></html>'''
