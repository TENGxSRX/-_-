"""Bounded utilities for the local autonomous CCTV long-session lab.

This module stores anonymous tracking metadata only. It does not store pixels,
create posture labels, call remote APIs, or load/modify a posture model.
"""

from __future__ import annotations

import json
import math
import threading
from collections import Counter, OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import median
from typing import Any, Mapping


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    index = int(round((len(ordered) - 1) * percentile))
    return ordered[index]


def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


@dataclass
class CandidateSelectorConfig:
    representative_interval_seconds: float = 30.0
    transition_distance: float = 0.12
    low_confidence_threshold: float = 0.45
    stable_confidence_threshold: float = 0.70
    occlusion_iou_threshold: float = 0.25
    event_cooldown_seconds: float = 3.0
    max_track_memory: int = 4096


@dataclass
class _TrackSelection:
    last_selected_at: float = -1e12
    last_vector: tuple[float, ...] | None = None
    last_reason_at: dict[str, float] = field(default_factory=dict)


class CandidateSelector:
    """Select diverse metadata records without retaining frames or labels."""

    def __init__(self, config: CandidateSelectorConfig | None = None) -> None:
        self.config = config or CandidateSelectorConfig()
        self._tracks: OrderedDict[int, _TrackSelection] = OrderedDict()
        self.counts: Counter[str] = Counter()
        self.duplicates_rejected = 0
        self.quality_accepted = 0

    @staticmethod
    def _vector(box: Any, width: int, height: int) -> tuple[float, ...]:
        return (
            ((box.x1 + box.x2) * 0.5) / width,
            ((box.y1 + box.y2) * 0.5) / height,
            (box.x2 - box.x1) / width,
            (box.y2 - box.y1) / height,
            float(box.confidence),
        )

    def _state(self, track_id: int) -> _TrackSelection:
        state = self._tracks.pop(track_id, None) or _TrackSelection()
        self._tracks[track_id] = state
        while len(self._tracks) > self.config.max_track_memory:
            self._tracks.popitem(last=False)
        return state

    def inspect(self, boxes: list[Any], *, timestamp_seconds: float, frame_width: int, frame_height: int,
                new_ids: set[int], reacquired_ids: set[int]) -> list[dict[str, Any]]:
        overlapping: set[int] = set()
        for index, left in enumerate(boxes):
            if left.track_id is None:
                continue
            for right in boxes[index + 1:]:
                if right.track_id is not None and iou((left.x1, left.y1, left.x2, left.y2), (right.x1, right.y1, right.x2, right.y2)) >= self.config.occlusion_iou_threshold:
                    overlapping.update((int(left.track_id), int(right.track_id)))

        selected: list[dict[str, Any]] = []
        for box in boxes:
            if box.track_id is None:
                continue
            track_id = int(box.track_id)
            state = self._state(track_id)
            vector = self._vector(box, frame_width, frame_height)
            distance = None if state.last_vector is None else math.sqrt(sum((a - b) ** 2 for a, b in zip(vector[:4], state.last_vector[:4])))
            reasons: list[str] = []
            if track_id in reacquired_ids:
                reasons.append("REACQUIRED")
            if track_id in new_ids:
                reasons.append("TRANSITION")
            if float(box.confidence) < self.config.low_confidence_threshold:
                reasons.append("LOW_CONFIDENCE")
            if track_id in overlapping:
                reasons.append("OCCLUSION")
            if distance is not None and distance >= self.config.transition_distance:
                reasons.append("TRANSITION")
            if not reasons and float(box.confidence) >= self.config.stable_confidence_threshold and timestamp_seconds - state.last_selected_at >= self.config.representative_interval_seconds:
                reasons.append("STABLE_REPRESENTATIVE")

            reasons = list(dict.fromkeys(reasons))
            reasons = [reason for reason in reasons if timestamp_seconds - state.last_reason_at.get(reason, -1e12) >= self.config.event_cooldown_seconds]
            if not reasons:
                self.duplicates_rejected += 1
                continue
            if float(box.confidence) >= self.config.stable_confidence_threshold and track_id not in overlapping:
                self.quality_accepted += 1
            state.last_selected_at = timestamp_seconds
            state.last_vector = vector
            for reason in reasons:
                state.last_reason_at[reason] = timestamp_seconds
                self.counts[reason] += 1
            selected.append({
                "record_type": "SHADOW_CANDIDATE", "timestamp_seconds": timestamp_seconds,
                "anonymous_track_id": track_id, "reasons": reasons,
                "bbox_normalized": [box.x1 / frame_width, box.y1 / frame_height, box.x2 / frame_width, box.y2 / frame_height],
                "detector_confidence": float(box.confidence),
                "feature_vector": list(vector), "label_status": "UNLABELED_SHADOW_CANDIDATE",
                "ground_truth": None, "pseudo_label": None, "raw_video_recorded": False,
            })
        return selected

    def event_candidate(self, track_id: int, reason: str, timestamp_seconds: float) -> dict[str, Any] | None:
        state = self._state(int(track_id))
        if timestamp_seconds - state.last_reason_at.get(reason, -1e12) < self.config.event_cooldown_seconds:
            self.duplicates_rejected += 1
            return None
        state.last_reason_at[reason] = timestamp_seconds
        self.counts[reason] += 1
        return {
            "record_type": "SHADOW_CANDIDATE", "timestamp_seconds": timestamp_seconds,
            "anonymous_track_id": int(track_id), "reasons": [reason], "bbox_normalized": None,
            "detector_confidence": None, "feature_vector": None,
            "label_status": "UNLABELED_SHADOW_CANDIDATE", "ground_truth": None,
            "pseudo_label": None, "raw_video_recorded": False,
        }

    def summary(self) -> dict[str, Any]:
        pool_size = int(sum(self.counts.values()))
        return {
            "candidate_pool_size": pool_size,
            "track_memory_size": len(self._tracks),
            "quality_accepted_samples": self.quality_accepted,
            "duplicates_rejected": self.duplicates_rejected,
            "candidate_counts": dict(self.counts),
            "representative_samples": self.counts.get("STABLE_REPRESENTATIVE", 0) + self.counts.get("TRANSITION", 0),
            "uncertain_samples": self.counts.get("LOW_CONFIDENCE", 0) + self.counts.get("OCCLUSION", 0),
            "novel_or_ood_candidates": "NOT_EVALUATED_NO_REFERENCE_DISTRIBUTION",
        }


class LiveSessionState:
    """Thread-safe bounded state shared with the localhost dashboard."""

    def __init__(self, session_id: str, started_utc: str) -> None:
        self._lock = threading.Lock()
        self.data: dict[str, Any] = {
            "session_id": session_id, "started_utc": started_utc, "duration_seconds": 0.0,
            "source_state": "SOURCE_CONNECTING", "people_now": 0, "max_people": 0,
            "active_tracks": [], "new_track_events": 0, "lost_events": 0, "reacquired_events": 0,
            "short_lived_tracks": 0, "fragmentation_proxy": 0, "processed_frames": 0,
            "track_lifetime_seconds": {"median": None, "max": None},
            "fps": None, "latency_median_ms": None, "latency_p95_ms": None,
            "reconnect_count": 0, "overwritten_frames": 0, "recent_events": [],
            "data_collection": {}, "pose_pipeline": "POSE_PIPELINE_PENDING",
            "behavior_patterns": {"status": "TRACK_LIFECYCLE_PATTERNS_ONLY", "counts": {}},
            "shadow_learning": {"status": "SHADOW_DATA_POOL_BUILDING", "candidate_model": "NOT_TRAINED"},
            "detector_miss_markers": "NOT_AVAILABLE_UNCONTROLLED_SESSION",
            "privacy": {"raw_video_recorded": False, "snapshots_recorded": False, "identity": "ANONYMOUS_TRACK_IDS_ONLY"},
        }

    def update(self, **values: Any) -> None:
        with self._lock:
            self.data.update(values)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self.data)


def dashboard_html() -> str:
    return """<!doctype html><html lang='th'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>
<title>Posture CCTV Long Session</title><style>body{font:16px system-ui;background:#10151c;color:#eef2f7;margin:24px}h1{font-size:24px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}.card{background:#1b2531;border:1px solid #34475b;border-radius:12px;padding:16px}.v{font-size:25px;color:#61dafb}small{color:#9fb0c2}ul{padding-left:20px}</style></head>
<body><h1>ระบบเฝ้าดู CCTV แบบไม่ระบุตัวตน</h1><div id='grid' class='grid'></div><script>
const esc=x=>String(x??'—').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function refresh(){try{const s=await(await fetch('/api/status',{cache:'no-store'})).json();const d=s.data_collection||{},p=s.privacy||{},b=s.behavior_patterns||{};
document.getElementById('grid').innerHTML=`<div class=card><small>SESSION</small><div class=v>${esc(s.session_id)}</div><p>เวลา ${esc((s.duration_seconds/60).toFixed(1))} นาที<br>Source: ${esc(s.source_state)}</p></div>
<div class=card><small>PEOPLE</small><div class=v>${esc(s.people_now)} คน</div><p>สูงสุด ${esc(s.max_people)} | Track ${esc((s.active_tracks||[]).join(', '))}</p></div>
<div class=card><small>TRACKING HEALTH</small><div class=v>Lost ${esc(s.lost_events)} / Return ${esc(s.reacquired_events)}</div><p>New ${esc(s.new_track_events)} | Short ${esc(s.short_lived_tracks)} | Proxy ${esc(s.fragmentation_proxy)}<br>Lifetime median/max ${esc(s.track_lifetime_seconds?.median?.toFixed?.(1))}/${esc(s.track_lifetime_seconds?.max?.toFixed?.(1))} s<br>Detector marker: ${esc(s.detector_miss_markers)}</p></div>
<div class=card><small>PERFORMANCE</small><div class=v>${esc(s.fps?.toFixed?.(1))} FPS</div><p>Latency median ${esc(s.latency_median_ms?.toFixed?.(1))} ms | p95 ${esc(s.latency_p95_ms?.toFixed?.(1))} ms<br>Reconnect ${esc(s.reconnect_count)} | overwritten ${esc(s.overwritten_frames)}</p></div>
<div class=card><small>DATA COLLECTION</small><div class=v>${esc(s.processed_frames)} frames</div><p>Pool ${esc(d.candidate_pool_size)} | accepted ${esc(d.quality_accepted_samples)} | duplicate rejected ${esc(d.duplicates_rejected)}<br>representative ${esc(d.representative_samples)} | uncertain ${esc(d.uncertain_samples)} | OOD ${esc(d.novel_or_ood_candidates)}</p></div>
<div class=card><small>SHADOW LEARNING</small><div class=v>${esc(s.shadow_learning?.status)}</div><p>Pose: ${esc(s.pose_pipeline)}<br>Pattern: ${esc(b.status)}<br>Clustering: ${esc(s.shadow_learning?.clustering)}<br>Model: ${esc(s.shadow_learning?.candidate_model)}</p></div>
<div class=card><small>PRIVACY</small><div class=v>RAW VIDEO OFF</div><p>${esc(p.identity)}<br>ไม่มี ground truth อัตโนมัติ</p></div>
<div class=card><small>RECENT EVENTS</small><ul>${(s.recent_events||[]).slice(-8).reverse().map(e=>`<li>${esc(e)}</li>`).join('')}</ul></div>`;}catch(e){document.getElementById('grid').textContent='Dashboard waiting for runtime…'}}
refresh();setInterval(refresh,2000);</script></body></html>"""


class LocalDashboardServer:
    def __init__(self, state: LiveSessionState, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.state, self.host, self.port = state, host, int(port)
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> str:
        state = self.state
        html = dashboard_html().encode("utf-8")

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/api/status":
                    body, content_type = json.dumps(state.snapshot(), ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8"
                elif self.path in {"/", "/index.html"}:
                    body, content_type = html, "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200); self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body)

            def log_message(self, _format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, name="local-dashboard", daemon=True)
        self.thread.start()
        return f"http://{self.host}:{self.port}/"

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown(); self.server.server_close()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
        self.server = None; self.thread = None


def performance_summary(values: list[float]) -> dict[str, float | None]:
    return {"median": None if not values else median(values), "p95": _percentile(values, 0.95), "max": None if not values else max(values)}


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


__all__ = ["CandidateSelector", "CandidateSelectorConfig", "LiveSessionState", "LocalDashboardServer", "atomic_write_json", "iou", "performance_summary"]
