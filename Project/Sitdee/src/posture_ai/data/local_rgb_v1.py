"""Contracts for the anonymous local RGB collection V1.

This module contains only deterministic naming, slot, and manifest helpers.
It never infers a label from image content; the controlled folder/name label is
the participant-instruction label and must be audited separately.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

LOCAL_POSTURES = ("TUP", "TLF", "TLB", "TLL", "TLR")
LOCAL_PARTICIPANTS = tuple(f"P{i:02d}" for i in range(1, 11))
LOCAL_SESSIONS = tuple(f"S{i:02d}" for i in range(1, 3))
LOCAL_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")
LOCAL_MANIFEST_COLUMNS = (
    "sample_id",
    "participant_id",
    "session_id",
    "video_path",
    "source_label",
    "project_label",
    "recording_type",
    "camera_view",
    "duration_seconds",
    "fps",
    "width",
    "height",
    "total_frames",
    "pose_detected_frames",
    "pose_detection_rate",
    "core_accepted_frames",
    "core_acceptance_rate",
    "status",
    "notes",
)

_FILENAME_RE = re.compile(
    r"^(?P<participant>P\d{2})_(?P<session>S\d{2})_(?P<posture>TUP|TLF|TLB|TLL|TLR)_(?P<take>\d{2})"
    r"(?P<extension>\.mp4|\.avi|\.mov|\.mkv|\.webm)$",
    re.IGNORECASE,
)


def parse_local_video_filename(filename: str | Path) -> dict[str, str]:
    """Parse the controlled V1 filename or raise a clear ValueError."""
    name = Path(filename).name
    match = _FILENAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(
            "Invalid local V1 filename; expected P01_S01_TUP_01.mp4-style name: "
            f"{name}"
        )
    values = {key: value.upper() for key, value in match.groupdict().items()}
    if values["participant"] not in LOCAL_PARTICIPANTS:
        raise ValueError(f"Participant is outside P01-P10: {values['participant']}")
    if values["session"] not in LOCAL_SESSIONS:
        raise ValueError(f"Session is outside S01-S02: {values['session']}")
    if values["posture"] not in LOCAL_POSTURES:
        raise ValueError(f"Unknown local posture: {values['posture']}")
    if values["take"] != "01":
        raise ValueError("Local V1 currently expects take suffix _01")
    return values


def expected_sample_id(participant_id: str, session_id: str, posture: str) -> str:
    participant = participant_id.upper()
    session = session_id.upper()
    label = posture.upper()
    if participant not in LOCAL_PARTICIPANTS:
        raise ValueError(f"Unknown participant: {participant_id}")
    if session not in LOCAL_SESSIONS:
        raise ValueError(f"Unknown session: {session_id}")
    if label not in LOCAL_POSTURES:
        raise ValueError(f"Unknown posture: {posture}")
    return f"{participant}_{session}_{label}_01"


def expected_video_relative_path(participant_id: str, session_id: str, posture: str) -> str:
    sample_id = expected_sample_id(participant_id, session_id, posture)
    participant_folder = f"participant_{int(participant_id[1:]):02d}"
    session_folder = f"session_{int(session_id[1:]):02d}"
    return (
        f"datasets/raw/local_rgb_v1/{participant_folder}/{session_folder}/{posture.upper()}/"
        f"{sample_id}.mp4"
    )


def expected_manifest_rows() -> list[dict[str, str]]:
    """Build the 100 planned recording slots with no fabricated media facts."""
    rows: list[dict[str, str]] = []
    for participant_id in LOCAL_PARTICIPANTS:
        for session_id in LOCAL_SESSIONS:
            for posture in LOCAL_POSTURES:
                rows.append(
                    {
                        "sample_id": expected_sample_id(participant_id, session_id, posture),
                        "participant_id": participant_id,
                        "session_id": session_id,
                        "video_path": expected_video_relative_path(participant_id, session_id, posture),
                        "source_label": posture,
                        "project_label": posture,
                        "recording_type": "video",
                        "camera_view": "",
                        "duration_seconds": "",
                        "fps": "",
                        "width": "",
                        "height": "",
                        "total_frames": "",
                        "pose_detected_frames": "",
                        "pose_detection_rate": "",
                        "core_accepted_frames": "",
                        "core_acceptance_rate": "",
                        "status": "WAITING_FOR_VIDEO",
                        "notes": "Planned slot; label comes from collection instruction, not visual inference.",
                    }
                )
    return rows


def validate_video_path(path: str | Path, raw_root: str | Path) -> tuple[dict[str, str] | None, list[str]]:
    """Validate filename and participant/session/posture folder agreement."""
    video_path = Path(path)
    root = Path(raw_root).resolve()
    errors: list[str] = []
    try:
        parsed = parse_local_video_filename(video_path.name)
    except ValueError as exc:
        return None, [str(exc)]
    try:
        relative = video_path.resolve().relative_to(root)
    except ValueError:
        return parsed, [f"Video is outside local collection root: {video_path}"]
    parts = relative.parts
    expected_parts = (
        f"participant_{int(parsed['participant'][1:]):02d}",
        f"session_{int(parsed['session'][1:]):02d}",
        parsed["posture"],
    )
    if len(parts) < 4 or tuple(parts[-4:-1]) != expected_parts:
        errors.append(
            "Folder does not match filename; expected "
            f"{'/'.join(expected_parts)}/{video_path.name}"
        )
    if video_path.suffix.lower() not in LOCAL_VIDEO_SUFFIXES:
        errors.append(f"Unsupported video extension: {video_path.suffix}")
    return parsed, errors


def validate_manifest_schema(rows: list[Mapping[str, object]]) -> None:
    for row in rows:
        missing = [column for column in LOCAL_MANIFEST_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"Local RGB manifest row missing columns: {missing}")
        if row.get("recording_type") != "video":
            raise ValueError("Local RGB V1 currently supports video recordings only")
        if str(row.get("project_label", "")) not in LOCAL_POSTURES:
            raise ValueError(f"Invalid project label in local manifest: {row.get('project_label')}")


__all__ = [
    "LOCAL_MANIFEST_COLUMNS",
    "LOCAL_PARTICIPANTS",
    "LOCAL_POSTURES",
    "LOCAL_SESSIONS",
    "LOCAL_VIDEO_SUFFIXES",
    "expected_manifest_rows",
    "expected_sample_id",
    "expected_video_relative_path",
    "parse_local_video_filename",
    "validate_manifest_schema",
    "validate_video_path",
]
