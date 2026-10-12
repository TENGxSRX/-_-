"""Read-only status report for the local RGB V1 recording layout."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.posture_ai.data.local_rgb_v1 import (  # noqa: E402
    LOCAL_PARTICIPANTS,
    LOCAL_POSTURES,
    LOCAL_SESSIONS,
    LOCAL_VIDEO_SUFFIXES,
    parse_local_video_filename,
    validate_video_path,
)

RAW_ROOT = ROOT / "datasets" / "raw" / "local_rgb_v1"


def inspect_collection() -> dict:
    files = sorted(
        path
        for path in RAW_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in LOCAL_VIDEO_SUFFIXES
    ) if RAW_ROOT.exists() else []
    valid: list[dict[str, str | Path]] = []
    invalid: list[dict[str, str]] = []
    present_by_slot: set[str] = set()
    posture_counts = Counter()
    participant_counts = Counter()
    session_counts = Counter()
    for path in files:
        parsed, errors = validate_video_path(path, RAW_ROOT)
        if parsed is None or errors:
            invalid.append({"path": str(path), "errors": "; ".join(errors)})
            continue
        slot = f"{parsed['participant']}_{parsed['session']}_{parsed['posture']}_01"
        if slot in present_by_slot:
            invalid.append({"path": str(path), "errors": f"Duplicate slot: {slot}"})
            continue
        present_by_slot.add(slot)
        valid.append({"path": path, **parsed})
        posture_counts[parsed["posture"]] += 1
        participant_counts[parsed["participant"]] += 1
        session_counts[f"{parsed['participant']}/{parsed['session']}"] += 1

    expected_slots = {
        f"{participant}_{session}_{posture}_01"
        for participant in LOCAL_PARTICIPANTS
        for session in LOCAL_SESSIONS
        for posture in LOCAL_POSTURES
    }
    missing = sorted(expected_slots - present_by_slot)
    complete_participants = sorted(
        participant for participant in LOCAL_PARTICIPANTS
        if all(f"{participant}_{session}_{posture}_01" in present_by_slot for session in LOCAL_SESSIONS for posture in LOCAL_POSTURES)
    )
    complete_sessions = sorted(
        f"{participant}/{session}"
        for participant in LOCAL_PARTICIPANTS
        for session in LOCAL_SESSIONS
        if all(f"{participant}_{session}_{posture}_01" in present_by_slot for posture in LOCAL_POSTURES)
    )
    return {
        "status": "READY_FOR_PREFLIGHT" if len(present_by_slot) == len(expected_slots) else "WAITING_FOR_LOCAL_RGB_DATA",
        "participants_complete": len(complete_participants),
        "participants_complete_ids": complete_participants,
        "sessions_complete": len(complete_sessions),
        "sessions_complete_ids": complete_sessions,
        "videos_present": len(present_by_slot),
        "videos_expected": len(expected_slots),
        "posture_classes_present": {posture: posture_counts.get(posture, 0) for posture in LOCAL_POSTURES},
        "participants_with_files": {participant: participant_counts.get(participant, 0) for participant in LOCAL_PARTICIPANTS},
        "sessions_with_files": dict(sorted(session_counts.items())),
        "missing": missing,
        "invalid_filenames_or_locations": invalid,
    }


def _print_human(report: dict) -> None:
    print(f"Status: {report['status']}")
    print(f"Participants complete: {report['participants_complete']} / 10")
    print(f"Sessions complete: {report['sessions_complete']} / 20")
    print(f"Videos: {report['videos_present']} / {report['videos_expected']}")
    print("Posture classes:")
    for label, count in report["posture_classes_present"].items():
        print(f"  {label}: {count} / 20")
    print(f"Missing: {len(report['missing'])}")
    for item in report["missing"][:20]:
        print(f"  {item}")
    if len(report["missing"]) > 20:
        print(f"  ... and {len(report['missing']) - 20} more")
    print(f"Invalid filenames/locations: {len(report['invalid_filenames_or_locations'])}")
    for item in report["invalid_filenames_or_locations"]:
        print(f"  {item['path']}: {item['errors']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human-readable report")
    args = parser.parse_args()
    report = inspect_collection()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_human(report)


if __name__ == "__main__":
    main()
