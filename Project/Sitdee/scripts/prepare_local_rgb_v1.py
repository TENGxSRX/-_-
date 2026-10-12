"""Create the empty, anonymous local RGB V1 collection layout."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.posture_ai.data.local_rgb_v1 import (  # noqa: E402
    LOCAL_MANIFEST_COLUMNS,
    LOCAL_PARTICIPANTS,
    LOCAL_POSTURES,
    LOCAL_SESSIONS,
    expected_manifest_rows,
)

RAW_ROOT = ROOT / "datasets" / "raw" / "local_rgb_v1"
PROCESSED_ROOT = ROOT / "datasets" / "processed" / "local_rgb_v1"


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]], *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare() -> dict[str, int | str]:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    for participant_id in LOCAL_PARTICIPANTS:
        participant_number = int(participant_id[1:])
        for session_id in LOCAL_SESSIONS:
            session_number = int(session_id[1:])
            for posture in LOCAL_POSTURES:
                folder = RAW_ROOT / f"participant_{participant_number:02d}" / f"session_{session_number:02d}" / posture
                folder.mkdir(parents=True, exist_ok=True)
                placeholder = folder / ".gitkeep"
                if not placeholder.exists():
                    placeholder.write_text("", encoding="utf-8")

    participant_rows = [
        {
            "participant_id": participant_id,
            "session_count": "2",
            "age_group": "",
            "height_cm_optional": "",
            "notes": "Anonymous ID only; fill after consent if needed.",
        }
        for participant_id in LOCAL_PARTICIPANTS
    ]
    _write_csv(
        RAW_ROOT / "participants.csv",
        ("participant_id", "session_count", "age_group", "height_cm_optional", "notes"),
        participant_rows,
        overwrite=False,
    )

    session_rows = [
        {
            "participant_id": participant_id,
            "session_id": session_id,
            "recording_date": "",
            "camera_device_optional": "",
            "camera_view": "",
            "camera_distance_estimate": "",
            "lighting": "",
            "chair_type_optional": "",
            "desk_present": "",
            "notes": "Fill after recording; do not store sensitive identifiers.",
        }
        for participant_id in LOCAL_PARTICIPANTS
        for session_id in LOCAL_SESSIONS
    ]
    _write_csv(
        RAW_ROOT / "sessions.csv",
        (
            "participant_id",
            "session_id",
            "recording_date",
            "camera_device_optional",
            "camera_view",
            "camera_distance_estimate",
            "lighting",
            "chair_type_optional",
            "desk_present",
            "notes",
        ),
        session_rows,
        overwrite=False,
    )

    expected_rows = expected_manifest_rows()
    manifest_path = PROCESSED_ROOT / "manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            existing_rows = list(csv.DictReader(handle))
        existing_by_sample = {row.get("sample_id", ""): row for row in existing_rows}
        rows = []
        for expected in expected_rows:
            row = dict(expected)
            row.update(existing_by_sample.get(expected["sample_id"], {}))
            # These identity fields are controlled by the slot and cannot drift.
            for key in ("sample_id", "participant_id", "session_id", "video_path", "source_label", "project_label", "recording_type"):
                row[key] = expected[key]
            rows.append(row)
        expected_ids = {row["sample_id"] for row in expected_rows}
        rows.extend(row for row in existing_rows if row.get("sample_id") not in expected_ids)
    else:
        rows = expected_rows
    _write_csv(PROCESSED_ROOT / "manifest.csv", LOCAL_MANIFEST_COLUMNS, rows)
    return {"participants": len(LOCAL_PARTICIPANTS), "sessions": len(session_rows), "recording_slots": len(rows), "status": "WAITING_FOR_LOCAL_RGB_DATA"}


if __name__ == "__main__":
    import json

    print(json.dumps(prepare(), indent=2, ensure_ascii=False))
