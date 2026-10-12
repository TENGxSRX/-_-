"""Read-only replay summary for an existing multi-person JSONL telemetry log.

The replay never opens a camera and never fabricates landmarks, detector order,
or identity labels.  It reports only fields present in the supplied log.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def analyze(path: Path) -> dict:
    rows = []
    invalid_lines = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                invalid_lines.append(line_number)
    per_id = defaultdict(list)
    subject_counts = Counter()
    for frame_index, row in enumerate(rows):
        subjects = row.get("subjects", [])
        subject_counts[int(row.get("subject_count", len(subjects)))] += 1
        for subject in subjects:
            subject_id = subject.get("subject_id")
            if subject_id:
                per_id[subject_id].append((frame_index, row.get("timestamp_seconds"), subject.get("tracking_state")))
    first_seen = {}
    for frame_index, row in enumerate(rows):
        for subject in row.get("subjects", []):
            subject_id = subject.get("subject_id")
            if subject_id and subject_id not in first_seen:
                first_seen[subject_id] = {"frame_index": frame_index, "timestamp_seconds": row.get("timestamp_seconds")}
    temporary_lost_ids = sorted(
        subject_id
        for subject_id, events in per_id.items()
        if any(state == "TEMPORARILY_LOST" for _frame, _timestamp, state in events)
    )
    return {
        "replay_status": "REAL_TELEMETRY_REPLAY_WITH_LIMITATIONS",
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "telemetry_frames": len(rows),
        "invalid_json_lines": invalid_lines,
        "inference_frames": sum(not bool(row.get("inference_skipped", False)) for row in rows),
        "skipped_frames": sum(bool(row.get("inference_skipped", False)) for row in rows),
        "unique_person_ids": sorted(per_id),
        "new_id_creation_events": len(first_seen),
        "max_simultaneous_subjects": max(subject_counts, default=0),
        "subject_count_distribution": {str(key): value for key, value in sorted(subject_counts.items())},
        "ids_with_temporary_lost": temporary_lost_ids,
        "raw_detector_order_available": any("detector_index" in subject for row in rows for subject in row.get("subjects", [])),
        "raw_landmarks_available": any("landmarks" in subject for row in rows for subject in row.get("subjects", [])),
        "normalized_center_available": any("center" in subject for row in rows for subject in row.get("subjects", [])),
        "track_first_seen": first_seen,
        "limitation": "Telemetry does not contain raw detector order, raw landmarks, or ground-truth biological identity; no missing values were reconstructed.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path("outputs/multiperson_test.jsonl"))
    parser.add_argument("--output", type=Path, help="Optional JSON summary path")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.log.is_file():
        parser.error(f"log does not exist: {args.log}")
    summary = analyze(args.log)
    if args.output is not None:
        if args.output.exists() and not args.overwrite:
            parser.error(f"output exists; choose another path or pass --overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote read-only replay summary to {args.output}")
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
