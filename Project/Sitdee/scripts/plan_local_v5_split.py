"""Plan participant-independent splits for future local V5 training.

This script only writes a proposed split plan. It never fits or tunes a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.posture_ai.data.local_rgb_v1 import LOCAL_PARTICIPANTS  # noqa: E402

MANIFEST_PATH = ROOT / "datasets" / "processed" / "local_rgb_v1" / "manifest.csv"
OUTPUT_PATH = ROOT / "outputs" / "metrics" / "local_v5_split_plan.json"
REPORT_PATH = ROOT / "reports" / "v5" / "local_v5_split_plan.md"


def _complete_participants() -> list[str]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    available = {participant: set() for participant in LOCAL_PARTICIPANTS}
    for row in rows:
        if row.get("status") in ("VALIDATED", "PASS"):
            available.setdefault(row["participant_id"], set()).add((row["session_id"], row["project_label"]))
    expected = {(f"S{i:02d}", label) for i in (1, 2) for label in ("TUP", "TLF", "TLB", "TLL", "TLR")}
    return [participant for participant in LOCAL_PARTICIPANTS if available.get(participant, set()) >= expected]


def plan() -> dict:
    participants = _complete_participants()
    result: dict = {
        "status": "WAITING_FOR_LOCAL_RGB_DATA" if not participants else ("SPLIT_PLAN_READY" if len(participants) >= 6 else "LOCAL_V5_DATA_PARTIALLY_READY"),
        "complete_participants": participants,
        "participant_overlap": [],
        "train": [],
        "validation": [],
        "test": [],
        "training_performed": False,
    }
    if len(participants) >= 6:
        test_count = max(2, round(len(participants) * 0.2))
        validation_count = max(1, round(len(participants) * 0.1))
        if test_count + validation_count >= len(participants):
            test_count, validation_count = 2, 1
        result["test"] = participants[-test_count:]
        result["validation"] = participants[-test_count - validation_count:-test_count]
        result["train"] = participants[: len(participants) - test_count - validation_count]
        sets = [set(result[key]) for key in ("train", "validation", "test")]
        result["participant_overlap"] = sorted((sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2]))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "# Local V5 participant split plan\n\n"
        f"Status: **{result['status']}**\n\n"
        f"Complete participants: `{', '.join(participants) or 'none'}`\n\n"
        f"- Train: `{', '.join(result['train']) or 'not planned yet'}`\n"
        f"- Validation: `{', '.join(result['validation']) or 'not planned yet'}`\n"
        f"- Test: `{', '.join(result['test']) or 'not planned yet'}`\n"
        f"- Participant overlap: `{result['participant_overlap']}`\n\n"
        "Both sessions from each participant remain in one split. This is only a "
        "proposed grouping plan; no training was executed.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(plan(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
