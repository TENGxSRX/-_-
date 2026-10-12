"""Fail when participant identities leak across Roboflow train/valid/test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def participant(filename: str) -> str:
    parts = Path(filename).stem.split("__")
    if len(parts) < 4 or not parts[0]:
        raise ValueError(f"filename lacks participant metadata: {filename}")
    return parts[0]


def audit(root: Path) -> dict:
    splits = {}
    for split in ("train", "valid", "test"):
        directory = root / split / "images"
        files = sorted(p for p in directory.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if directory.exists() else []
        splits[split] = {participant(p.name) for p in files}
    overlaps = {f"{a}_{b}": sorted(splits[a] & splits[b]) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    leaking = {key: value for key, value in overlaps.items() if value}
    result = {"status": "FAIL_PARTICIPANT_LEAKAGE" if leaking else "PASS", "participants": {k: sorted(v) for k, v in splits.items()}, "overlaps": overlaps}
    if leaking:
        raise ValueError(json.dumps(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Roboflow export containing train/valid/test/images")
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

