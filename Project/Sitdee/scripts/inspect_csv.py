"""Streamed CSV verification for the office-posture dataset collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path


MISSING = {"", "na", "n/a", "nan", "null", "none"}


def is_number(value: str) -> bool:
    try:
        x = float(value.strip())
        return math.isfinite(x)
    except (TypeError, ValueError):
        return False


def inspect_csv(path: Path, duplicate_check: bool = True) -> dict:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)

    row_count = 0
    header: list[str] = []
    missing: Counter[str] = Counter()
    value_counts: dict[str, Counter[str]] = {}
    numeric_stats: dict[str, dict[str, float | int | None]] = {}
    id_candidates: dict[str, set[str]] = {}
    low_cardinality: dict[str, Counter[str]] = {}
    duplicate_rows = None
    duplicate_db_path = None
    connection = None
    cursor = None

    if duplicate_check:
        fd, duplicate_db_path = tempfile.mkstemp(prefix="csv_row_hashes_", suffix=".sqlite")
        os.close(fd)
        connection = sqlite3.connect(duplicate_db_path)
        cursor = connection.cursor()
        cursor.execute("PRAGMA journal_mode=OFF")
        cursor.execute("PRAGMA synchronous=OFF")
        cursor.execute("CREATE TABLE row_hashes (h BLOB PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            try:
                header = next(reader)
            except StopIteration:
                return {
                    "path": str(path),
                    "size_bytes": byte_count,
                    "sha256": digest.hexdigest(),
                    "rows": 0,
                    "columns": 0,
                    "header": [],
                    "empty": True,
                }

            for name in header:
                low_cardinality[name] = Counter()
                value_counts[name] = Counter()
                numeric_stats[name] = {"count": 0, "min": None, "max": None}
                lowered = name.lower()
                if any(token in lowered for token in ("subject", "participant", "person", "session", "video", "frame", "clip", "record", "sample", "id")):
                    id_candidates[name] = set()

            for row in reader:
                row_count += 1
                if len(row) != len(header):
                    missing["__malformed_row_length__"] += 1
                padded = row + [""] * max(0, len(header) - len(row))
                for idx, name in enumerate(header):
                    value = padded[idx].strip() if idx < len(padded) else ""
                    if value.lower() in MISSING:
                        missing[name] += 1
                    if value and len(value_counts[name]) < 10000:
                        value_counts[name][value] += 1
                    if value and len(low_cardinality[name]) <= 200:
                        low_cardinality[name][value] += 1
                    if name in id_candidates and value and len(id_candidates[name]) < 100000:
                        id_candidates[name].add(value)
                    if is_number(value):
                        numeric = float(value)
                        stats = numeric_stats[name]
                        stats["count"] = int(stats["count"]) + 1
                        stats["min"] = numeric if stats["min"] is None else min(float(stats["min"]), numeric)
                        stats["max"] = numeric if stats["max"] is None else max(float(stats["max"]), numeric)

                if cursor is not None:
                    encoded = "\x1f".join(row).encode("utf-8")
                    row_hash = hashlib.sha256(encoded).digest()
                    cursor.execute(
                        "INSERT INTO row_hashes(h,n) VALUES(?,1) ON CONFLICT(h) DO UPDATE SET n=n+1",
                        (row_hash,),
                    )
                if cursor is not None and row_count % 10000 == 0:
                    connection.commit()
            if connection is not None:
                connection.commit()

        if cursor is not None:
            duplicate_rows = int(cursor.execute("SELECT COALESCE(SUM(n-1),0) FROM row_hashes WHERE n > 1").fetchone()[0])
    finally:
        if connection is not None:
            connection.close()
        if duplicate_db_path:
            try:
                os.unlink(duplicate_db_path)
            except FileNotFoundError:
                pass

    label_columns = {}
    for name in header:
        lowered = name.lower()
        if "label" in lowered or "class" in lowered or "posture" in lowered:
            label_columns[name] = dict(value_counts[name])

    id_summary = {}
    for name, values in id_candidates.items():
        id_summary[name] = {
            "unique_count_observed": len(values),
            "capped_at": 100000,
            "sample": sorted(values)[:50],
        }

    return {
        "path": str(path),
        "size_bytes": byte_count,
        "sha256": digest.hexdigest(),
        "rows": row_count,
        "columns": len(header),
        "header": header,
        "missing_values_by_column": dict(missing),
        "duplicate_rows": duplicate_rows,
        "label_columns": label_columns,
        "id_like_columns": id_summary,
        "numeric_stats": numeric_stats,
        "low_cardinality_values": {k: dict(v) for k, v in low_cardinality.items() if len(v) <= 200},
        "media_pipe_feature_columns": [
            name for name in header if name.lower().endswith(("_x", "_y", "_z"))
        ],
        "well_formed_rows": row_count - missing.get("__malformed_row_length__", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--skip-duplicates", action="store_true")
    args = parser.parse_args()
    result = inspect_csv(args.csv_path, duplicate_check=not args.skip_duplicates)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

