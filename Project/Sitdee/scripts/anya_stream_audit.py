"""Bounded-memory ANYA CSV audit and four-file comparison.

The CSVs are treated as immutable sources. This script reads them sequentially
with pandas chunks and hashes raw serialized rows without materializing a full
file in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_FILES = ("ANN.csv", "Random_Forest.csv", "lgbm.csv", "xgb.csv")
LABELISH = re.compile(r"label|target|class|prediction|predicted|cluster", re.I)
COORDINATE = re.compile(r"^(x|y|z)\d+(normalized)?$", re.I)
VISIBILITY = re.compile(r"^vis\d+$", re.I)


def _hash_serialized_rows(path: Path, max_rows: int | None = None) -> dict:
    all_digest = hashlib.sha256()
    core_digest = hashlib.sha256()
    prediction_digest = hashlib.sha256()
    duplicate_hashes: set[bytes] = set()
    duplicate_rows = 0
    duplicate_core_hashes: set[bytes] = set()
    duplicate_core_rows = 0
    rows = 0
    prediction_counts: Counter[str] = Counter()
    with path.open("rb") as handle:
        header = handle.readline().rstrip(b"\r\n")
        for raw in handle:
            if max_rows is not None and rows >= max_rows:
                break
            line = raw.rstrip(b"\r\n")
            if not line:
                continue
            rows += 1
            all_digest.update(line)
            all_digest.update(b"\n")
            _separator, _comma, prediction = line.rpartition(b",")
            core = _separator
            core_digest.update(core)
            core_digest.update(b"\n")
            core_hash = hashlib.blake2b(core, digest_size=16).digest()
            if core_hash in duplicate_core_hashes:
                duplicate_core_rows += 1
            else:
                duplicate_core_hashes.add(core_hash)
            prediction_digest.update(prediction)
            prediction_digest.update(b"\n")
            prediction_counts[prediction.decode("utf-8", errors="replace")] += 1
            row_hash = hashlib.blake2b(line, digest_size=16).digest()
            if row_hash in duplicate_hashes:
                duplicate_rows += 1
            else:
                duplicate_hashes.add(row_hash)
    return {
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "serialized_rows": rows,
        "serialized_all_sha256": all_digest.hexdigest(),
        "serialized_core_sha256": core_digest.hexdigest(),
        "serialized_prediction_sha256": prediction_digest.hexdigest(),
        "serialized_duplicate_hash_rows": duplicate_rows,
        "serialized_duplicate_core_hash_rows": duplicate_core_rows,
        "serialized_prediction_counts": dict(prediction_counts),
    }


def _add_numeric_stats(stats: dict, column: str, values: np.ndarray) -> None:
    finite = np.isfinite(values)
    if (~finite).any():
        stats[column]["non_finite"] += int((~finite).sum())
    values = values[finite]
    if values.size == 0:
        return
    item = stats[column]
    item["count"] += int(values.size)
    item["sum"] += float(values.sum(dtype=np.float64))
    item["sum_sq"] += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
    item["min"] = min(item["min"], float(values.min()))
    item["max"] = max(item["max"], float(values.max()))


def _stream_file(path: Path, chunk_size: int, max_rows: int | None) -> dict:
    started = time.perf_counter()
    first_chunk: pd.DataFrame | None = None
    rows = 0
    null_counts: Counter[str] = Counter()
    numeric_stats: dict[str, dict] = {}
    dtypes: dict[str, str] = {}
    categorical_columns: list[str] = []
    categorical_counts: dict[str, Counter[str]] = {}
    low_card_numeric_columns: list[str] = []
    low_card_numeric_counts: dict[str, Counter[str]] = {}
    previous_frame = None
    frame_decreases = 0
    frame_contiguous_pairs = 0
    previous_filename = None
    filename_transitions = 0
    filename_blocks = 0
    previous_prediction = None
    prediction_transitions = 0
    prediction_blocks = 0
    prediction_run_lengths: Counter[str] = Counter()
    current_prediction_run = 0
    sample_parts: list[pd.DataFrame] = []

    # The official files are comma-delimited UTF-8 CSVs. Keeping these parser
    # settings explicit makes the ingestion decision auditable and prevents a
    # locale-specific default from silently changing the schema.
    reader = pd.read_csv(path, chunksize=chunk_size, low_memory=False, sep=",", encoding="utf-8")
    for chunk in reader:
        if max_rows is not None and rows >= max_rows:
            break
        if max_rows is not None and rows + len(chunk) > max_rows:
            chunk = chunk.iloc[: max_rows - rows].copy()
        if first_chunk is None:
            first_chunk = chunk.copy()
            dtypes = {str(col): str(dtype) for col, dtype in chunk.dtypes.items()}
            for col in chunk.columns:
                if (pd.api.types.is_object_dtype(chunk[col]) or pd.api.types.is_string_dtype(chunk[col])) and chunk[col].nunique(dropna=True) <= 1000:
                    categorical_columns.append(str(col))
                    categorical_counts[str(col)] = Counter()
                if pd.api.types.is_numeric_dtype(chunk[col]):
                    numeric_stats[str(col)] = {
                        "count": 0,
                        "non_finite": 0,
                        "sum": 0.0,
                        "sum_sq": 0.0,
                        "min": float("inf"),
                        "max": float("-inf"),
                    }
                    if chunk[col].nunique(dropna=True) <= 100:
                        low_card_numeric_columns.append(str(col))
                        low_card_numeric_counts[str(col)] = Counter()
        rows += len(chunk)
        null_counts.update({str(col): int(value) for col, value in chunk.isna().sum().items() if value})
        for col in numeric_stats:
            values = chunk[col].to_numpy(dtype=np.float64, na_value=np.nan)
            _add_numeric_stats(numeric_stats, col, values)
        for col in categorical_columns:
            values = chunk[col].dropna().astype(str).value_counts()
            categorical_counts[col].update({str(key): int(value) for key, value in values.items()})
        for col in low_card_numeric_columns:
            values = chunk[col].dropna().astype(str).value_counts()
            low_card_numeric_counts[col].update({str(key): int(value) for key, value in values.items()})
        if "frame" in chunk.columns:
            for value in chunk["frame"].dropna().to_numpy():
                value = int(value)
                if previous_frame is not None:
                    if value < previous_frame:
                        frame_decreases += 1
                    if value == previous_frame + 1:
                        frame_contiguous_pairs += 1
                previous_frame = value
        if "filename" in chunk.columns:
            for value in chunk["filename"].fillna("<NA>").astype(str):
                if previous_filename is None or value != previous_filename:
                    if previous_filename is not None:
                        filename_transitions += 1
                    filename_blocks += 1
                previous_filename = value
        if "Predicted_Label" in chunk.columns:
            for value in chunk["Predicted_Label"].fillna("<NA>").astype(str):
                if previous_prediction is None or value != previous_prediction:
                    if previous_prediction is not None:
                        prediction_transitions += 1
                    if previous_prediction is not None:
                        prediction_run_lengths[previous_prediction] += current_prediction_run
                    prediction_blocks += 1
                    current_prediction_run = 1
                else:
                    current_prediction_run += 1
                previous_prediction = value
        if len(sample_parts) == 0:
            sample_parts.append(chunk.head(10000).copy())

    if previous_prediction is not None:
        prediction_run_lengths[previous_prediction] += current_prediction_run
    sample = sample_parts[0] if sample_parts else pd.DataFrame()
    for item in numeric_stats.values():
        if item["count"]:
            mean = item["sum"] / item["count"]
            variance = max(item["sum_sq"] / item["count"] - mean * mean, 0.0)
            item["mean"] = mean
            item["std"] = float(np.sqrt(variance))
        else:
            item["mean"] = None
            item["std"] = None
        item.pop("sum", None)
        item.pop("sum_sq", None)
        if item["min"] == float("inf"):
            item["min"] = None
            item["max"] = None
    sample_ranges = {}
    for col in sample.columns:
        if col in numeric_stats:
            values = pd.to_numeric(sample[col], errors="coerce").to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                sample_ranges[str(col)] = {"min": float(finite.min()), "max": float(finite.max())}
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": [str(col) for col in (first_chunk.columns if first_chunk is not None else [])],
        "column_count": int(len(first_chunk.columns)) if first_chunk is not None else 0,
        "dtypes": dtypes,
        "null_counts": dict(null_counts),
        "total_null_cells": int(sum(null_counts.values())),
        "numeric_stats": numeric_stats,
        "categorical_columns": categorical_columns,
        "categorical_counts": {col: dict(counts) for col, counts in categorical_counts.items()},
        "low_card_numeric_columns": low_card_numeric_columns,
        "low_card_numeric_counts": {col: dict(counts) for col, counts in low_card_numeric_counts.items()},
        "parse_config": {"delimiter": ",", "encoding": "utf-8", "pandas_low_memory": False},
        "sample_ranges": sample_ranges,
        "frame_decreases": frame_decreases,
        "frame_contiguous_pairs": frame_contiguous_pairs,
        "filename_transitions": filename_transitions,
        "filename_blocks": filename_blocks,
        "prediction_transitions": prediction_transitions,
        "prediction_blocks": prediction_blocks,
        "prediction_longest_run_by_value": dict(prediction_run_lengths),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def _quality_summary(file_result: dict) -> dict:
    numeric = file_result["numeric_stats"]
    constant = []
    nonfinite = 0
    coordinate_ranges = {}
    for col, stats in numeric.items():
        nonfinite += stats["non_finite"]
        if stats["count"] and stats["min"] == stats["max"]:
            constant.append(col)
        if COORDINATE.match(col) or VISIBILITY.match(col):
            prefix = "visibility" if VISIBILITY.match(col) else ("normalized_xyz" if col.endswith("normalized") else "xyz")
            coordinate_ranges.setdefault(prefix, {"min": float("inf"), "max": float("-inf")})
            coordinate_ranges[prefix]["min"] = min(coordinate_ranges[prefix]["min"], stats["min"])
            coordinate_ranges[prefix]["max"] = max(coordinate_ranges[prefix]["max"], stats["max"])
    return {
        "constant_numeric_columns": constant,
        "non_finite_numeric_cells": nonfinite,
        "coordinate_group_ranges": coordinate_ranges,
    }


def run_audit(root: Path, names: list[str], chunk_size: int, max_rows: int | None) -> dict:
    results = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        stream = _stream_file(path, chunk_size, max_rows)
        serialized = _hash_serialized_rows(path, max_rows)
        if stream["rows"] != serialized["serialized_rows"]:
            raise RuntimeError(f"Pandas/raw row count mismatch for {name}: {stream['rows']} vs {serialized['serialized_rows']}")
        stream["serialized"] = serialized
        stream["quality"] = _quality_summary(stream)
        results[name] = stream
    header_sets = {name: tuple(result["columns"]) for name, result in results.items()}
    core_hashes = {name: result["serialized"]["serialized_core_sha256"] for name, result in results.items()}
    all_hashes = {name: result["serialized"]["serialized_all_sha256"] for name, result in results.items()}
    prediction_hashes = {name: result["serialized"]["serialized_prediction_sha256"] for name, result in results.items()}
    comparison = {
        "same_schema": len(set(header_sets.values())) == 1,
        "same_row_count": len({result["rows"] for result in results.values()}) == 1,
        "same_serialized_core_row_sequence": len(set(core_hashes.values())) == 1,
        "same_complete_row_sequence": len(set(all_hashes.values())) == 1,
        "same_prediction_sequence": len(set(prediction_hashes.values())) == 1,
        "core_sha256_by_file": core_hashes,
        "complete_sha256_by_file": all_hashes,
        "prediction_sha256_by_file": prediction_hashes,
        "canonical_file_candidate": names[0] if len(set(core_hashes.values())) == 1 else "NONE",
    }
    return {"chunk_size": chunk_size, "max_rows": max_rows, "files": results, "comparison": comparison}


def _write_report(payload: dict, path: Path) -> None:
    comparison = payload["comparison"]
    lines = [
        "# ANYA file-level streaming audit",
        "",
        "This report was generated from the official Kaggle download. Original CSVs were read sequentially and were not modified.",
        "",
        f"- Chunk size: {payload['chunk_size']:,} rows",
        f"- Complete scan: {'yes' if payload['max_rows'] is None else 'no; max rows=' + str(payload['max_rows'])}",
        f"- Same schema: {comparison['same_schema']}",
        f"- Same row count: {comparison['same_row_count']}",
        f"- Same serialized core row sequence excluding final prediction field: {comparison['same_serialized_core_row_sequence']}",
        f"- Same complete row sequence: {comparison['same_complete_row_sequence']}",
        f"- Same prediction sequence: {comparison['same_prediction_sequence']}",
        f"- Canonical file candidate: **{comparison['canonical_file_candidate']}**",
        "",
        "| File | Bytes | Rows | Columns | Null cells | Non-finite numeric cells | Duplicate full rows | Duplicate core rows | Parse seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["files"].items():
        lines.append(
            f"| {name} | {result['bytes']:,} | {result['rows']:,} | {result['column_count']} | {result['total_null_cells']:,} | {result['quality']['non_finite_numeric_cells']:,} | {result['serialized']['serialized_duplicate_hash_rows']:,} | {result['serialized']['serialized_duplicate_core_hash_rows']:,} | {result['elapsed_seconds']:.1f} |"
        )
    lines += ["", "## Categorical/prediction observations", ""]
    for name, result in payload["files"].items():
        lines.append(f"### `{name}`")
        for col in result["categorical_columns"]:
            counts = result["categorical_counts"].get(col, {})
            lines.append(f"- `{col}` unique values: {len(counts):,}; counts: `{counts}`")
        for col in result.get("low_card_numeric_columns", []):
            counts = result["low_card_numeric_counts"].get(col, {})
            lines.append(f"- Low-cardinality numeric `{col}` unique values: {len(counts):,}; counts: `{counts}`")
        pred = result["serialized"]["serialized_prediction_counts"]
        lines.append(f"- Serialized final-field values: `{pred}`")
        lines.append(f"- Filename blocks: {result['filename_blocks']:,}; filename transitions: {result['filename_transitions']:,}")
        lines.append(f"- Frame decreases: {result['frame_decreases']:,}; adjacent frame increments: {result['frame_contiguous_pairs']:,}")
        lines.append(f"- Prediction blocks: {result['prediction_blocks']:,}; prediction transitions: {result['prediction_transitions']:,}")
    lines += ["", "The core-row sequence equality is strong evidence that the four files share the same ordered observations and differ only in the final prediction field. It is still interpreted together with the schema and per-file audit, not from filenames alone.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/raw/anya/official-download"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/metrics/anya_stream_audit.json"))
    parser.add_argument("--output-report", type=Path, default=Path("reports/anya_file_level_audit.md"))
    parser.add_argument("--chunk-size", type=int, default=20_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--files", nargs="*", default=list(EXPECTED_FILES))
    args = parser.parse_args()
    payload = run_audit(args.root, args.files, args.chunk_size, args.max_rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    _write_report(payload, args.output_report)
    print(json.dumps({"files": list(payload["files"]), "comparison": payload["comparison"], "output_json": str(args.output_json), "output_report": str(args.output_report)}, indent=2))


if __name__ == "__main__":
    main()
