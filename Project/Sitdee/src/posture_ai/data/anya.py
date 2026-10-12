"""Safe, explicit helpers for the downloaded ANYA CSV representation.

The official release is large and must be consumed in bounded chunks.  These
helpers intentionally distinguish identifiers, cluster IDs, model prediction
outputs, and a verified ground-truth target; a binary column named
``Predicted_Label`` is not silently promoted to a training target.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd


_RAW_COORDINATE = re.compile(r"^(x|y|z)(\d+)$", re.IGNORECASE)
_RAW_VISIBILITY = re.compile(r"^vis(\d+)$", re.IGNORECASE)
_NORMALIZED_COORDINATE = re.compile(r"^(x|y|z)(\d+)normalized$", re.IGNORECASE)


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file incrementally so verification does not require full RAM."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_anya_csv(path: str | Path, *, chunksize: int = 20_000, usecols: Sequence[str] | None = None) -> Iterator[pd.DataFrame]:
    """Yield an ANYA CSV sequentially with explicit UTF-8/comma parsing."""

    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    return pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=list(usecols) if usecols is not None else None,
        sep=",",
        encoding="utf-8",
        low_memory=False,
    )


def detect_anya_schema(columns: Sequence[str]) -> dict:
    """Classify the measured ANYA columns without assigning anatomy by guess."""

    names = [str(column) for column in columns]
    raw_by_axis: dict[str, list[tuple[int, str]]] = {axis: [] for axis in ("x", "y", "z")}
    visibility: list[tuple[int, str]] = []
    normalized_by_axis: dict[str, list[tuple[int, str]]] = {axis: [] for axis in ("x", "y", "z")}
    for name in names:
        raw = _RAW_COORDINATE.match(name)
        if raw:
            raw_by_axis[raw.group(1).lower()].append((int(raw.group(2)), name))
            continue
        visible = _RAW_VISIBILITY.match(name)
        if visible:
            visibility.append((int(visible.group(1)), name))
            continue
        normalized = _NORMALIZED_COORDINATE.match(name)
        if normalized:
            normalized_by_axis[normalized.group(1).lower()].append((int(normalized.group(2)), name))

    for values in (*raw_by_axis.values(), visibility, *normalized_by_axis.values()):
        values.sort()
    raw_indices = sorted({index for values in raw_by_axis.values() for index, _ in values})
    normalized_indices = sorted({index for values in normalized_by_axis.values() for index, _ in values})
    raw_complete = all([sorted(index for index, _ in raw_by_axis[axis]) == raw_indices for axis in ("x", "y", "z")])
    normalized_complete = all([sorted(index for index, _ in normalized_by_axis[axis]) == normalized_indices for axis in ("x", "y", "z")])
    metadata = [name for name in names if name not in {item[1] for values in raw_by_axis.values() for item in values} and name not in {item[1] for item in visibility} and name not in {item[1] for values in normalized_by_axis.values() for item in values}]
    return {
        "column_count": len(names),
        "columns": names,
        "raw_coordinate_indices": raw_indices,
        "raw_coordinate_columns": {axis: [name for _, name in raw_by_axis[axis]] for axis in ("x", "y", "z")},
        "raw_coordinate_complete": raw_complete,
        "visibility_indices": [index for index, _ in visibility],
        "visibility_columns": [name for _, name in visibility],
        "normalized_coordinate_indices": normalized_indices,
        "normalized_coordinate_columns": {axis: [name for _, name in normalized_by_axis[axis]] for axis in ("x", "y", "z")},
        "normalized_coordinate_complete": normalized_complete,
        "metadata_columns": metadata,
        "raw_landmark_count": len(raw_indices) if raw_complete else None,
        "normalized_landmark_count": len(normalized_indices) if normalized_complete else None,
    }


def detect_anya_target(columns: Sequence[str]) -> dict:
    """Return a conservative target audit; never treat predictions as truth."""

    names = [str(column) for column in columns]
    lower = {name: name.lower() for name in names}
    prediction_columns = [name for name in names if "predicted" in lower[name] or lower[name].startswith("prediction")]
    cluster_columns = [name for name in names if "cluster" in lower[name]]
    explicit_ground_truth = [
        name for name in names
        if any(token in lower[name] for token in ("label", "target", "class"))
        and name not in prediction_columns
        and name not in cluster_columns
    ]
    return {
        "verified_ground_truth_columns": explicit_ground_truth,
        "prediction_output_columns": prediction_columns,
        "cluster_id_columns": cluster_columns,
        "status": "verified_ground_truth_available" if explicit_ground_truth else "no_verified_ground_truth_column",
        "decision": "train_allowed" if explicit_ground_truth else "training_blocked_until_ground_truth_is_supplied_or_verified",
    }


def detect_anya_grouping(columns: Sequence[str]) -> dict:
    """Classify grouping evidence without claiming filename means participant."""

    names = [str(column) for column in columns]
    lower = {name: name.lower() for name in names}
    explicit_participant = [name for name in names if any(token in lower[name] for token in ("subject", "participant", "person", "user"))]
    explicit_session = [name for name in names if "session" in lower[name]]
    explicit_video = [name for name in names if any(token in lower[name] for token in ("video", "clip", "recording", "sequence"))]
    filename = [name for name in names if lower[name] == "filename"]
    return {
        "participant_columns": explicit_participant,
        "session_columns": explicit_session,
        "video_or_recording_columns": explicit_video,
        "filename_columns": filename,
        "recommended_group_column": (explicit_participant or explicit_session or explicit_video or filename or [None])[0],
        "participant_independence_proven": bool(explicit_participant),
    }
