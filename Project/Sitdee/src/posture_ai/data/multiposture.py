"""MultiPosture loading with strict source-schema checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.posture_ai.features.pose_features import expected_feature_columns


LABEL_COLUMNS = ["upperbody_label", "lowerbody_label"]
GROUP_COLUMN = "subject"


def load_multiposture(path: str | Path) -> pd.DataFrame:
    """Load the verified MultiPosture CSV without changing its values."""
    frame = pd.read_csv(path)
    expected = [GROUP_COLUMN, *LABEL_COLUMNS, *expected_feature_columns()]
    if list(frame.columns) != expected:
        raise ValueError(
            "MultiPosture schema mismatch. "
            f"Expected {len(expected)} exact columns but received {len(frame.columns)}."
        )
    if frame[GROUP_COLUMN].isna().any():
        raise ValueError("MultiPosture subject identifiers contain missing values")
    if frame[LABEL_COLUMNS].isna().any().any():
        raise ValueError("MultiPosture labels contain missing values")
    return frame


def validate_multiposture(frame: pd.DataFrame) -> dict:
    """Return machine-readable source facts and fail on unsafe values."""
    numeric = frame[expected_feature_columns()].to_numpy(dtype=np.float64)
    non_finite = int((~np.isfinite(numeric)).sum())
    if non_finite:
        raise ValueError(f"MultiPosture contains {non_finite} non-finite feature cells")
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "subjects": sorted(str(x) for x in frame[GROUP_COLUMN].unique()),
        "upper_labels": {str(k): int(v) for k, v in frame["upperbody_label"].value_counts().items()},
        "lower_labels": {str(k): int(v) for k, v in frame["lowerbody_label"].value_counts().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "non_finite_feature_cells": non_finite,
        "rows_per_subject": {str(k): int(v) for k, v in frame.groupby(GROUP_COLUMN).size().items()},
    }
