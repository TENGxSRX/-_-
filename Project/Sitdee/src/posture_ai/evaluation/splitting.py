"""Deterministic subject-grouped splits and cross-validation."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def grouped_holdout_split(groups, *, test_groups: int = 2, validation_groups: int = 2, random_state: int = 42) -> dict[str, np.ndarray]:
    """Split whole participants into train, validation, and final-test sets."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) <= test_groups + validation_groups:
        raise ValueError("Not enough participant groups for the requested holdout")
    first = GroupShuffleSplit(n_splits=1, test_size=test_groups, random_state=random_state)
    train_val_idx, test_idx = next(first.split(np.zeros(len(groups)), groups=groups))
    remaining_groups = groups[train_val_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=validation_groups, random_state=random_state + 1)
    train_rel, val_rel = next(second.split(np.zeros(len(train_val_idx)), groups=remaining_groups))
    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]
    result = {"train": np.sort(train_idx), "validation": np.sort(val_idx), "test": np.sort(test_idx)}
    assert set(groups[result["train"]]).isdisjoint(set(groups[result["validation"]]))
    assert set(groups[result["train"]]).isdisjoint(set(groups[result["test"]]))
    assert set(groups[result["validation"]]).isdisjoint(set(groups[result["test"]]))
    return result


def grouped_cv(groups, n_splits: int = 5):
    """Return GroupKFold splits, never splitting a participant across folds."""
    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("At least two participant groups are required")
    return GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups)

