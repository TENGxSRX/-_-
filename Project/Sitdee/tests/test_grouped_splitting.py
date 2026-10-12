from __future__ import annotations

import numpy as np

from src.posture_ai.evaluation.splitting import grouped_holdout_split, grouped_cv


def test_grouped_holdout_has_no_subject_overlap():
    groups = np.repeat(np.arange(13), 4)
    split = grouped_holdout_split(groups, test_groups=2, validation_groups=2, random_state=42)
    train_groups = set(groups[split["train"]])
    validation_groups = set(groups[split["validation"]])
    test_groups = set(groups[split["test"]])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
    assert sorted(np.concatenate([split["train"], split["validation"], split["test"]]).tolist()) == list(range(len(groups)))


def test_group_kfold_has_no_group_overlap():
    groups = np.repeat(np.arange(13), 4)
    for train_idx, eval_idx in grouped_cv(groups, n_splits=5):
        assert set(groups[train_idx]).isdisjoint(set(groups[eval_idx]))

