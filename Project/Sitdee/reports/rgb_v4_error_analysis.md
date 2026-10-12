# RGB-native V4 error analysis

Selected track: `normalized_world` + `LogisticRegression`. Holdout participant: `3`.

```json
{
  "accuracy": 1.0,
  "balanced_accuracy": 1.0,
  "macro_precision": 1.0,
  "macro_recall": 1.0,
  "macro_f1": 1.0,
  "weighted_f1": 1.0,
  "per_class": {
    "TLB": {
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      "support": 300
    },
    "TUP": {
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0,
      "support": 300
    }
  },
  "confusion_matrix": [
    [
      300,
      0
    ],
    [
      0,
      300
    ]
  ],
  "labels": [
    "TLB",
    "TUP"
  ],
  "roc_auc": 1.0,
  "pr_auc": 1.0,
  "brier_score": 1.5217909565779867e-05,
  "sensitivity": 1.0,
  "specificity": 1.0,
  "false_positive_rate": 0.0,
  "false_negative_rate": 0.0
}
```

The source videos are controlled driver-posture recordings. Remaining errors, if any, should be analyzed as source-domain errors; they do not establish clinical or office-environment performance.
