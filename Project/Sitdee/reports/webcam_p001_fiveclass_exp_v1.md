# Webcam P001 Five-Class Experimental Model V1

สถานะ: **EXPERIMENTAL_ONLY — ไม่ใช่ V5 production และยังไม่ได้ live-validate**

## Data and split

- Accepted Human GT: `257` records, `30` segments, participant `['P001']`
- Classes: `{'TUP': 54, 'TLF': 44, 'TLB': 54, 'TLL': 51, 'TLR': 54}`
- Feature schema: `upper_body_posture_v2:NO_HIP`
- Feature parity: `RUNTIME_FEATURE_PARITY_CONFIRMED`, max absolute difference `0`
- Development session: `SESSION_20260919T040053Z_P001`; test session: `SESSION_20260919T035749Z_P001`
- Development: `212` rows / `25` segments
- Test: `45` rows / `5` segments
- All validation folds group by segment; no segment appears in both train and validation.
- Test session was not used for model or UNKNOWN-threshold selection.

## Model selection

- Selected model: **HistGradientBoosting**
- Selected UNKNOWN threshold: `0.5`
- Selection used development-session GroupKFold out-of-fold results only.

| Model | Development OOF macro-F1 | Development OOF balanced accuracy | Threshold | Coverage |
|---|---:|---:|---:|---:|
| LogisticRegression | 0.8341 | 0.8410 | 0.50 | 91.0% |
| RBF_SVM | 0.8329 | 0.8333 | 0.50 | 89.6% |
| HistGradientBoosting | 0.9040 | 0.9076 | 0.50 | 98.1% |

## Untouched session holdout

- Raw classifier metrics: `{"accuracy": 0.8888888888888888, "balanced_accuracy": 0.888888888888889, "macro_f1": 0.890219421101774, "weighted_f1": 0.890219421101774, "per_class": {"TUP": {"precision": 0.75, "recall": 1.0, "f1": 0.8571428571428571, "support": 9}, "TLF": {"precision": 1.0, "recall": 0.8888888888888888, "f1": 0.9411764705882353, "support": 9}, "TLB": {"precision": 0.7777777777777778, "recall": 0.7777777777777778, "f1": 0.7777777777777778, "support": 9}, "TLL": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 9}, "TLR": {"precision": 1.0, "recall": 0.7777777777777778, "f1": 0.875, "support": 9}}, "confusion_matrix": [[9, 0, 0, 0, 0], [1, 8, 0, 0, 0], [2, 0, 7, 0, 0], [0, 0, 0, 9, 0], [0, 0, 2, 0, 7]], "labels": ["TUP", "TLF", "TLB", "TLL", "TLR"]}`
- With UNKNOWN refusal: `{"threshold": 0.5, "coverage": 1.0, "unknown_count": 0, "unknown_percentage": 0.0, "end_to_end_known_class_metrics": {"accuracy": 0.8888888888888888, "balanced_accuracy": 0.888888888888889, "macro_f1": 0.890219421101774, "weighted_f1": 0.890219421101774, "per_class": {"TUP": {"precision": 0.75, "recall": 1.0, "f1": 0.8571428571428571, "support": 9}, "TLF": {"precision": 1.0, "recall": 0.8888888888888888, "f1": 0.9411764705882353, "support": 9}, "TLB": {"precision": 0.7777777777777778, "recall": 0.7777777777777778, "f1": 0.7777777777777778, "support": 9}, "TLL": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 9}, "TLR": {"precision": 1.0, "recall": 0.7777777777777778, "f1": 0.875, "support": 9}}, "confusion_matrix": [[9, 0, 0, 0, 0], [1, 8, 0, 0, 0], [2, 0, 7, 0, 0], [0, 0, 0, 9, 0], [0, 0, 2, 0, 7]], "labels": ["TUP", "TLF", "TLB", "TLL", "TLR"]}, "accepted_only_metrics": {"accuracy": 0.8888888888888888, "balanced_accuracy": 0.888888888888889, "macro_f1": 0.890219421101774, "weighted_f1": 0.890219421101774, "per_class": {"TUP": {"precision": 0.75, "recall": 1.0, "f1": 0.8571428571428571, "support": 9}, "TLF": {"precision": 1.0, "recall": 0.8888888888888888, "f1": 0.9411764705882353, "support": 9}, "TLB": {"precision": 0.7777777777777778, "recall": 0.7777777777777778, "f1": 0.7777777777777778, "support": 9}, "TLL": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 9}, "TLR": {"precision": 1.0, "recall": 0.7777777777777778, "f1": 0.875, "support": 9}}, "confusion_matrix": [[9, 0, 0, 0, 0], [1, 8, 0, 0, 0], [2, 0, 7, 0, 0], [0, 0, 0, 9, 0], [0, 0, 2, 0, 7]], "labels": ["TUP", "TLF", "TLB", "TLL", "TLR"]}}`
- This is same-participant, same-camera-session evidence only; it is not participant-independent generalization.
- Confidence intervals are not reported as reliable because the holdout contains only five segments from P001.

## Current UNKNOWN/OOD diagnosis

- The current live V2 artifact was trained on a different source-domain distribution.
- Webcam Human GT features use the exact same six-feature runtime schema, so the new experiment has feature parity.
- The old artifact's training ranges exclude most webcam values, especially shoulder yaw and nose/shoulder z; this explains the previous `POSSIBLE_OOD_EXPERIMENTAL` refusals.
- This experiment does not weaken the old safety gate and does not replace the live demo model.

## Limitations

- One participant (`P001`) and two non-empty sessions.
- No participant-independent test.
- Left/right convention is consistent within this camera/person collection but not externally validated against a mirrored camera.
- No live webcam test was run in this milestone.
- Human GT is a controlled calibration dataset, not broad office CCTV evidence.
