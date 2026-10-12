# RGB-native V4 model comparison

V4 is an experimental RGB-native pilot. It uses only DriveNet source labels with a defensible target mapping (TUP and TLB). It is not a six-class replacement for MultiPosture.

| Rank | Feature family | Model | Features | Grouped CV macro F1 | CV std | Participant-3 holdout macro F1 | Holdout balanced accuracy |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | normalized_world | LogisticRegression | 15 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| 2 | world | ExtraTrees | 15 | 0.9992 | 0.0008 | 1.0000 | 1.0000 |
| 3 | world | HistGradientBoosting | 15 | 0.9992 | 0.0008 | 1.0000 | 1.0000 |
| 4 | world | LogisticRegression | 15 | 0.9992 | 0.0008 | 1.0000 | 1.0000 |
| 5 | hybrid | LogisticRegression | 30 | 0.9975 | 0.0025 | 1.0000 | 1.0000 |
| 6 | world | GradientBoosting | 15 | 0.9967 | 0.0033 | 1.0000 | 1.0000 |
| 7 | normalized_world | GradientBoosting | 15 | 0.9967 | 0.0033 | 1.0000 | 1.0000 |
| 8 | world | RandomForest | 15 | 0.9967 | 0.0033 | 1.0000 | 1.0000 |
| 9 | normalized_world | RandomForest | 15 | 0.9967 | 0.0033 | 1.0000 | 1.0000 |
| 10 | geometry | GradientBoosting | 15 | 0.9942 | 0.0058 | 1.0000 | 1.0000 |
| 11 | geometry | RandomForest | 15 | 0.9942 | 0.0058 | 1.0000 | 1.0000 |
| 12 | hybrid | GradientBoosting | 30 | 0.9942 | 0.0058 | 1.0000 | 1.0000 |

Selected by grouped-CV macro F1 (fixed tie-breaks): `normalized_world` + `LogisticRegression`.

The participant-3 holdout is not used for model selection. It is one held-out participant, so uncertainty remains high and these scores are not a claim of office-camera generalization.

## Frozen DriveNet re-test

Status: `SAME_SOURCE_OVERLAP_DIAGNOSTIC`. This diagnostic is not independent generalization because the clips are from the same DriveNet source as training.
- `LB_2.mp4`: accepted 300/300 frames; predictions `{'TLB': 300}`; source mapping `TLB`.
- `MB_1.mp4`: accepted 300/300 frames; predictions `{'TLB': 300}`; source mapping `UNLABELED`.
- `SS_1.mp4`: accepted 300/300 frames; predictions `{'TLB': 2, 'TUP': 298}`; source mapping `TUP`.

## Cross-dataset status

No independent cross-dataset RGB score is reported. MultiPosture has no raw RGB release, CCTV Exam Monitor has no joinable per-image labels in the downloaded archive, SitStraightNet has no local label sidecar, and MVSP/LGCSPNet was not locally verified. The DriveNet re-test above is a same-source overlap diagnostic. A true cross-dataset test requires an independently labeled RGB source whose labels and pose representation can be audited.
