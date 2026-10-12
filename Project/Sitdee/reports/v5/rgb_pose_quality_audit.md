# V5 RGB pose-quality audit

## Source fact

The V5 contract uses MediaPipe Pose World Landmarks and five core landmarks:
`nose`, `left_shoulder`, `right_shoulder`, `left_hip`, and `right_hip`.

## Verified local observation

`datasets/processed/rgb_native_v5/manifest.csv` contains 452 sampled media rows:
417 from six DriveNet videos and 35 from the public MMD-MSD image subset. All
452 rows have `pose_detected=1` and are feature-eligible. The recorded
`pose_quality` is the mean visibility of the five core landmarks:

| Scope | Rows | Min | Mean | Median | Max |
|---|---:|---:|---:|---:|---:|
| All V5 rows | 452 | 0.994645 | 0.999658 | 0.999696 | 0.999989 |
| DriveNet v6 | 417 | 0.998923 | 0.999653 | 0.999686 | 0.999940 |
| MMD-MSD subset | 35 | 0.994645 | 0.999717 | 0.999916 | 0.999989 |

## Experiment result

A row is feature-eligible only when all five core landmarks meet visibility
`>= 0.5` and World Landmark coordinates are finite. The current sampled rows
all pass this gate. This is a pose-quality/extraction result, not a
classifier-accuracy result. Twenty-five MMD-MSD rows remain in the manifest
without a project label because their source conditions are not directionally
compatible or are ambiguous; they were not silently removed or imputed.

## Engineering decision

The available V5 bottleneck is verified label coverage and independent source
diversity, not MediaPipe core-landmark detection. No webcam source was opened.
