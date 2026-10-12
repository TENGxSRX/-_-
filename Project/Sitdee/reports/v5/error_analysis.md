# V5 error analysis

## Experiment result

**NOT RUN.** V5 model fitting was blocked before training because TLF, TLL,
and TLR lack eligible source-backed RGB labels. There are no V5 classifier
false positives, false negatives, confidence scores, or worst-group metrics to
analyze, and none are fabricated.

## Available audit evidence

Pose extraction passed for all 452 sampled rows. The current actionable issue
is label/source coverage: 244 TUP rows, 183 TLB rows, and 25 unlabeled audit
rows. See `datasets/processed/rgb_native_v5/manifest.csv` and
`reports/v5/rgb_pose_quality_audit.md`.
