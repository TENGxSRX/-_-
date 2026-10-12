# Coordinate adapter V3 study

## Decision

**REAL RGB COMPATIBILITY STATUS: PARTIALLY_VALIDATED**

The source-level convention is now strongly supported as MediaPipe World Landmarks in meters with hip-midpoint origin. Hip-midpoint behavior is numerically stable in both the local CSV and MediaPipe world output. However, the released CSV's measured 99-column schema conflicts with the paper's stated reduced 33-input model matrix, the exact StandardScaler artifact is unavailable, and DriveNet geometry/predictions still do not provide semantic validation. Therefore this is not status A (`VALIDATED`).

## Answers

1. **Representation:** source paper: MediaPipe Pose Heavy World Landmarks, meters, hip-midpoint origin; local release: 99 named XYZ columns, with exact reduced-model serialization unresolved.
2. **Hip centering:** verified as a stable near-zero midpoint residual, not exact zero. The local and MediaPipe world midpoints are small residuals; image-normalized coordinates require translation.
3. **Scale:** no geometric torso/shoulder normalization was verified. Distances vary by subject/posture. Paper-level `StandardScaler` is model preprocessing, not evidence for geometric adapter scaling.
4. **Axes/sign:** official MediaPipe semantics are x right, y down in image coordinates, z smaller toward camera; world uses meters and hip origin. Dataset observation: TLL has positive trunk-x, TLR negative; TLF negative trunk-z, TLB positive. Mirror/z flips remain unsupported.
5. **Closer representation:** World Landmarks are source-supported and are the correct V3 candidate. Distribution scores are recorded without label-based selection.
6. **Old V2 failure:** image-normalized coordinates were fed to a model trained on a source described as world-meter coordinates; hip translation alone does not repair scale/projection/z semantics. The 99-versus-33 release mismatch adds uncertainty.
7. **SS→TLL:** not one-axis-proven; source-supported world control and feature trace are in `reports/ss_tup_to_tll_root_cause.md`. The mismatch remains consistent with release/model-input mismatch plus camera/domain and/or source-label semantic gap.
8. **Best-supported V3:** `adapter_v3_a_world_hip_centered`; frozen as a forensic recommendation only, not production-connected.
9. **Mechanical OOD before/after:** current image V2: `6/30`; source-supported world V2: `5/30`. V1 current image: `10/126`; world: `9/126`.
10. **Geometry alignment:** not aligned to the mapped `TUP` semantics. Under the world candidate, `LB_2` and `MB_1` are nearest to the local `TLF` centroid, while `SS_1` is nearest to `TLL` (standardized geometry distance `3.129` versus `12.072` for `TUP`). This is a label-independent diagnostic, not a relabeling decision.
11. **Ablation:** temporary grouped ablations are in `reports/coordinate_adapter_v3_ablations.md`; no final model was selected from DriveNet accuracy.
12. **Geometry-only baseline:** grouped CV macro-F1 `0.9623 ± 0.0353`, holdout macro-F1 `0.9756`.
13. **Frozen external validation:** all `900/900` sampled frames had pose/world output and the five-core V2 gate accepted `900/900`; under the frozen world candidate the distributions were `LB_2: TLR=71/TUP=229`, `MB_1: TLR=4/TUP=296`, and `SS_1: TUP=1/TLL=299`. These are diagnostic only because external semantic equivalence is invalid.
14. **Real RGB compatibility:** `PARTIALLY_VALIDATED`, not fully validated.
15. **Retraining:** not recommended yet. First resolve released-column/model-input and scaler provenance.
16. **Own RGB dataset:** necessary for final calibration/validation; collect labeled office-view videos across people/camera placements.
17. **YOLO:** still unnecessary; the blocker is coordinate semantics, not detection.

## Integrity

V1 and V2 artifacts were loaded read-only. No V1/V2 model was retrained or overwritten. Raw data were not changed. ANYA was not trained. Webcam source 0 was not opened. Host `DESKTOP-ALH6CUB`; Python `3.12.10`; run `2026-09-05T06:05:03.585451+00:00`.

## Key metrics

- DriveNet pose detection: 900/900 sampled frames.
- Current image V2 core acceptance: 900/900.
- V2 selected model: `HistGradientBoosting` + `hybrid`, 30 features.
- V2 grouped-CV macro-F1: `0.9925 ± 0.0087`; subject-held-out validation macro-F1 `0.9910`.
- World candidate distribution: V2 mean SMD `2.318`, mean Wasserstein/std `2.430`, row-any-feature OOD `100.0%`.
