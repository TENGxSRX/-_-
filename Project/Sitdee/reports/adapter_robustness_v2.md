# Adapter robustness V2 study

## Executive result

The old 126-feature V1 adapter rejected **0/900** external frames because it requires all 33 coordinate slots to have visibility ≥ 0.5. MediaPipe itself detected a pose in **900/900** frames. The dominant failures are low-visibility knees, ankles, heels, feet, and some elbows/fingers caused by seated side-view occlusion.

V2 was scientifically feasible as a separate experimental track because the measured MultiPosture CSV contains the five core column groups needed for head/shoulder/trunk/hip posture features. V2 uses `nose, left_shoulder, right_shoulder, left_hip, right_hip` only and does not overwrite V1.

## 1–6. Diagnosis and semantic verification

1. **Why old V1 failed:** full-body visibility gate, not pose detection failure; the exact ranked visibility table is in `reports/landmark_visibility_analysis.md`.
2. **Necessary office-posture landmarks:** nose, both shoulders, both hips. Ears/elbows are optional; knees/ankles/feet are low-value for the stated goal.
3. **Was old rule too strict?** Yes for an upper-body office posture track; no for the existing V1 full 126-feature contract. Policy results are below.
4. **Policy acceptance:**

| Policy | Accepted | Rejected | Acceptance | Average quality |
|---|---:|---:|---:|---:|
| `strict` | 0 | 900 | 0.0% | 0.674 |
| `core-only` | 900 | 0 | 100.0% | 1.000 |
| `tiered` | 900 | 0 | 100.0% | 1.000 |
| `quality` | 900 | 0 | 100.0% | 1.000 |

5. **MultiPosture vs MediaPipe semantics:** exact 33-name/99-column identity is present in the local CSV, but the source prose says 11 joints and does not fully define projection/scale/z. Therefore name mapping is `EXACT_MATCH` at the column-name level while real-image semantic compatibility remains **NOT VALIDATED**. See `reports/multiposture_landmark_mapping_verified.md`.
6. **Ten-feature domain shift:** the leading causes are coordinate convention/projection, scale/normalization, z semantics, side-view foreshortening, and low-visibility limb coordinates—not a reason to clip values. See `reports/domain_shift_root_cause.md`.

## 7–9. V2 training and comparison

- V2 selected model/config: `HistGradientBoosting` + `hybrid` (30 features)
- V2 grouped-CV macro F1: `0.9925 ± 0.0087`
- V2 validation macro F1: `0.9910`
- V2 final subject-held-out macro F1: `1.0000`

| Attribute | V1 | V2 |
|---|---|---|
| Required landmarks | 33 | 5 core |
| Feature count | 126 | 30 |
| Grouped CV macro F1 | 0.9891 ± 0.0060 | 0.9925 ± 0.0087 |
| Holdout macro F1 | 1.0000 | 1.0000 |
| External adapter acceptance | 0/900 (0.0%) | 900/900 (100.0%) |
| Inference latency | Not measured in the prior V1 offline performance record | 23.0 ms/accepted frame (adapter + classifier; pose extraction excluded) |
| Domain-shift severity (mechanical train min/max diagnostic) | 10/126 features outside range for >50% of external rows | 6/30 V2 features outside range for >50% of external rows; semantics unresolved |
| Expected occlusion robustness | Poor for seated side view | Better for core torso/head occlusion pattern |
| Real-image semantic compatibility | Not validated | Not validated; operational gate improved only if core passes |

The goal is robustness, not a higher benchmark score. The V2 holdout is still drawn from the same 13-subject MultiPosture source and is not real-camera generalization.

## 10–11. DriveNet V2 test and temporal behavior

Valid mapped-frame metrics: accuracy 0.0000, balanced accuracy 0.0000, macro F1 0.0000; interpret cautiously because the external set has one clip per mapped source class and is not a subject-independent test.

| Clip | Source label | Accepted | Acceptance | Predictions | Raw switches | Smoothed switches | Pose-lost/rejected |
|---|---|---:|---:|---|---:|---:|---:|
| `LB_2.mp4` | `TLB` | 300 | 100.0% | {'TLR': 222, 'TUP': 78} | 36 | 12 | 0.0% |
| `MB_1.mp4` | `UNLABELED` | 300 | 100.0% | {'TLF': 48, 'TLR': 79, 'TUP': 173} | 54 | 21 | 0.0% |
| `SS_1.mp4` | `TUP` | 300 | 100.0% | {'TLL': 300} | 0 | 0 | 0.0% |

Temporal smoothing results are descriptive only. It reduces or preserves switching according to the per-clip values; it does not repair invalid landmarks. Ambiguous DriveNet source classes were not mapped to project labels.

## 12–14. Backend decision and blockers

**YOLO gate:** KEEP MEDIAPIPE as the primary backend; YOLO Pose benchmarking is not justified by this study.

Current real-image compatibility statement: **MODEL TRAINING READY; UPPER-BODY REAL-IMAGE ADAPTER OPERATIONALLY IMPROVED / SEMANTICALLY NOT VALIDATED.** The V2 gate is more appropriate for seated occlusion, but its coordinate convention still needs paired validation.

Remaining blockers before own real sitting videos:

- verify the source-coordinate convention with a paired RGB/landmark example or controlled recording;
- collect the planned five office-view posture videos when hardware is available;
- evaluate across people and camera placements, not only one external driver clip per class;
- keep ANYA training blocked until a verified ground-truth mapping is obtained.

No raw dataset was modified, V1 was not overwritten, no webcam source 0 was opened, and no medical claim is made.
