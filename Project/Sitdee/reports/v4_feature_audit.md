# V4 feature audit

The frozen V4 training cache contains the six DriveNet videos sampled at 300 frames/video. This audit compares participants 1/2 (the declared training participants) with participant 3 (the declared holdout) without retraining.

## Participant/video confound inventory

- Participant-label counts: `{'1': {'TLB': 300, 'TUP': 300}, '2': {'TLB': 300, 'TUP': 300}, '3': {'TLB': 300, 'TUP': 300}}`
- Video-label counts: `{'LB_1': {'TLB': 300, 'TUP': 0}, 'LB_2': {'TLB': 300, 'TUP': 0}, 'LB_3': {'TLB': 300, 'TUP': 0}, 'SS_1': {'TLB': 0, 'TUP': 300}, 'SS_2': {'TLB': 0, 'TUP': 300}, 'SS_3': {'TLB': 0, 'TUP': 300}}`
- Every participant has both TUP and TLB source labels in the pilot, so participant identity is not perfectly identical to the class label. Nevertheless, frame-level observations are temporally correlated and the small number of videos limits the strength of this conclusion.

## Shift table

| Feature | Holdout abs SMD | Wasserstein / train std | Holdout outside train range | Train mean ± std | Holdout mean ± std |
|---|---:|---:|---:|---:|---:|
| `nose_y_world_hip_centered_scaled` | 1.212 | 1.212 | 14.3% | -1.1307 ± 0.0796 | -1.0343 ± 0.1520 |
| `left_hip_y_world_hip_centered_scaled` | 0.932 | 0.939 | 16.8% | 0.0785 ± 0.0124 | 0.0669 ± 0.0229 |
| `right_hip_y_world_hip_centered_scaled` | 0.932 | 0.939 | 16.8% | -0.0785 ± 0.0124 | -0.0669 ± 0.0229 |
| `right_shoulder_y_world_hip_centered_scaled` | 0.925 | 0.925 | 38.2% | -1.0142 ± 0.1078 | -0.9145 ± 0.1721 |
| `left_hip_z_world_hip_centered_scaled` | 0.657 | 0.892 | 13.0% | -0.1531 ± 0.0109 | -0.1459 ± 0.0194 |
| `right_hip_z_world_hip_centered_scaled` | 0.657 | 0.892 | 13.0% | 0.1531 ± 0.0109 | 0.1459 ± 0.0194 |
| `left_shoulder_y_world_hip_centered_scaled` | 0.522 | 0.913 | 56.2% | -0.7573 ± 0.1041 | -0.7030 ± 0.1974 |
| `nose_x_world_hip_centered_scaled` | 0.477 | 0.487 | 0.2% | -0.1891 ± 0.4205 | 0.0114 ± 0.4231 |
| `right_shoulder_z_world_hip_centered_scaled` | 0.466 | 0.467 | 1.3% | 0.3112 ± 0.0851 | 0.2716 ± 0.0987 |
| `right_shoulder_x_world_hip_centered_scaled` | 0.460 | 0.462 | 1.0% | 0.2197 ± 0.3673 | 0.3885 ± 0.3655 |
| `left_shoulder_x_world_hip_centered_scaled` | 0.395 | 0.399 | 35.3% | 0.2920 ± 0.3497 | 0.4303 ± 0.3543 |
| `nose_z_world_hip_centered_scaled` | 0.285 | 0.351 | 1.0% | -0.3146 ± 0.1006 | -0.3433 ± 0.1202 |
| `left_shoulder_z_world_hip_centered_scaled` | 0.077 | 0.143 | 0.2% | -0.4342 ± 0.1084 | -0.4425 ± 0.1131 |
| `left_hip_x_world_hip_centered_scaled` | 0.009 | 0.095 | 0.0% | 0.0173 ± 0.0254 | 0.0170 ± 0.0260 |
| `right_hip_x_world_hip_centered_scaled` | 0.009 | 0.095 | 0.0% | -0.0173 ± 0.0254 | -0.0170 ± 0.0260 |

## Top 15 frozen LogisticRegression coefficient magnitudes

| Rank | Feature | Max absolute coefficient | Coefficients by class |
|---:|---|---:|---|
| 1 | `left_shoulder_x_world_hip_centered_scaled` | 1.72654 | `[1.72654]` |
| 2 | `right_shoulder_x_world_hip_centered_scaled` | 1.58877 | `[1.58877]` |
| 3 | `nose_x_world_hip_centered_scaled` | 1.56265 | `[1.56265]` |
| 4 | `left_shoulder_y_world_hip_centered_scaled` | 1.40064 | `[1.40064]` |
| 5 | `right_shoulder_y_world_hip_centered_scaled` | 1.02734 | `[1.02734]` |
| 6 | `nose_y_world_hip_centered_scaled` | 0.75765 | `[0.75765]` |
| 7 | `left_hip_z_world_hip_centered_scaled` | 0.66603 | `[0.66603]` |
| 8 | `right_hip_z_world_hip_centered_scaled` | 0.66603 | `[-0.66603]` |
| 9 | `left_hip_y_world_hip_centered_scaled` | 0.47946 | `[0.47946]` |
| 10 | `right_hip_y_world_hip_centered_scaled` | 0.47946 | `[-0.47946]` |
| 11 | `left_shoulder_z_world_hip_centered_scaled` | 0.32294 | `[0.32294]` |
| 12 | `right_hip_x_world_hip_centered_scaled` | 0.30080 | `[0.3008]` |
| 13 | `left_hip_x_world_hip_centered_scaled` | 0.30080 | `[-0.3008]` |
| 14 | `nose_z_world_hip_centered_scaled` | 0.09694 | `[-0.09694]` |
| 15 | `right_shoulder_z_world_hip_centered_scaled` | 0.09340 | `[0.0934]` |

## Interpretation

- This is descriptive domain-shift evidence, not a reason to clip or rescale the model input.
- Coefficient magnitude indicates what the frozen linear V4 model uses in its own feature scale; it is not a causal importance measure.
- The pilot has only three participants and six videos. A new RGB dataset with held-out people and camera views remains the strongest next experiment.
