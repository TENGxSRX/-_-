# V2 feature importance audit

Permutation importance was measured on grouped validation subjects: `4, 9`. The frozen V2 artifact was loaded; no V2 retraining occurred.

| Rank | Feature | Permutation importance mean | Std | Known shifted feature |
|---:|---|---:|---:|---|
| 1 | `trunk_vector_z` | 0.228513 | 0.013251 | False |
| 2 | `trunk_vector_y` | 0.046907 | 0.005867 | False |
| 3 | `left_shoulder_x` | 0.014764 | 0.001864 | False |
| 4 | `nose_x` | 0.000000 | 0.000000 | False |
| 5 | `nose_y` | 0.000000 | 0.000000 | False |
| 6 | `nose_z` | 0.000000 | 0.000000 | False |
| 7 | `left_shoulder_y` | 0.000000 | 0.000000 | False |
| 8 | `left_shoulder_z` | 0.000000 | 0.000000 | False |
| 9 | `right_shoulder_x` | 0.000000 | 0.000000 | False |
| 10 | `right_shoulder_y` | 0.000000 | 0.000000 | False |
| 11 | `right_shoulder_z` | 0.000000 | 0.000000 | False |
| 12 | `left_hip_x` | 0.000000 | 0.000000 | True |
| 13 | `left_hip_y` | 0.000000 | 0.000000 | False |
| 14 | `left_hip_z` | 0.000000 | 0.000000 | True |
| 15 | `right_hip_x` | 0.000000 | 0.000000 | True |

Known shifted features in top 15: `left_hip_x, left_hip_z, right_hip_x`.
Across all 30 features, shifted-feature permutation importance mean sum is `0.000000`. This is a model-behavior diagnostic, not proof of causal domain failure.
