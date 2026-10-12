# External RGB domain-shift analysis

## Scope

This is a diagnostic comparison of the 126-feature vector produced from MediaPipe landmarks on the external RGB videos versus the 126-feature training representation used by MultiPosture. The comparison ignores visibility only to inspect feature distributions; those rows were **not** sent to the classifier for external accuracy because the safe adapter rejected the low-visibility landmarks.

## Schema identity versus semantic compatibility

- Feature count: `126`
- Feature-name/order identity with the implemented adapter: `True`
- External diagnostic rows: `900`
- Semantic compatibility validated: **False**

The external representation uses MediaPipe image-normalized x/y/z followed by hip-centering. The stored MultiPosture training values are documented as hip-centered source coordinates. Equal feature names and count do not prove identical coordinate scale, projection, camera geometry, visibility semantics, or population distribution.

## Range comparison

`outside_fraction` is the fraction of external diagnostic rows outside the training min/max for a feature. `robust_distance` is the median absolute distance from the training median measured in training IQR units.

| feature | outside_fraction | robust_distance |
|---|---:|---:|
| `left_hip_z` | 1.000 | 14.439 |
| `right_hip_z` | 1.000 | 14.315 |
| `left_hip_x` | 1.000 | 7.546 |
| `right_hip_x` | 1.000 | 7.290 |
| `left_knee_x` | 1.000 | 2.358 |
| `head_to_shoulder_z` | 0.963 | 5.986 |
| `left_shoulder_to_hip` | 0.827 | 5.909 |
| `left_ankle_x` | 0.699 | 2.146 |
| `right_elbow_z` | 0.661 | 2.207 |
| `right_knee_z` | 0.502 | 3.751 |

Features with more than half of external rows outside the training min/max: `10`.

## Decision

**MODEL TRAINING READY; REAL-IMAGE INFERENCE ADAPTER NOT YET VALIDATED.** The exact 126-column schema is implemented and mechanically reproducible, but the external camera/view and visibility behavior do not establish domain compatibility. Do not claim RGB generalization or use this external run as classifier accuracy.
