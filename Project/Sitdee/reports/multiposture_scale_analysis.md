# MultiPosture scale analysis

Distances were computed directly from the measured local CSV in its stored coordinate units: shoulder width, hip width, and shoulder-midpoint to hip-midpoint torso length.

## Overall distributions

| Distance | Mean | Std | Median | P10 | P90 |
|---|---:|---:|---:|---:|---:|
| `shoulder_width` | 0.33010 | 0.01979 | 0.33264 | 0.30231 | 0.35302 |
| `hip_width` | 0.21137 | 0.01224 | 0.21263 | 0.19633 | 0.22514 |
| `torso_length` | 0.42654 | 0.02624 | 0.42862 | 0.39234 | 0.45603 |


Subject-level coefficient of variation of subject means:

- `shoulder_width`: 3.08%
- `hip_width`: 2.73%
- `torso_length`: 3.02%


## By subject

| Subject | Rows | Shoulder width mean | Shoulder width CV | Hip width mean | Hip width CV | Torso length mean | Torso length CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1` | 540 | 0.32178 | 4.86% | 0.21158 | 4.41% | 0.43784 | 6.04% |
| `2` | 450 | 0.34114 | 5.04% | 0.21310 | 5.82% | 0.43829 | 3.64% |
| `3` | 480 | 0.33382 | 3.61% | 0.21287 | 4.36% | 0.44111 | 4.47% |
| `4` | 480 | 0.32867 | 4.89% | 0.21483 | 4.57% | 0.43045 | 3.44% |
| `5` | 444 | 0.34695 | 5.05% | 0.22266 | 5.47% | 0.43767 | 6.22% |
| `6` | 480 | 0.33307 | 6.36% | 0.21008 | 4.45% | 0.41975 | 5.80% |
| `7` | 300 | 0.34095 | 3.36% | 0.21354 | 6.75% | 0.42581 | 3.03% |
| `8` | 270 | 0.32942 | 5.81% | 0.20464 | 6.82% | 0.40418 | 4.85% |
| `9` | 270 | 0.30948 | 6.97% | 0.20589 | 4.80% | 0.41892 | 2.95% |
| `10` | 270 | 0.31616 | 7.32% | 0.21401 | 5.61% | 0.42844 | 6.88% |
| `11` | 270 | 0.32563 | 3.45% | 0.19821 | 4.49% | 0.40405 | 7.95% |
| `12` | 270 | 0.32844 | 3.75% | 0.20810 | 5.97% | 0.41617 | 4.16% |
| `13` | 270 | 0.32057 | 6.57% | 0.20652 | 3.96% | 0.40703 | 9.16% |


## By posture

| Label | Rows | Shoulder width mean | Hip width mean | Torso length mean | Torso length P10 | Torso length P90 |
|---|---:|---:|---:|---:|---:|---:|
| `TLB` | 442 | 0.32053 | 0.20624 | 0.42250 | 0.39361 | 0.44918 |
| `TLF` | 1897 | 0.33616 | 0.21806 | 0.42827 | 0.39007 | 0.45601 |
| `TLL` | 420 | 0.29887 | 0.20853 | 0.44287 | 0.41402 | 0.47752 |
| `TLR` | 420 | 0.33211 | 0.19457 | 0.43575 | 0.41758 | 0.45073 |
| `TUP` | 1615 | 0.33320 | 0.21001 | 0.41897 | 0.38658 | 0.45176 |


## Decision

The distances vary within subjects and across subjects; they are not a single fixed constant. The source explicitly documents world meters and a StandardScaler for model input, but no geometric torso/shoulder scaling formula is documented. Therefore V3 does not invent post-centering geometric scaling. A model-side scaler can be a separate controlled experiment, not a silent adapter transform.
