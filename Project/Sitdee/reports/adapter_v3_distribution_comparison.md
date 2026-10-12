# Adapter V3 distribution comparison

Selection order is source semantics first, then distribution diagnostics. No DriveNet label was used to select a candidate.

| Candidate | Input representation | V1 within range | V1 OOD features >50% | V1 mean SMD | V1 mean Wasserstein/std | V2 within range | V2 OOD features >50% | V2 mean SMD | V2 mean Wasserstein/std |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `adapter_v3_a_world_hip_centered` | `world_landmarks` | 90.7% | 9/126 | 1.652 | 1.737 | 82.9% | 5/30 | 2.318 | 2.430 |
| `adapter_v3_b_world_hip_centered_zflip` | `world_landmarks` | 86.2% | 15/126 | 1.675 | 1.764 | 82.6% | 5/30 | 2.219 | 2.330 |
| `adapter_v3_c_image_hip_centered` | `normalized_image_landmarks` | 92.1% | 10/126 | 1.784 | 1.889 | 80.3% | 6/30 | 2.971 | 3.070 |
| `adapter_v3_d_image_hip_centered_zflip` | `normalized_image_landmarks` | 90.1% | 11/126 | 1.748 | 1.854 | 77.0% | 7/30 | 2.714 | 2.805 |
| `adapter_v3_e_world_hip_centered_mirror` | `world_landmarks` | 89.5% | 10/126 | 1.707 | 1.800 | 81.1% | 5/30 | 2.616 | 2.718 |

## Candidate decisions

- `adapter_v3_a_world_hip_centered` is the best-supported candidate because the paper explicitly specifies World Landmarks in meters with hip-midpoint origin.
- `adapter_v3_c_image_hip_centered` is the current V2 adapter baseline.
- `adapter_v3_b_world_hip_centered_zflip` has slightly lower V2 distribution distances in this sample, but z-flip is not source-supported and was not selected from that numerical advantage.
- z-flip and mirror candidates are controls, not production recommendations; they are not source-supported.
- Distribution compatibility does not prove semantic compatibility. The exact release mismatch (99 measured columns versus 33 paper model-input dimensions) remains material.
