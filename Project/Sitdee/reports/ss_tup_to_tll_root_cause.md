# SS → TLL root-cause trace

## Frozen observation

- Clip: `SS_1.mp4`
- Source mapping: `SS → TUP`
- Previous/current image-space V2 result: `TLL` for all 300 evaluated frames
- Source-supported world candidate result: `{'TUP': 1, 'TLL': 299}`

## Candidate prediction comparison

| Candidate | Representation | SS prediction distribution | Mean confidence |
|---|---|---|---:|
| `adapter_v3_a_world_hip_centered` | `world_landmarks` | `{'TUP': 1, 'TLL': 299}` | 0.9997 |
| `adapter_v3_b_world_hip_centered_zflip` | `world_landmarks` | `{'TUP': 1, 'TLL': 299}` | 1.0000 |
| `adapter_v3_c_image_hip_centered` | `normalized_image_landmarks` | `{'TUP': 1, 'TLF': 1, 'TLL': 298}` | 0.9949 |
| `adapter_v3_d_image_hip_centered_zflip` | `normalized_image_landmarks` | `{'TUP': 1, 'TLB': 1, 'TLL': 298}` | 0.9986 |
| `adapter_v3_e_world_hip_centered_mirror` | `world_landmarks` | `{'TUP': 1, 'TLR': 299}` | 0.9975 |

## Trace of the source-supported world candidate

First-frame world-core coordinates after hip recentering:

```json
{
  "nose": [
    -0.06200690381228924,
    -0.5402642646804452,
    -0.07973100244998932
  ],
  "left_shoulder": [
    0.13211012072861195,
    -0.346841380931437,
    -0.13920321315526962
  ],
  "right_shoulder": [
    0.05793765373528004,
    -0.46107144746929407,
    0.19014938920736313
  ],
  "left_hip": [
    0.016846129670739174,
    0.019738405011594296,
    -0.06657717376947403
  ],
  "right_hip": [
    -0.016846129670739174,
    -0.019738405011594296,
    0.06657717376947403
  ]
}
```

First-frame V2 probabilities:

```json
{
  "TLB": 6.669638787619404e-05,
  "TLF": 5.0800251370895556e-05,
  "TLL": 0.0003423245829795967,
  "TLR": 3.487943794678382e-05,
  "TUP": 0.9995052993398266
}
```

| Feature | Permutation importance | First SS value | Training mean | Training std |
|---|---:|---:|---:|---:|
| `trunk_vector_z` | 0.228513 | 0.06127 | -0.16748 | 0.30888 |
| `trunk_vector_y` | 0.046907 | -0.97160 | -0.89106 | 0.16410 |
| `left_shoulder_x` | 0.014764 | 0.13211 | 0.17980 | 0.09638 |
| `nose_x` | 0.000000 | -0.06201 | 0.00508 | 0.11950 |
| `nose_y` | 0.000000 | -0.54026 | -0.45220 | 0.11909 |
| `nose_z` | 0.000000 | -0.07973 | -0.34346 | 0.15027 |
| `left_shoulder_y` | 0.000000 | -0.34684 | -0.35686 | 0.07355 |
| `left_shoulder_z` | 0.000000 | -0.13920 | -0.09456 | 0.13208 |
| `right_shoulder_x` | 0.000000 | 0.05794 | -0.13802 | 0.10905 |
| `right_shoulder_y` | 0.000000 | -0.46107 | -0.40694 | 0.07836 |
| `right_shoulder_z` | 0.000000 | 0.19015 | -0.04524 | 0.12988 |
| `left_hip_x` | 0.000000 | 0.01685 | 0.10002 | 0.00901 |
| `left_hip_y` | 0.000000 | 0.01974 | 0.00221 | 0.01922 |
| `left_hip_z` | 0.000000 | -0.06658 | 0.02958 | 0.01008 |
| `right_hip_x` | 0.000000 | -0.01685 | -0.09848 | 0.00894 |

## Interpretation

The world-landmark candidate is the source-supported control, but the trace does not establish that SS→TLL is caused by one axis. The remaining plausible combination is: released CSV/model-input dimensionality mismatch, downstream StandardScaler not represented in the current V2 artifact, camera/viewpoint/domain gap, and/or source-class semantic mismatch. No label-driven transform search was performed.
