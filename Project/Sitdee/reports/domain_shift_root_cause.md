# Domain-shift root-cause analysis

The ten features previously identified as outside the MultiPosture training min/max for more than half of the 900 external diagnostic rows were inspected without clipping or forced rescaling.

| Feature | Source landmark(s) | Training range | External range | Training mean ± std | External mean ± std | Max dependency low-vis | Root-cause assessment |
|---|---|---|---|---|---|---:|---|
| `left_hip_z` | left_hip | -0.0046…0.0705 | -0.1534…-0.1105 | 0.0296 ± 0.0101 | -0.1311 ± 0.0071 | 0.0% | Likely coordinate convention/projection and left-right body-frame difference; hip-centering removes translation but not camera-axis scale/orientation. |
| `right_hip_z` | right_hip | -0.0679…0.0083 | 0.1105…0.1534 | -0.0267 ± 0.0105 | 0.1311 ± 0.0071 | 0.0% | Likely coordinate convention/projection and left-right body-frame difference; hip-centering removes translation but not camera-axis scale/orientation. |
| `left_hip_x` | left_hip | 0.0649…0.1292 | -0.0105…0.0298 | 0.1000 ± 0.0090 | 0.0199 ± 0.0058 | 0.0% | Likely coordinate convention/projection and left-right body-frame difference; hip-centering removes translation but not camera-axis scale/orientation. |
| `right_hip_x` | right_hip | -0.1297…-0.0658 | -0.0298…0.0105 | -0.0985 ± 0.0089 | -0.0199 ± 0.0058 | 0.0% | Likely coordinate convention/projection and left-right body-frame difference; hip-centering removes translation but not camera-axis scale/orientation. |
| `left_knee_x` | left_knee | -0.0534…0.4223 | -0.2436…-0.1749 | 0.1899 ± 0.1052 | -0.2131 ± 0.0173 | 74.9% | Strongly affected by low-visibility lower limbs and side-view occlusion; also subject to camera projection and depth convention. |
| `head_to_shoulder_z` | nose, left_shoulder, right_shoulder | -1.1387…-0.2341 | -0.3844…0.1750 | -0.6443 ± 0.0865 | -0.0491 ± 0.0825 | 0.0% | Depth-axis convention/projection mismatch is the leading explanation; this is not evidence of a human posture error. |
| `left_shoulder_to_hip` | left_shoulder, left_hip, right_hip | 0.7919…1.2367 | 1.0326…1.5050 | 1.0444 ± 0.0500 | 1.2988 ± 0.0726 | 0.0% | Derived ratio is sensitive to side-view foreshortening, shoulder/hip localization, and different source normalization. |
| `left_ankle_x` | left_ankle | -0.2952…0.6003 | -0.3970…-0.2250 | 0.1305 ± 0.1847 | -0.3184 ± 0.0337 | 97.2% | Strongly affected by low-visibility lower limbs and side-view occlusion; also subject to camera projection and depth convention. |
| `right_elbow_z` | right_elbow | -0.5333…0.2182 | 0.0853…0.3968 | -0.0990 ± 0.1203 | 0.2452 ± 0.0548 | 99.6% | Depends on a low-visibility elbow in the external side view and on MediaPipe z semantics. |
| `right_knee_z` | right_knee | -0.3877…0.0983 | -0.0125…0.3460 | -0.2609 ± 0.0705 | 0.1103 ± 0.0645 | 100.0% | Strongly affected by low-visibility lower limbs and side-view occlusion; also subject to camera projection and depth convention. |

## Cross-cutting findings

- The external rows were generated from MediaPipe RGB image-normalized x/y/z and hip-centered only for diagnostics. MultiPosture stores source coordinates documented as hip-centered, but its exact camera projection, axis orientation, scale, and z convention are not fully specified.
- Side/three-quarter vehicle footage changes foreshortening and visibility relative to the MultiPosture home-environment recordings. This is a camera/domain effect, not proof that the person has a different project posture label.
- Several shifted features depend directly or indirectly on low-visibility limbs. Their coordinates are retained in diagnostics but are not silently trusted by the V2 adapter.
- Z values are especially unsafe to equate across sources: MediaPipe z is model/image-relative depth semantics, while the dataset's generation convention is not independently documented in the local CSV.
- The correct response is an explicit adapter/feature redesign and source-convention validation, not clipping values into the training range.
