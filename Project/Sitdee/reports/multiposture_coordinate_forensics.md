# MultiPosture coordinate forensics

## Source evidence

| Claim | Status | Evidence |
|---|---|---|
| MediaPipe Pose Heavy was used | VERIFIED_FROM_SOURCE | IEEE Access paper methodology and Zenodo record |
| Frames were downscaled to 640×480 before pose extraction | VERIFIED_FROM_SOURCE | IEEE Access paper methodology |
| World Landmarks were extracted | VERIFIED_FROM_SOURCE | IEEE Access paper methodology |
| World coordinates are meters | VERIFIED_FROM_SOURCE | IEEE Access paper methodology and MediaPipe documentation |
| World origin is midpoint between hips | VERIFIED_FROM_SOURCE | IEEE Access paper methodology and MediaPipe documentation |
| StandardScaler was applied before model input | VERIFIED_FROM_SOURCE | IEEE Access paper methodology |
| Arms and facial landmarks except nose were removed for the paper's 33-input model | VERIFIED_FROM_SOURCE | IEEE Access paper methodology |
| Local release stores exactly the paper's 11-joint/33-value matrix | UNKNOWN | Official local CSV has 102 columns and 99 coordinate columns |
| Local CSV coordinate columns are named in canonical 33-landmark order | VERIFIED_FROM_LOCAL_DATA | exact local header inspection |
| Exact per-column scaler parameters used for the released CSV/model | UNKNOWN | no scaler artifact or preprocessing code in the local release |
| Mirroring or axis inversion was applied | UNKNOWN | not stated in source material inspected |

## Reconstructed pipeline

The strongest source-supported reconstruction is:

`RGB frame → resize 640×480 → MediaPipe Pose Heavy → pose_world_landmarks → hip-midpoint/world coordinates → StandardScaler for the paper's model input → paper feature selection`

The released `data.csv` is the authoritative local file and does not match the paper's stated 33-input dimensionality: it contains 33 named XYZ groups. This means the high-level source convention is resolved as **world landmarks + hip-centered origin**, but the exact relationship between the released 99 columns and the paper's reduced/scaled 33-input model matrix remains unresolved.

## What is not justified

No source supports arbitrary torso scaling, horizontal mirroring, z sign reversal, or fitting a transform to DriveNet labels. Those are diagnostic controls only in V3.

## Sources

- Zenodo record: https://zenodo.org/records/14230872
- IEEE Access paper DOI: https://doi.org/10.1109/ACCESS.2024.3510034
- Open full-text copy inspected for methodology: https://www.researchgate.net/publication/386356382_A_comparison_between_Multilayer_Perceptrons_and_Kolmogorov-Arnold_Networks_for_multi-task_classification_in_sitting_posture_recognition
- MediaPipe Pose documentation: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md

Local raw hash: `d21579200db9a57dd01c24fa6ece961b0be73aa0d5d402ae8e629761c03ebe38`.
