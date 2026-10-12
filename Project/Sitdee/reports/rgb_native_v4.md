# RGB-native posture training track V4

## Status

**RGB-NATIVE V4: PILOT TRAINED — TWO COMPATIBLE CLASSES ONLY**

V4 uses the same authoritative pipeline for its training and offline inference path: `RGB video → MediaPipe Pose World Landmarks → hip-midpoint translation → V4 feature family → classifier`. It is intentionally not connected to the MultiPosture V1/V2/V3 artifacts.

## Dataset and labels

- Official source: [DriveNet v6](https://data.mendeley.com/datasets/v37nxdn2gh/6), license `CC BY 4.0`.
- Locally used mapped videos: `6`; sampled frames per video: `300`.
- Accepted mapped training rows: `1800`.
- `SS → TUP` and `LB → TLB` are COMPATIBLE mappings.
- `MB` and `RP` are AMBIGUOUS and were excluded from supervised target training; no label was invented.
- Participant split was predeclared: participants 1/2 train, participant 3 holdout; no participant overlap.

## Model result

- Selected model: `LogisticRegression` with `normalized_world` features (15).
- Grouped-CV macro F1: `1.0000 ± 0.0000`.
- Participant-3 holdout macro F1: `1.0000`; balanced accuracy `1.0000`.
- These are a two-class, three-participant source-domain pilot, not six-class office-camera generalization.

## Required interpretation

- MultiPosture remains the five-class direction model and is preserved unchanged.
- V4 demonstrates a same-pipeline RGB training feasibility track, but it does not yet cover TLF, TLL, TLR, or SLOUCH.
- The frozen DriveNet re-test is a same-source overlap diagnostic, not an independent external test.
- No webcam device was opened; no live-camera result is claimed.
- No ANYA training, YOLO installation, or raw-dataset modification was performed.
