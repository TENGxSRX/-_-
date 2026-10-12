# ANYA feature definitions

Status: **AUDITED; no supervised feature track trained**.

## Native representation (TRACK A candidate)

The verified native candidate is the 171 numeric pose columns:

1. `x0,y0,z0,vis0` through `x32,y32,z32,vis32` (33 indexed slots × 4 = 132).
2. `x0normalized,y0normalized,z0normalized` through index 12 (13 indexed slots × 3 = 39).

`frame`, `filename`, `clusterid`, `normclusterid`, and `Predicted_Label` are metadata or outputs and are excluded from a future feature matrix unless their role is independently justified. In particular, `Predicted_Label` must not be a feature or target for an ergonomic model.

## Normalization (TRACK B candidate)

The release already contains a column group named `normalized`, but the exact formula, reference landmark, scale, and handling of visibility are not documented in the CSV. The measured range alone does not prove hip centering, torso scaling, or equivalence to `src/posture_ai/features/pose_features.py`. A new normalization must therefore be treated as an experiment, not silently applied as if it reproduced the source.

## Engineered geometry (TRACK C candidate)

The project can calculate distances, angles, offsets, and left/right asymmetry only after indexed landmarks are mapped to anatomy. The ANYA release does not provide that mapping. Therefore no C7, neck, shoulder, hip, or spine landmark names are invented here, and no engineered geometry is promoted into a trained model.

If a source-backed mapping is supplied, every feature must be defined mathematically before training, for example `angle(a,b,c) = arccos(((a-b)·(c-b))/(||a-b|| ||c-b||))`, with explicit coordinate reference, scaling, visibility threshold, and missing-landmark policy.

## Engineering decision

The project’s MultiPosture 126-feature hybrid schema is not interchangeable with this ANYA schema. The exact landmark identity, order, normalization, and label target must be verified before an RGB/MediaPipe adapter or shared model is connected.

