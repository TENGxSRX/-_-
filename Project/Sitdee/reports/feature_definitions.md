# Feature definitions and adapter contract

## Source schema

MultiPosture supplies 33 MediaPipe-named landmarks in this exact order, each with `x`, `y`, `z`: nose; six eye points; two ears; two mouth points; shoulders; elbows; wrists; fingers/thumbs; hips; knees; ankles; heels; and foot-index points. The canonical order is owned by `src/posture_ai/pose/landmarks.py` and reused by `src/posture_ai/features/pose_features.py`.

## Feature tracks

- `raw`: the source 99 XYZ values without alteration.
- `normalized`: source keypoints treated as already hip-centred and divided by shoulder-centre distance from the hip-centred origin.
- `engineered`: 27 geometry features including trunk inclination/vector, shoulder/hip line angles, asymmetries, head-to-shoulder offsets, torso/limb lengths, and elbow/knee angles.
- `hybrid`: raw plus engineered features; this was the selected 126-feature track.

Angles use Euclidean vectors and clipped cosine values. Length features are normalized by torso length where defined. Degenerate lengths use an epsilon to avoid non-finite values.

## MediaPipe adapter contract

The offline adapter requires exactly 33 ordered MediaPipe landmark objects, checks each landmark's visibility (default threshold 0.5), reads `x/y/z`, subtracts the left/right hip centre, and returns the same 33×3 order. The exported model then rebuilds the selected `hybrid` feature names and rejects any name/order mismatch.

This proves identity, count, order, XYZ extraction, hip-centering, feature ordering, and engineered calculations in code. It does **not** prove exact source normalization equivalence: the source CSV's coordinate-generation and scale conventions are not fully documented, while live MediaPipe image coordinates and z values depend on the pose model and image. Accordingly, real-image adapter validation remains open and no forced incompatible transform is used.
