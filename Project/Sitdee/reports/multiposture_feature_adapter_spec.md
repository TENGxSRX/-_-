# MultiPosture feature adapter specification

## Source contract

The local MultiPosture training file contains 33 MediaPipe-order landmarks and
99 XYZ values. The exact canonical order is defined once in
`src/posture_ai/pose/landmarks.py` and reused by feature extraction. The
exported model expects the exact 126-column hybrid schema in
`outputs/models/multiposture_v1/feature_config.json`.

## Transform contract

For a canonical pose with landmark coordinates `p_i = (x_i, y_i, z_i)`:

1. Require all 33 landmarks and visibility at or above the configured project
   threshold (default 0.5).
2. Compute hip midpoint `h = (p_left_hip + p_right_hip) / 2`.
3. Produce the 99-value model input `q_i = p_i - h` in exact landmark/axis
   order. No second torso rescaling is applied to the selected hybrid track.
4. Append the deterministic engineered feature vector in the exact order
   returned by `engineered_geometry_features`.
5. Compare the resulting names to the exported model's names byte-for-byte and
   in order. Any mismatch fails loudly.

The model's selected feature vector is therefore:

```text
99 hip-centred XYZ values + 27 engineered geometry values = 126 features
```

## What is verified

- 33-landmark count and canonical index identity are implemented.
- 99 XYZ ordering matches the training CSV schema.
- Hip-centering is explicit and deterministic.
- The 126 feature names and order match the exported model package.
- Missing, non-finite, or low-visibility landmarks are rejected.
- No silent feature-column reorder or positional fallback is used.

## What is not proven

The stored CSV does not fully specify the original camera/MediaPipe coordinate
generation, model version, image normalization, or z-scale convention. A live
MediaPipe image therefore cannot be claimed semantically equivalent merely
because the column names and hip-centering transform match. The adapter's
`semantic_compatibility_validated` status remains `False` until a paired,
source-compatible validation experiment is available.

**Status: REAL-IMAGE ADAPTER NOT VALIDATED.**

## Alternate pose backends

`src/posture_ai/pose/backend.py` defines the small `PoseBackend` contract. A
future YOLO backend must return the same canonical 33-landmark representation;
the ergonomic engine should not need to know which backend produced it.

