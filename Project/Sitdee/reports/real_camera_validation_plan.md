# Offline and future real-camera validation plan

## Implemented now

The CLI accepts an image file, image directory, or prerecorded video:

```text
python -m src.posture_ai.inference --source <image-or-directory-or-video> --model outputs/models/multiposture_v1 --pose-model metadata/pose_landmarker_heavy.task
```

The pipeline performs MediaPipe Pose extraction, 33-landmark validation, visibility rejection, the exported feature adapter, prediction, confidence reporting, and optional video temporal smoothing with rolling probabilities and persistence frames. No webcam source was opened on this machine. `--source 0` is reserved for a future run.

The SitStraightNet image smoke test completed through the offline path and returned `no_pose_detected` for the selected unlabeled image. That is a valid smoke-test outcome, not a prediction result and not a label claim. The generated two-frame blank video test exercised the video path without camera hardware.

## Compatibility gate

Before claiming RGB image/video model validity, verify landmark identities, order, x/y/z definitions, hip-centering, scale, visibility policy, feature order, and geometry calculations against the model's exported 126-column schema. The implementation currently proves code-level order and feature-name agreement, but not exact source normalization equivalence. Therefore the current status is:

`MODEL TRAINING READY`

`REAL-IMAGE INFERENCE ADAPTER NOT YET VALIDATED`

## Future webcam validation

When hardware is available, use a declared source `0` only after confirming the device and obtaining consent. Record camera model, resolution, frame rate, lighting, distance, seating setup, participant/session IDs, and ground-truth protocol. Keep participant/session/video groups disjoint from training; save raw media separately from derived predictions; report frame coverage, no-pose rate, low-visibility rate, confidence distribution, class distribution, temporal persistence behavior, and failure cases. Do not describe the result as clinical screening or Office Syndrome diagnosis.

Expected current wording:

`LIVE WEBCAM VALIDATION: NOT TESTED — HARDWARE NOT CURRENTLY AVAILABLE`
