# Webcam Human-Ground-Truth V1 readiness

## Implementation

- Collector: `scripts/collect_webcam_human_gt_v1.py`
- Participant IDs are anonymous (`P001` format).
- Labels are accepted only after the operator presses the matching key: `1=TUP`, `2=TLF`, `3=TLB`, `4=TLL`, `5=TLR`.
- The UI is blind: no posture-model prediction is loaded, displayed, or stored.
- Each confirmed round settles for 2 seconds and records for 5 seconds at a globally limited pose rate.
- Accepted and rejected evidence are written separately. No image or video is stored.
- The current padded-crop contamination check and selected/all-box safety context are preserved.
- Hips are optional; each accepted record stores the observed hip-visibility state.

## Verification

- Focused regression suite: 38 passed.
- Live USB source: opened at camera index 0 with 1280x720 and requested 30 FPS.
- Live readiness run: 2,247 frames, one anonymous track, no decode failures.
- The operator did not click/confirm a target or label during this run, so it was stopped safely with zero completed segments and zero accepted observations.
- Generated readiness session: `datasets/derived/webcam_human_gt_v1/SESSION_20260919T033830Z_P001/`
- Dataset audit: PASS for the empty readiness session; this does **not** make it training-ready.
- Frozen V4 was not called and remained SHA-256 `af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd`.

## Status

The collection software is ready, but the requested human-confirmed 5-class P001 dataset has not yet been collected. A person must physically select the target, perform each requested posture, and press the matching label key. Until at least three completed rounds per class exist, the dataset is not structurally ready for future V5 training.
