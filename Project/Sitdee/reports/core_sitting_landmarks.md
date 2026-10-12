# Core sitting landmarks

## Candidate design

The experimental V2 core is intentionally minimal and maps directly to the posture questions required by this project:

### CORE_REQUIRED

`nose`, `left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`.

These support head-to-shoulder offsets, shoulder/hip alignment, trunk inclination, and left/right torso lean. They are the only landmarks used in the V2 model feature schema.

### OPTIONAL

`left_ear`, `right_ear`, `left_elbow`, `right_elbow`.

Ears can add head orientation and elbows can add upper-limb context, but neither is required for the core V2 adapter. Under the tiered policy they are reported as explicit quality warnings when visibility is below 0.2; no silent imputation is performed.

### LOW_VALUE_FOR_OFFICE_POSTURE

left_eye_inner, left_eye, left_eye_outer, right_eye_inner, right_eye, right_eye_outer, mouth_left, mouth_right, left_wrist, right_wrist, left_pinky, right_pinky, left_index, right_index, left_thumb, right_thumb, left_knee, right_knee, left_ankle, right_ankle, left_heel, right_heel, left_foot_index, right_foot_index.

These landmarks may be useful for other tasks, but they are not necessary for the intended seated head/shoulder/trunk/hip posture state and are vulnerable to desk, chair, steering-wheel, and lower-body occlusion.

## Decision

Do not change V1. V2 is a separate core-only experimental model family with raw, normalized, geometry, and hybrid feature tracks. The V2 adapter still requires exact finite XYZ for the five core landmarks and visibility ≥ 0.5; it never treats low-visibility coordinates as automatically trustworthy.
