# Multi-person readiness audit

Audit date: 2026-09-07.

## Current classification

**SINGLE_PERSON_READY**

The current product path is ready for one selected person per camera. It is
not a multi-person monitoring system and does not assign identity to people.

## Evidence from the implementation

1. `src/posture_ai/inference/pipeline.py:create_pose_landmarker` sets
   `num_poses=1` in `PoseLandmarkerOptions`.
2. `extract_pose_landmarks` returns `result.pose_landmarks[0]`.
3. `LiveFrameProcessor.process` reads `result.pose_landmarks[0]` and
   `result.pose_world_landmarks[0]`.
4. `RGBNativeV4Runner` accepts one 33-landmark pose and returns one prediction.
5. `TemporalRiskEngine` stores one probability history, one stable direction,
   one duration, and one cooldown state per instance.
6. `_base_record` and the live JSONL schema contain no `person_id`,
   `subject_id`, or anonymous track identifier.

## Direct answers

| Question | Verified answer |
|---|---|
| How many people can the current MediaPipe path process? | One pose slot. A second visible person is not independently processed. |
| Does V4 assume one subject? | Yes. One landmark array enters one V4 feature vector and one prediction. |
| Is temporal state per person? | No. The current live instance has one global state. |
| Does telemetry contain a person identifier? | No. It has session/frame/time fields, not a subject ID. |
| Would two people break assumptions? | The code may continue with one selected pose, but the result can ignore or switch between people. It must not be interpreted as two-person monitoring. |
| Would two cameras be easier? | Usually yes for a prototype: one person/zone per camera preserves the current contract, but each camera still needs its own session state and resource budget. |
| What can be reused? | Canonical landmarks, core visibility gate, V4 adapter/classifier, geometry functions, one-person temporal engine, and output concepts. |

## Required future changes

- Request/configure multiple MediaPipe pose slots and retain all returned poses.
- Add an anonymous runtime track layer that emits IDs such as `PERSON_01` and
  `PERSON_02`; do not use faces, names, or biometric identity.
- Keep one `TemporalRiskEngine`, neutral baseline, stable-core gate, and session
  accumulator per anonymous track.
- Add track creation, missed-frame timeout, re-association, and ID-switch
  diagnostics.
- Add camera/zone identifiers and an explicit `subject_id` to telemetry.
- Decide whether calibration is per person, per seat, or both.
- Define UI behavior when people overlap or a track is uncertain.
- Test with synthetic multi-pose results before any hardware run.

## Recommended deployment direction

For the first room prototype, zone cameras or one camera per workstation are
operationally simpler than one wide camera. A single wide camera reduces device
count but increases occlusion, pixel-size, side-view, and tracking problems.
The floor plan and real camera FOV are required before selecting a count.

## Safe future abstraction

The new `src/posture_ai/multiperson/schema.py` data model is deliberately
standalone and is not wired into the production single-person path. It carries
anonymous IDs, landmarks, quality, V4 result, geometry, and temporal state so a
future tracker can be tested without changing V1/V2/V4.
