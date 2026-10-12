# Overnight System Preparation Progress

Run completed across 2026-09-07 → 2026-09-08 (Asia/Bangkok).

## START

- Run scope: prepare demonstration, room-planning, multi-person readiness and Acer workflow.
- Safety: no new dataset download, no model training, no ANYA/V5, no YOLO, no webcam/source 0, no raw-data or saved-session mutation.
- Starting baseline: 89 automated tests; existing frozen V4 and V1/V2 artifacts preserved.

## ARCHITECTURE

- Completed: authoritative English and Thai architecture maps.
- Verified in code: MediaPipe Tasks PoseLandmarker is configured with `num_poses=1`; runtime consumes the first pose; one temporal state is used.

## MULTI_PERSON

- Completed: `SINGLE_PERSON_READY` readiness report and Thai explanation.
- Completed: non-breaking anonymous `PoseSubject` / `MultiPersonFrameResult` data contract for future work.
- Tracking and multi-person inference are intentionally not implemented tonight.

## ROOM_PLANNER

- Completed: JSON/YAML validation, FOV/distance/occlusion estimates, target coverage, candidate scoring, greedy estimate and SVG renderer.
- Completed: `TEST_ROOM_ONLY` fixture and room measurement checklist.
- Pending: real dimensions, camera FOV and real-room decision.

## DASHBOARD

- Completed: file-based offline dashboard module and CLI; AI posture is separated from geometry indicators; missing telemetry is never guessed.
- Completed: generated a `TEST_FIXTURE` dashboard and a dashboard from an existing saved session with its JSONL timeline.

## LOW_SPEC

- Existing: DESKTOP/BALANCED/LOW_SPEC scheduling and wall-clock timestamp behavior.
- Completed: small prerecorded smoke benchmark and timing regression test; no Acer-specific FPS claim.

## GIT_PREP

- Completed: Thai safe Git bootstrap guide.
- Repository initialization and remote publication remain intentionally untouched.

## INSTRUCTOR_REPORT

- Completed: Thai instructor summary, privacy design and room checklist.

## TESTS

- Completed: full suite passed 100 tests; Acer lightweight workflow passed its 15-test selection.

## FINAL

- Completed: ANYA CSVs rehashed and matched the prior manifest; MultiPosture raw
  hash matched the recorded project hash; V1/V2/V4 model snapshots were
  captured without writes; original Acer bundle remained unchanged.
- Completed: final versioned `outputs/laptop_dev_bundle_v3.zip` verified with
  136 entries and zero forbidden dataset/video/cache entries.
- Completed: C: free space remained above the 15 GiB safety floor (`21.82 GiB`
  at final check).
- Final safety result: no model training, no ANYA/V5 training, no YOLO install,
  no webcam/source 0 access, no raw-data mutation and no saved-session mutation.

## RTSP AI LAB LOW-LATENCY V2 FOLLOW-UP (2026-09-09)

- Completed: RTSP AI Lab now uses a separate capture reader with a one-slot
  latest-frame buffer for RTSP; old frames are intentionally replaced and the
  application queue cannot grow without bound.
- Completed: measured frame age, inference/render timing, source FPS, AI FPS,
  captured/inferred/dropped frames, stale events, decode failures and
  reconnects. MediaPipe timestamps are monotonic and session-relative.
- Completed: MAIN inference copy defaults to 960×540 and display copy defaults
  to 1280×720; the source frame remains available for UI/debug. Default view is
  non-mirrored and exposes D=debug, L=landmarks, Q=quit.
- Completed: real short benchmark on HIKVISION_MAIN and HIKVISION_SUB with no
  recording, controlled collection or training. Latest results are in
  `reports/rtsp_ai_lab_low_latency_v2_th.md` and
  `reports/rtsp_ai_lab_main_sub_benchmark_th.md`.
- Integrity: V1/V2/V4 and raw datasets were not written. The local
  `.env.example` was sanitized to placeholders; the real `.env` was not
  printed or modified. No real credential-bearing RTSP URL remains in code or
  documentation; synthetic test credentials are limited to tests.
- Tests: final full suite passed `144 tests`.
- Disk: C: had approximately `18.85 GiB` free at final check.
