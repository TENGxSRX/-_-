# Live Webcam Prototype V1 / UX Test Mode V2 design

Status: **REAL HARDWARE ANALYZED — CALIBRATION NEEDED**

Hardware status: **PASS — CAMERA 0 SESSION COMPLETED**

## Scope

This prototype is a real-time OpenCV application around the frozen
RGB-native V4 pilot. The UX Test Mode V2 keeps the camera image as the main
visual element, uses small translucent edge cards, and is prepared for camera
indices 0, 1, and 2.

The processing frame is never mirrored. The application runs MediaPipe, V4,
and geometry on the original camera orientation, then mirrors only a copy for
the user-facing preview. `--no-mirror` is available for a non-mirrored display.

V4 is a two-class pilot only:

| Internal label | Screen label |
|---|---|
| `TUP` | `UPRIGHT` |
| `TLB` | `LEAN BACKWARD` |

The application never presents Forward, Left, or Right as V4 AI classes.
Those values are separate camera-relative geometry indicators.

## Frame path

`OpenCV camera -> MediaPipe Pose World Landmarks -> five-core visibility gate -> frozen V4 feature adapter -> frozen LogisticRegression -> temporal engine -> geometry indicators -> overlay and telemetry`

The five required landmarks are nose, both shoulders, and both hips. Image
landmark visibility gates the frame; World Landmarks provide the V4 input.
No low-visibility imputation is performed.

Before validation, the application stays on a CAMERA SETUP screen until the
five core landmarks pass a continuous stable-ready gate and the user presses
SPACE. It then collects a short runtime-only neutral baseline. The user must
press SPACE to start each validation step; step 8 additionally waits for core
landmark reacquisition. During an active step, the measurement timer pauses
while core input is lost. The framing assistant reports the five landmarks as
visible/missing and gives simple adjustment messages. Coverage hints are
image-framing heuristics, not distance measurements.

## Temporal behavior

`configs/posture_engine.yaml` is the single source of temporal smoothing,
confidence, persistence, warning, recovery, cooldown, visibility, and
geometry settings. The webcam code does not duplicate those thresholds.

The states are `NORMAL`, `OBSERVE`, `WARNING`, `HIGH_DURATION_RISK`, and
`POSE_LOST`. These are product/engineering states, not medical scores.

## User controls

- `SPACE`: start neutral calibration, or start/advance the confirmed next step
- `R`: restart the current validation step
- `Q` or `ESC`: exit safely

Validation uses a four-second neutral baseline, a three-second preparation
countdown, configurable hold times, input-loss pause, and a short STEP
COMPLETE/final summary card. It does not create labels or accuracy scores.

The runtime baseline subtracts natural session geometry only from experimental
geometry indicators. It never alters the frozen V4 feature input and is not
training data. Calibration values are shown only when `--calibration` is
enabled.

## Outputs

- JSONL telemetry: `outputs/live/logs/`
- Session summaries: `outputs/live/session_reports/`
- Optional annotated MP4: `outputs/live/videos/`

Validation mode records behavioral observations only. It does not label,
retrain, or calculate five-class accuracy.

## Safe operation

Press `Q` or `ESC` to exit. The camera, MediaPipe landmarker, writer, and
display window are released in a `finally` block. Recording and sound are off
unless explicitly enabled.

## Known limitations

V4 is not a five-class model, and real-camera semantic compatibility remains a
deployment validation question. Geometry signs are camera-relative and must be
checked on the first real camera. The output is descriptive monitoring
telemetry, not medical diagnosis or a health score.
