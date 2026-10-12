# Live Hardware Calibration + Failure Analysis V1

Status: **REAL HARDWARE EVIDENCE — CALIBRATION NEEDED**

Session: `20260907T155515Z`  
Camera: index `0`  
Source files: `outputs/live/logs/webcam_20260907T155515Z.jsonl` and `outputs/live/session_reports/webcam_20260907T155515Z_summary.json`

This report uses the generated session files only. It does not retrain V4,
train V5 or ANYA, install YOLO, or alter raw data/models.

## Executive result

| Area | Result | Interpretation |
|---|---:|---|
| Camera I/O | PASS | Camera 0 opened and the session completed cleanup. |
| MediaPipe pose detection | 88.86% | Promising, but not sufficient for a complete validation session. |
| Five-core acceptance | 66.30% | NEEDS_CALIBRATION; the hips are the dominant failure source. |
| V4 live TUP/TLB | NEEDS_VALIDATION | Step 2 produced TUP only; the old log lacks class probabilities/features. |
| Geometry | NEEDS_CALIBRATION | Absolute head-depth proxy was almost always over threshold. |
| Temporal warning | NEEDS_RETEST | No warning event occurred; the test did not provide a supported sustained poor class. |
| FPS | MEASURED | Mean 10.95 FPS; this is diagnostic, not a target claim. |

The session is evidence of real camera behavior, not evidence of live model
accuracy. The validation controller used for this session did not yet have
the neutral-baseline stage, stable-ready gate, input-loss pause, or step-8
reacquisition wait. Those safeguards are now implemented for the next run.

## 1. Session facts

- Records: `1,249`
- Reported duration: `103.617 s`
- Pose-detection time: `88.86%`
- Five-core acceptance: `66.30%`
- Pose-lost time: `33.11%`
- V4 observed output: approximately `TUP 94.75%`, `TLB 5.25%`
- Forward indicator: `98.87%` of valid-pose time
- Left indicator: `8.04%` of valid-pose time
- Right indicator: `7.23%` of valid-pose time
- Warning events: `0`
- Longest continuous poor-posture duration in summary: `3.55 s`
- Mean measured loop rate: `10.95 FPS`

The JSONL contains `1,045` records with `pose_detected=true` and `808` records
with an accepted V4 prediction. It contains no `ai_probabilities`, no V4
feature vector, and no saved video. Therefore exact step-2 TUP/TLB probability
curves and V4 feature means cannot be reconstructed retrospectively from this
session.

## 2. Step-level observations

The summary rates below are frame-record rates for each logged validation
step. The indicator percentages in the last column are calculated among valid
V4-prediction records, matching the session summary's valid-time denominator.

| Step | Frames | Pose detection | Core accepted | AI labels | Forward / Left / Right | Main finding |
|---:|---:|---:|---:|---|---:|---|
| 1 | 107 | 100.00% | 100.00% | TUP 107 | 100 / 0 / 0% | Stable input, but absolute forward proxy already active. |
| 2 | 71 | 100.00% | 100.00% | TUP 71 | 100 / 0 / 0% | Lean-back instruction did not change the final V4 class. |
| 3 | 66 | 100.00% | 43.94% | TUP 29 | 100 / 0 / 0% | Both hips frequently fell below the visibility gate while leaning forward. |
| 4 | 68 | 100.00% | 67.65% | TUP 46 | 100 / 0 / 84.78% | Hips again caused most rejected frames while leaning left. |
| 5 | 71 | 100.00% | 100.00% | TUP 42, TLB 29 | 100 / 69.01 / 14.08% | Geometry changed strongly; V4 output also mixed. |
| 6 | 70 | 100.00% | 100.00% | TUP 57, TLB 13 | 100 / 22.86 / 0% | Some transient TLB/poor-duration evidence, but no warning threshold reached. |
| 7 | 84 | 48.81% | 48.81% | TUP 41 | 97.56 / 2.44 / 21.95% | Leaving the frame produced expected input loss. |
| 8 | 106 | 0.94% | 0.94% | TUP 1 | 100 / 0 / 100% | Return-to-frame step did not reacquire stable input. |
| 9 | 302 | 81.46% | 81.46% | TUP 245, TLB 1 | 100 / 0 / 0% | Mostly TUP; no supported sustained poor direction. |

## 3. Core landmark failure analysis

Failure counts below are calculated only over the `1,045` records where a
pose was detected. A failure means visibility was missing or below the
configured `0.5` core threshold.

| Landmark | Failures | Failure rate | Mean visibility | Median | P10 | P25 | P75 | P90 | >=0.3 | >=0.5 | >=0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nose | 0/1045 | 0.00% | 0.9874 | 0.9980 | 0.9790 | 0.9943 | 0.9991 | 0.9994 | 100.00% | 100.00% | 99.33% |
| left shoulder | 0/1045 | 0.00% | 0.9975 | 0.9986 | 0.9941 | 0.9970 | 0.9992 | 0.9996 | 100.00% | 100.00% | 100.00% |
| right shoulder | 0/1045 | 0.00% | 0.9951 | 0.9956 | 0.9898 | 0.9935 | 0.9974 | 0.9993 | 100.00% | 100.00% | 100.00% |
| left hip | 229/1045 | 21.91% | 0.6694 | 0.7783 | 0.2309 | 0.5557 | 0.8656 | 0.9225 | 84.78% | 78.09% | 59.81% |
| right hip | 233/1045 | 22.30% | 0.6677 | 0.7418 | 0.2427 | 0.5644 | 0.8547 | 0.9032 | 87.27% | 77.70% | 62.01% |

The hips account for all observed core-gate failures. The head and shoulders
were reliable even in the difficult steps. In step 3, left-hip failure was
`37/66` and right-hip failure was `35/66`, leaving `43.94%` core acceptance.
In step 4, the corresponding counts were `22/68` and `20/68`, leaving
`67.65%` acceptance. Pose detection remained `100%` in both steps, so the
problem is not person detection; it is usable hip landmark visibility under
the selected framing and lean angle.

### Camera guidance

For the next run:

1. Move the camera farther back until both hips have visible margin on every
   side; do not frame tightly around the head and shoulders.
2. Keep the full head-to-hip region inside the image before pressing SPACE.
3. Use a slightly wider field of view if available, while retaining enough
   resolution for the head and shoulders.
4. Put the camera near chest/shoulder height and avoid a low upward angle.
5. Use front or mild diagonal placement; avoid an occluding desk edge or
   chair arm across either hip.
6. Keep the chair and torso lighting even and avoid strong backlight.

These are framing recommendations, not distance measurements or clinical
requirements.

## 4. Why the forward indicator was active 98.87%

The current code defines the forward flag as:

`abs(head_forward_proxy) >= 0.20`

where `head_forward_proxy` is the normalized nose-to-shoulder **z** offset.
It is not the hip-centered trunk z component. In the valid records:

| Quantity | Step 1 mean | Step 2 mean | All valid mean | All valid median |
|---|---:|---:|---:|---:|
| `head_forward_proxy` | `-0.6414` | `-0.5876` | `-0.5733` | `-0.5591` |
| `trunk_vector[2]` / logged `forward_back_component` | `-0.1639` | `-0.1379` | `-0.1257` | `-0.1460` |
| `trunk_inclination_deg` | `-0.4476` | `-0.2085` | `-2.0418` | `-0.7447` |

The first quantity is naturally far beyond `0.20` in this camera/person
setup, including step 1 upright. The absolute rule therefore marks nearly
every valid frame. The sign is not the primary problem: reversing the sign
would still leave the absolute value above threshold. The root cause is an
absolute camera-relative depth proxy being used as an ergonomic forward flag
without a session-neutral reference. The logged `forward_back_component` name
can also be misleading because the actual directional flag uses
`head_forward_proxy`.

The new runtime calibration collects a contiguous stable natural-upright
baseline and subtracts it only for geometry indicators. It does not alter the
raw coordinates or the V4 input. This is the scientifically safer first
correction; thresholds remain project heuristics and are not silently widened.

## 5. Why step 2 did not produce Lean Backward

Step 2 contained `71/71` pose-detected and core-accepted records. Every logged
AI label was `TUP`. The available confidence values were extremely high:

- mean confidence: `0.999904`
- median: `0.999905`
- minimum: `0.999712`

However, the old JSONL did not record the two class probabilities or the V4
feature vector, and no frame video was saved. The result cannot therefore be
traced to an exact probability margin or feature-range comparison.

The available geometry moved only modestly from step 1 to step 2:

- head-depth proxy: `-0.6414` to `-0.5876`
- trunk z component: `-0.1639` to `-0.1379`
- trunk inclination: `-0.4476°` to `-0.2085°`

This is consistent with either a movement too small to leave the V4 TUP
region, a camera-domain/generalization gap, or both. With the evidence saved,
the defensible classification is **INCONCLUSIVE**, not a V4 failure claim.
V4 was not changed. The live logger now records instantaneous and smoothed
TUP/TLB probabilities plus the exact V4 feature names/values for the next
controlled run.

## 6. Step 8 return-to-frame failure

The exact logged timeline was:

- step 8 began at `73.550 s` (frame `841`) with pose/core not accepted;
- one accepted frame appeared at `74.740 s` (frame `862`);
- pose/core was lost again at `74.857 s` (frame `863`);
- the step ended at `78.590 s` while the remaining records were lost.

The old controller started the step timer immediately and did not require a
stable reacquisition. The single good frame was not enough to establish that
the user had returned. This is a validation-flow problem, not evidence that
the user or MediaPipe failed permanently.

The updated flow now:

- waits in a user-controlled `WAITING` state before each step;
- requires stable core visibility before SPACE can start a step;
- specifically blocks step 8 until core landmarks are reacquired;
- pauses an active step timer while core input is lost;
- records `INPUT GOOD` / `INPUT LOST` in telemetry and on the UI.

## 7. Why the warning did not trigger

The configured temporal settings are:

- `minimum_warning_duration = 15.0 s`
- `high_risk_duration = 60.0 s`
- `recovery_duration = 3.0 s`
- smoothing window `5`
- minimum persistence `0.5 s`

The temporal engine treats only `TLF`, `TLB`, `TLL`, and `TLR` as poor
directions. V4 supplies only `TUP` and `TLB`; the forward/left/right values
are explicitly experimental geometry indicators and do not become poor model
directions automatically. Step 9 contained `245` TUP records and only `1`
TLB record, with `56` pose-lost records. Its maximum continuous poor duration
was `0.0 s`; the session-wide maximum was `3.55 s` in step 6. Pose loss also
resets the temporal duration.

Therefore zero warning events is expected from this evidence. The warning
threshold was not reduced. A future warning retest must hold a supported
poor direction continuously for at least the configured duration with stable
input, or must be designed as a separate geometry-warning experiment rather
than silently fusing the two systems.

## 8. Runtime neutral calibration design

Implemented in `src/posture_ai/ergonomics/calibration.py`:

- setup requires a `1.5 s` continuous stable core gate;
- SPACE starts a `4.0 s` neutral collection interval;
- the user sits naturally upright during this interval;
- any invalid core frame clears the contiguous neutral streak;
- the baseline stores means for trunk inclination, lateral displacement,
  head-depth proxy, shoulder tilt, and trunk vector;
- geometry indicators use neutral-relative values after completion;
- V4 keeps its original raw World Landmark feature path;
- R resets an unfinished neutral calibration;
- the baseline is runtime/session configuration only, not training data.

Angle subtraction uses a circular shortest-angle difference. These are
engineering heuristics, not clinical thresholds.

## 9. V4 diagnostics and timing instrumentation

The old session cannot provide the requested step-2 probability/feature means.
The live path now records for future sessions:

- instantaneous `ai_probabilities` for TUP/TLB;
- temporally smoothed `ai_smoothed_probabilities`;
- ordered `v4_feature_names`;
- ordered `v4_feature_values`;
- raw `geometry` and neutral-relative `geometry_relative`;
- input state and whether the validation timer was paused.

The existing logger still does not claim per-stage accuracy. A future timing
instrumentation pass should separately time MediaPipe, V4, geometry, overlay,
and logging; this session only provides the measured end-to-end mean of
`10.95 FPS`.

## 10. Next controlled run

Use the calibration launcher:

`run_webcam_calibration.bat`

Expected flow:

`CAMERA SETUP -> stable core visibility -> SPACE -> NEUTRAL CALIBRATION -> SPACE per step -> 3-second preparation -> active measurement`

For step 8, return fully into frame and wait for the body-visibility panel to
show all five core landmarks before pressing SPACE. If input becomes lost
during a step, the timer should pause instead of counting invalid frames.

## Final status

WEBCAM HARDWARE: **PASS**  
MEDIAPIPE DETECTION: **PROMISING**  
CORE LANDMARK QUALITY: **NEEDS_CALIBRATION**  
V4 LIVE TUP/TLB: **NEEDS_VALIDATION**  
GEOMETRY: **NEEDS_CALIBRATION**  
TEMPORAL WARNING: **NEEDS_RETEST**  
LIVE FPS: **MEASURED — 10.95 FPS**

The live posture system is **not fully validated**. The next run must be a
user-guided calibration/validation session using the updated flow.

The full automated suite after these safeguards passes **85/85**. V4, V5,
ANYA, and raw datasets were not modified.
