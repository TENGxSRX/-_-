# ERGONOMIC POSTURE ENGINE V1 design

Verification date: 2026-09-05

## Purpose and boundary

V1 is an offline posture-monitoring prototype. It screens for persistent
non-upright direction states and reports geometric posture-risk indicators. It
does not diagnose Office Syndrome, injury, disease, or any medical condition.

## Architecture

```text
Image / prerecorded video
          |
          v
MediaPipePoseBackend (PoseBackend interface)
          |
          v
Canonical PoseLandmarks (33 landmarks, x/y/z/visibility)
          |
          +--> strict MultiPostureFeatureAdapter --> HistGradientBoosting
          |                                             |
          |                                             v
          |                                      direction + confidence
          |
          +--> Geometry Engine --> experimental indicators
          |
          v
TemporalRiskEngine --> event log + annotated video + session summary
```

The model direction path and deterministic geometry path are separate. The
temporal engine combines their outputs only into a product risk state.

## Supported direction states

The trained MultiPosture upper-body labels are:

| Source label | Display state | V1 status |
|---|---|---|
| `TUP` | Upright | Core model state |
| `TLF` | Lean Forward | Core model state |
| `TLB` | Lean Backward | Core model state |
| `TLL` | Lean Left | Core model state |
| `TLR` | Lean Right | Core model state |

## Risk state separation

- `NORMAL`: upright with no active experimental indicator.
- `OBSERVE`: non-upright direction has not persisted long enough, or an
  experimental indicator is active.
- `WARNING`: a non-upright direction persists for the configured warning
  duration.
- `HIGH_DURATION_RISK`: the warning direction persists for the configured
  longer duration.
- `POSE_LOST`: no sufficiently valid pose is available; this is not a risk
  judgment and does not accumulate poor-posture duration.

## Product settings

`configs/posture_engine.yaml` contains the initial settings. The 15-second
warning and 60-second high-duration values are project behavior settings, not
medical thresholds. The configuration also controls rolling probability
smoothing, discrete majority voting, minimum direction persistence, visibility,
recovery, and warning cooldown.

## Offline interfaces

The CLI supports `--source` with an image file, image directory, or prerecorded
video. The V1 runner writes JSONL events, a session JSON summary, a performance
JSON report, and an annotated MP4 for video sources. Webcam source `0` is
intentionally disabled in the V1 runner and was not executed.

## Dataset roles

- MultiPosture: supervised direction model only.
- ANYA: research/reference only; `PSEUDO_LABEL_ONLY`, no V1 training.
- SitPose: secondary research only, not merged.
- SitStraightNet: unlabeled real-image domain smoke test only.
- YOLO Pose: future backend interface only; not installed or benchmarked.
