# Overnight run summary

Run date: 2026-09-06. Host: `DESKTOP-ALH6CUB`.

## Executive result

The overnight run completed the safe expansion work without changing the frozen project models or raw datasets. The strongest result is operational: MediaPipe and the five-landmark V4 RGB adapter stayed at 100% on a dense DriveNet pass. The remaining limitation is scientific coverage and cross-domain semantics, not pose detection.

## Track A — external dataset search

Decision: **NO_VALID_EXTERNAL_5CLASS_DATA**.

No new archive was downloaded. MVSP/LGCSPNet is still the highest-value manual-access candidate because its published categories include upright/back/left/right, but its official README points to a Baidu archive whose contents, label join, participant IDs, and license were not verified locally. Its `down` label is a compound forward-and-bend posture and was not silently mapped to TLF. MOPED25 is a secondary research lead, not a drop-in sitting five-class source. See `reports/overnight_external_dataset_search.md`.

## Track B — frozen DriveNet/V4 benchmark

All six verified videos were processed read-only:

- source frames decoded: **12,395**
- dense MediaPipe/V4 frames at 10 FPS: **4,134**
- pose detected: **4,134 / 4,134 (100%)**
- five-core-landmark accepted: **4,134 / 4,134 (100%)**
- POSE_LOST in 1/2/5/10 FPS reports: **0**
- 1 FPS rows: **417**; 2 FPS: **830**; 5 FPS: **2,069**; 10 FPS: **4,134**
- raw class switches across six clips at 5 FPS: **3**; temporal smoothing reduced this to **1**
- dense wall time: **500.21 s**, RSS `183.3 → 222.9 MB`

The V4 model is frozen and the DriveNet labels come from the same source used to train this pilot. Therefore the prediction distribution is reported as a source-domain diagnostic, not independent accuracy. The only notable diagnostic anomaly was `SS_2`: 2 of 355 dense 5-FPS frames were predicted TLB while the remainder were TUP. `SS_3` had three feature dimensions outside the P01/P02 training range for more than half its accepted rows, despite 100% pose/core acceptance. See `reports/overnight_performance_benchmark.md`.

## Track C — feature and confound audit

The frozen V4 training cache was audited with participants 1/2 as the declared training domain and participant 3 as the holdout. The largest holdout shifts were:

- `nose_y_world_hip_centered_scaled`: absolute SMD **1.212**
- `left_hip_y_world_hip_centered_scaled`: **0.932**
- `right_hip_y_world_hip_centered_scaled`: **0.932**
- `right_shoulder_y_world_hip_centered_scaled`: **0.925**

The frozen LogisticRegression model's largest coefficient magnitudes were shoulder x, nose x, and shoulder y features. Each participant contains both TUP and TLB, so participant identity is not perfectly equal to the target; nevertheless six videos and temporally correlated frames are too small to support broad generalization claims. No new model was trained from this audit.

## Track D — architecture and decision

- Experimental offline V4 + geometry demo: **prepared and smoke-tested on 60 frames of SS_1**. It reports V4 direction plus camera-relative geometry indicators and temporal state; it is not a medical score and does not replace V4.
- V5: **not trained**. The source-backed RGB inventory still lacks defensible TLF/TLL/TLR rows.
- ANYA: **not trained**. The downloaded release remains pseudo-label-only for this project.
- Coordinate adapter: **PARTIALLY_VALIDATED / not fully real-RGB semantic validation**. V4's own RGB-native path is internally consistent for its DriveNet pilot, but it is not proof that the processed MultiPosture representation is reproducible from arbitrary RGB cameras.
- YOLO: **not installed**. Current evidence supports keeping MediaPipe as the primary backend; the blocker is data/coordinate-domain coverage, not detection.

## Final status matrix

| Component | Status |
|---|---|
| MediaPipe RGB pose detection | READY — 100% on 4,134 dense DriveNet frames |
| V4 five-core extraction | READY — 100% accepted on dense benchmark |
| Geometry indicators | READY — experimental/camera-relative |
| Temporal smoothing | READY — tested; 3 raw switches → 1 smoothed at 5 FPS |
| MultiPosture V1 | READY, preserved |
| MultiPosture V2 | READY, preserved; grouped CV macro-F1 0.9925 ± 0.0087 |
| MultiPosture V3 | Preserved; coordinate semantics still not fully resolved |
| RGB-native V4 | FROZEN PILOT — TUP/TLB only |
| RGB-native V5 | NOT TRAINED — missing TLF/TLL/TLR |
| ANYA | NOT TRAINED — no verified target |
| Hybrid V4 + geometry | EXPERIMENTAL OFFLINE DEMO READY |
| Offline image/video pipeline | READY and tested |
| Live webcam | NOT TESTED — HARDWARE NOT CURRENTLY AVAILABLE |
| YOLO | DEFERRED / not installed |

## Integrity and tests

- C: free space after run: **23.72 GB**; no new large archive was downloaded.
- ANYA four CSV hashes: all matched the prior inventory.
- MultiPosture raw hash: `d21579200db9a57dd01c24fa6ece961b0be73aa0d5d402ae8e629761c03ebe38`.
- DriveNet six benchmark-video hashes matched the existing verified inventory.
- Full test suite: **65 passed in 9.68 s** using a dedicated project temp directory.
- No file under `datasets/raw/` was modified, renamed, or deleted; V1/V2/V3/V4 artifacts were not overwritten.

## Recommended next step

Choose one scientifically valid data action: manually verify and legally import MVSP/LGCSPNet if its archive provides joinable labels and participant/view metadata, or collect a small multi-participant office RGB set covering all five direction classes. Until then, keep V4 as a two-class RGB pilot and MultiPosture as the five-class research reference; do not claim five-class real-camera compatibility.
