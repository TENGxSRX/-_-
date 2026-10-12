# Final project status

## Person-centric monitoring update — 2026-09-10

Generic anonymous PERSON tracking is now the recommended primary runtime;
seat-aware mode remains available as an optional deployment mode. Each person
has independent activity, frozen V4/Geometry, temporal, and lifecycle state.
Standing pauses sitting-posture monitoring. UNKNOWN pauses risk accumulation.
The activity gate is explicitly `GEOMETRY_EXPERIMENTAL`; no Sit/Stand AI was
trained. Controlled human-labelled activity collection is prepared but not
started. The full suite now passes `168 passed`.

The latest short RTSP run (`HIKVISION_MAIN`, `max_people=4`, 8 seconds) had
100% pose/core validity on inferred poses, mean detected people 2.45, maximum
3, maximum active tracks 4, source 43.15 FPS, inference 3.39 FPS, and zero
decode failures/reconnects. The experimental activity gate returned UNKNOWN
for all 81 subject observations because sufficient lower-body evidence was not
available; sitting V4 evaluation was therefore paused safely. This is runtime
telemetry, not person-count, tracking, activity, or posture accuracy.

## Hierarchical CCTV intelligence update — 2026-09-10

The desktop now contains an evidence-gated hierarchical decision contract:

```text
VideoSource → MediaPipe multi-pose → anonymous PERSON_XX tracks
           → frozen V4 + Geometry + per-person Temporal
           → activity → sitting quality → subtype decision gate
```

The activity, sitting-quality, and subtype models are **not trained**. The
current dataset gate is `STANDING_DATA_INSUFFICIENT` and
`SITTING_POSTURE_DATA_INSUFFICIENT`; the system returns `UNCERTAIN` rather than
inventing labels. YOLO is an optional candidate interface only: no package or
weights were installed, and the NVIDIA GTX 1660 SUPER is present but this venv
has no PyTorch/CUDA runtime. The canonical CCTV schema is versioned as
`cctv_canonical_v1`, is not V4-compatible by assumption, and has no proven
viewpoint invariance.

The frozen V4 artifact hash remains
`af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd`.
The full test suite passes `165 passed`. No model was retrained, no raw data was
modified, and no physical camera was opened in this preparation run.

Latest preparation update: 2026-09-09 on `DESKTOP-ALH6CUB`. The CCTV
multi-person RTSP pipeline, manual-label collection gate, and multi-person
metrics are now prepared. The overnight system-preparation run added the
offline dashboard, room camera planner, multi-person readiness contract,
privacy/Git/instructor documentation and a versioned Acer bundle. It did not
train a model, open a webcam, change raw data, or overwrite V1/V2/V4. The
current full suite is `153 passed`; see
`reports/controlled_ai_robustness_baseline_th.md`,
`reports/overnight_system_prep_progress.md` and
`reports/overnight_system_prep_integrity.md`. The detailed sections below
retain their original verification dates.

## Current CCTV multi-person status

**CCTV_MULTIPERSON_PIPELINE_READY** — the primary generic runtime accepts
`max_people=1..4` (CCTV default 3), uses MediaPipe multi-pose, anonymous
`PERSON_XX` tracks, and maintains V4, geometry, and temporal state per person
through the existing `VideoSource` boundary. The latest 5-second max_people=3
RTSP run observed one pose in that scene, without manual ground truth; this is not a
person-count, IDF1, or real-room validation result.

**PENDING_TRUSTED_MULTIPERSON_LABELS** — no CCTV_V5 or hierarchical model was
trained. `scripts/collect_controlled_cctv.py` is prepared for attended,
human-labelled collection only.

Verification date: 2026-09-05. Host: `DESKTOP-ALH6CUB`.

## Controlled RGB robustness readiness

**READY_FOR_CONTROLLED_DATA_COLLECTION** — `scripts/run_rtsp_ai_lab.py` is a
single-person interactive wrapper around frozen V4, with configurable
`HIKVISION_MAIN`/`HIKVISION_SUB`, source health, reconnect state, calibration
context, landmarks, and geometry kept separate from AI posture. The collection
tool accepts only manual `TUP`/`TLB` selection and writes accepted 15-feature
rows under `data/controlled_rgb_v5/`; raw video is off by default.

No verified human-labelled samples exist yet, so no new V4 evaluation metrics
or V5 candidate were created. The evaluator deliberately refuses to produce
metrics until at least two participant groups and a valid controlled schema are
present. See `reports/controlled_ai_robustness_baseline_th.md` and
`reports/controlled_collection_protocol_th.md`.

## RTSP low-latency AI Lab status

**RTSP_REAL_BENCHMARK_PASS** — a short real-source benchmark completed for
`HIKVISION_MAIN` and `HIKVISION_SUB` without recording, collection, or
training. RTSP now uses a one-slot latest-frame reader; old frames are replaced
when AI is slower, so backlog does not grow. Median observed frame age was
0–16 ms across the short runs, with zero decode failures and zero reconnects.

The latest headless benchmark measured MAIN 2560x1440 at 20.39 source FPS,
14.12 AI FPS, and 23.9 ms median inference; SUB measured 640x360 at 20.18
source FPS, 14.57 AI FPS, and 62.8 ms median inference. A visible MAIN run
with the 1280x720 display copy reached 8.73 AI/display FPS and 20.9 ms median
render time. Pose/core acceptance was only 31.7%/56.6% in these short
sessions, so it is not an accuracy claim and must be repeated with a person in
the target seated view.

`RTSP_LOW_LATENCY` with a 960x540 inference copy is the conservative default
for MAIN. A 1280x720 run had lower inference time but zero pose detections in
that session, so it was not promoted on FPS alone. Details:
`reports/rtsp_ai_lab_low_latency_v2_th.md`.

## ANYA download and audit

Official Kaggle authentication was verified after installing Kaggle `2.2.4` in `.venv`. The official version-1 archive was downloaded and retained. Extracted files are under `datasets/raw/anya/official-download/`; original CSV names were preserved. Total local CSV size is `7,966,483,972` bytes; the retained ZIP is `3,607,095,951` bytes. CSV and archive hashes are in `metadata/anya_hashes.sha256` and `metadata/anya_archive.sha256`.

Full bounded-memory audit found four files with exactly 605,955 rows and 176 columns. The four files share one core row sequence and differ only in `Predicted_Label` outputs. No verified TRUE/FALSE ground-truth column is present. The downloaded release therefore cannot support an honest ergonomic classifier training run yet.

The follow-up source investigation classifies the current release as **B — PSEUDO_LABEL_ONLY**. The paper reports an expert-labeled 144,000-frame subset from the first 10 recordings, but the current Kaggle version-1 ZIP, local metadata, and checked author repository do not expose a row-joinable copy or cluster-to-TRUE/FALSE mapping. The paper-reported 432,000 fully labeled entries also do not match the locally observed 605,955 rows per file. Details are in `reports/anya_ground_truth_investigation.md`.

The four prediction columns were compared without training: 459,841 rows (75.887%) are unanimous, 146,114 (24.113%) are not unanimous, and 55,135 (9.099%) are 2-vs-2 ties. These are pseudo-label agreement diagnostics, not correctness metrics; see `reports/anya_pseudolabel_consensus.md`.

## Training status

**MULTIPOSTURE MODEL: READY** — HistGradientBoosting with the hybrid 126-feature schema. Grouped-CV macro F1 is `0.9891 ± 0.0060`; validation macro F1 `0.9910`; final subject-held-out test macro F1 `1.0000`. The final test contains only two subjects and must be interpreted cautiously.

**ANYA MODEL: NOT READY — TRAINING BLOCKED** — the current release is classified `B — PSEUDO_LABEL_ONLY`. It contains cluster IDs and algorithm-specific prediction outputs, but no verified supervised target. No `outputs/models/anya_v1/` model was exported, and no metric was fabricated.

## Evaluation status

MultiPosture evaluation remains subject-grouped: 9 train, 2 validation, and 2 final test subjects, with five-fold grouped CV inside train. ANYA file-level schema, quality, four-file comparison, grouping, and temporal-risk audits are complete. ANYA model comparison, leakage-safe metrics, calibration, and error analysis are correctly marked not run because the target is unverified.

## Offline image/video inference status

**OFFLINE IMAGE/VIDEO PIPELINE: READY AND TESTED** — the CLI accepts image files, image directories, and prerecorded video. ERGONOMIC POSTURE ENGINE V1 adds canonical landmarks, geometry indicators, configurable temporal risk states, JSONL event logs, session summaries, and annotated prerecorded-video output. No webcam source was executed. SitStraightNet media was used only for smoke/preprocessing validation, not as supervised ground truth.

**ANYA REAL-IMAGE ADAPTER: NOT VALIDATED** — the downloaded ANYA indexed landmark order, anatomy mapping, normalization formula, and target semantics cannot be proven equivalent to the MediaPipe/MultiPosture adapter. The existing adapter must not be connected to ANYA by positional guess.

## External RGB validation status

**EXTERNAL RGB MEDIA SEARCH: COMPLETED** — a small DriveNet v6 pack was downloaded from the official Mendeley Data record under `validation_data/external/`. The pack contains three unchanged 1920×1440 RGB MP4 files totaling `387,761,277` bytes: `SS_1.mp4` mapped to `TUP`, `LB_2.mp4` mapped to `TLB`, and `MB_1.mp4` kept `UNLABELED`. All local SHA-256 hashes match the source metadata and the source license is CC BY 4.0.

**EXTERNAL OFFLINE RGB SMOKE: COMPLETED** — MediaPipe detected a pose in `900/900` processed frames (300 per clip), but the strict 33-landmark adapter accepted `0/900` because lower-limb and some arm landmarks were below visibility 0.5 from the side/three-quarter vehicle view. No labeled accuracy or confidence claim was made. The annotated videos, telemetry, manifest, attribution, and machine-readable audit are retained under `outputs/validation/external/` and `validation_data/external/`.

**EXTERNAL RGB DOMAIN COMPATIBILITY: NOT PROVEN** — the 126 feature names/order are mechanically identical, but the external MediaPipe coordinate/projection and visibility behavior differ from the stored MultiPosture representation. See `reports/external_rgb_domain_shift.md`.

**ADAPTER ROBUSTNESS + UPPER-BODY V2: COMPLETED** — a four-policy study on the same 900 DriveNet frames found strict full-body acceptance `0/900`, while core-only, tiered, and aggregate-quality policies accepted `900/900`. The experimental V2 family requires only five core landmarks (`nose`, both shoulders, both hips), uses a separate 30-feature hybrid schema, and selected HistGradientBoosting with grouped-CV macro F1 `0.9925 ± 0.0087` and subject-held-out macro F1 `1.0000`. These same-source metrics are not real-camera generalization. The V2 DriveNet adapter accepted `900/900`, but prediction agreement is not treated as semantic validation: the mapped clips produced aggregate macro F1 `0.0000`, demonstrating that the coordinate/label semantics remain unresolved. Details are in `reports/adapter_robustness_v2.md`, `reports/landmark_visibility_analysis.md`, `reports/domain_shift_root_cause.md`, and `reports/multiposture_landmark_mapping_verified.md`.

**COORDINATE ADAPTER V3: PARTIALLY VALIDATED** — source forensics supports MediaPipe Pose Heavy World Landmarks in meters with a hip-midpoint origin, while the measured release contains 99 coordinate columns and does not expose the exact paper-model StandardScaler artifact. Hip-midpoint residuals are numerically near zero in the local CSV and MediaPipe world output; geometric scale normalization was not verified. The source-supported `adapter_v3_a_world_hip_centered` reduced V2 features outside the training range for more than half of external rows from `6/30` to `5/30`, but every external row still had at least one OOD feature. SS_1 remained `TLL` for `299/300` frames under the world candidate, so real-RGB semantic compatibility is not fully validated. No label-driven transform search was used. Details are in `reports/coordinate_adapter_v3.md`, `reports/multiposture_coordinate_forensics.md`, `reports/adapter_v3_distribution_comparison.md`, and `reports/ss_tup_to_tll_root_cause.md`.

## ERGONOMIC POSTURE ENGINE V1 status

- **MULTIPOSTURE DIRECTION MODEL: READY** — five direction states: Upright,
  Lean Forward, Lean Backward, Lean Left, Lean Right.
- **MULTIPOSTURE FEATURE ADAPTER: READY for exact schema/order; REAL-IMAGE
  SEMANTIC COMPATIBILITY NOT VALIDATED** — the 126 names/order match the
  exported model, but source coordinate-scale equivalence is not proven.
- **GEOMETRY ENGINE: READY** as deterministic indicators; head-forward,
  shoulder-tilt, and slouch remain **EXPERIMENTAL**.
- **TEMPORAL RISK ENGINE: READY** as configurable project logic with NORMAL,
  OBSERVE, WARNING, HIGH_DURATION_RISK, and POSE_LOST states.
- **OFFLINE IMAGE PIPELINE: READY** for file orchestration; the 756-image
  SitStraightNet smoke workload had 0 pose detections.
- **OFFLINE VIDEO PIPELINE: READY** for prerecorded input; annotated MP4,
  event log, and session report were generated in a 2-frame smoke run.
- **EXTERNAL RGB VALIDATION MEDIA: READY FOR POSE SMOKE ONLY** — DriveNet
  source videos were processed offline with a verified manifest and hashes;
  the strict adapter gate exposed a real side-view landmark-visibility
  limitation, so classifier accuracy was not reported.
- **UPPER-BODY V2: EXPERIMENTAL READY FOR CONTROLLED STUDY** — exported under
  `outputs/models/multiposture_upperbody_v2/`; V1 remains unchanged.
- **YOLO POSE: NOT YET BENCHMARKED** — interface only.

The 15-second warning and 60-second high-duration defaults are project settings,
not clinical or medical thresholds. See `reports/posture_engine_v1_design.md`,
`reports/geometry_feature_definitions.md`, and `reports/posture_engine_v1_results.md`.

## Live webcam status

**LIVE WEBCAM VALIDATION: REAL SESSION COMPLETED — CALIBRATION NEEDED**

A real user-guided Camera 0 session `20260907T155515Z` completed with 1,249 records over 103.617 seconds. It produced real telemetry, but it is not a claim of live posture accuracy. The detailed analysis is in `reports/live_hardware_calibration_analysis.md`.

**LIVE WEBCAM SOFTWARE: READY_FOR_HARDWARE_TEST** — `check_camera.bat` probes
only indices 0, 1, and 2 when explicitly started, and `run_webcam_demo.bat`
launches the timed validation/calibration prototype. The live application uses
the frozen RGB-native V4 two-class artifact, keeps geometry indicators separate
from AI posture, reuses `configs/posture_engine.yaml`, writes telemetry and
session summaries under `outputs/live/`, and exits cleanly with Q or ESC.

**LIVE WEBCAM HARDWARE: PASS** — Camera 0 opened, frames were read, and cleanup
completed. MediaPipe detection was `88.86%`, five-core acceptance was `66.30%`,
and mean measured FPS was `10.95`. Core quality and geometry remain
`NEEDS_CALIBRATION`; V4 TUP/TLB remains `NEEDS_VALIDATION`; temporal warning
remains `NEEDS_RETEST`.

The current automated suite, including hardware-free live-prototype, Acer
workflow, room-planner, multi-person and dashboard tests, passes `100/100`.

## Integrity and tests

The ANYA CSV hashes still match the pre-audit manifest, and the MultiPosture raw hash remains unchanged. The current full automated suite passes: `144 passed`. Raw files under `datasets/raw/` were not rewritten or deleted; V1/V2/V4 artifacts remain isolated. See `reports/overnight_system_prep_integrity.md` for the latest read-only hash snapshot.

## RGB-native V4 track (2026-09-05)

**RGB-NATIVE V4: PILOT TRAINED — TWO COMPATIBLE CLASSES ONLY.** The official DriveNet v6 per-file video subset was downloaded and SHA-256 verified for six videos: `LB_1..3` and `SS_1..3`, totaling `907,778,242` bytes. Each video contributed 300 deterministic samples, all 1,800 sampled rows had a detected pose and passed the five-landmark core gate. `SS → TUP` and `LB → TLB` were the only supervised mappings. `MB` and `RP` were not mapped because their names do not prove a sustained posture class.

The same pipeline was used for training and offline inference: MediaPipe World Landmarks → hip-midpoint translation → V4 feature family → classifier. The selected pilot is Logistic Regression with 15 normalized-world features. Grouped CV over participants 1/2 is `1.0000 ± 0.0000` macro F1; the predeclared participant-3 holdout is `1.0000` macro F1. These are source-domain two-class pilot results, not six-class office-camera generalization. The export is isolated under `outputs/models/rgb_native_v4/`.

An offline V4 smoke inference was completed on source video files. The existing DriveNet SS/LB re-test is retained as a **same-source overlap diagnostic**, not an independent cross-dataset test. CCTV Exam Monitor and SitStraightNet remain audit/smoke-only because the downloaded/repository snapshots lack a joinable per-image label sidecar; MVSP/LGCSPNet remains inaccessible for a verified local archive/license in this run.

The full automated suite now passes: `81 passed`. V1/V2/V3, ANYA, and raw dataset contents remain unchanged. The real Camera 0 session is diagnostic only, and YOLO was not installed.

## Remaining blocker and next experiment

Obtain the source-backed expert label mapping or a ground-truth sidecar that can be joined to these exact rows/clusters. A polite request is prepared in `reports/anya_author_request.md` and has not been sent. For V1/V2 deployment, repeat the real-camera session with the new neutral calibration, wider hip-safe framing, stable-ready gating, and step-8 reacquisition flow. Resolve the released MultiPosture 99-column versus paper 33-input/scaler provenance before any retraining. The current real-camera evidence is sufficient for calibration diagnosis but not real-image classifier validation. YOLO benchmarking remains deferred because the main blocker is geometry/input quality and coordinate semantics, not camera opening.

## RGB-native V5 dataset expansion (2026-09-05)

**V5 DATA READINESS GATE: NOT TRAINED.** The source-backed RGB inventory now
contains 452 traceable rows: 417 deterministic 1 fps samples from six existing
DriveNet videos and 35 images from the official MMD-MSD public subset. MediaPipe
World Landmark extraction and the frozen five-landmark V4-compatible feature
schema succeeded for all 452 rows. Verified project labels are TUP 244 and TLB
183; 25 MMD-MSD rows remain audit-only/ambiguous.

TLF, TLL, and TLR have zero eligible source-backed rows. Therefore V5 grouped
CV, holdout, leave-one-dataset-out, model comparison, and error analysis were
not run, and no V5 classifier artifact was created. V4 was not retrained or
overwritten. The 44-file source hash inventory is recorded in
`outputs/metrics/rgb_native_v5_source_hashes.json`; the sampling and gate
decisions are in `reports/rgb_native_v5.md`.

The full automated suite now passes: `89 passed`. The next valid step is to
obtain a licensed, traceable RGB source containing TLF/TLL/TLR (MVSP/LGCSPNet
remains a candidate pending license/archive verification) or collect an
explicitly labelled multi-participant office RGB dataset.

## Local RGB V1 collection system (2026-09-05)

The empty local collection layout is ready under `datasets/raw/local_rgb_v1/`:
10 anonymous participant folders × 2 sessions × 5 posture folders, with 100
planned recording slots. `participants.csv`, `sessions.csv`, and a separate
100-row master manifest are present. All media fields are blank and every slot
is `WAITING_FOR_VIDEO`; no participant recording or fabricated frame data was
created.

Filename/folder validation, SHA-256 inventory, MediaPipe World Landmark
pre-flight, frozen V4/V5 feature extraction, dataset summary, and participant
split planning tools are ready. The current local status is
`WAITING_FOR_LOCAL_RGB_DATA`; no V5 training was performed.

## Overnight autonomous run update (2026-09-06)

The overnight external-search and frozen V4 robustness run is complete. No new large raw archive was downloaded. All six verified DriveNet videos were decoded (`12,395` source frames); a deterministic dense `10 FPS` pass processed `4,134` frames, with MediaPipe pose detection and V4 five-core acceptance both `100%`. The nested `1/2/5 FPS` summaries also had `100%` pose/core acceptance and `0` `POSE_LOST` rows. At `5 FPS`, raw class switching across the six clips was `3`, reduced to `1` by the existing temporal smoother. These are same-source diagnostics, not independent external accuracy.

The frozen V4 feature audit found the largest participant-3 versus train-domain shifts in `nose_y_world_hip_centered_scaled` (absolute SMD `1.212`), both hip-y features (`0.932`), and right-shoulder-y (`0.925`). The frozen LogisticRegression model is most sensitive by coefficient magnitude to shoulder-x, nose-x, and shoulder-y features. This supports the existing conclusion that camera/domain semantics remain a limitation even when pose detection is excellent.

The external search result is **NO_VALID_EXTERNAL_5CLASS_DATA**. MVSP/LGCSPNet remains a manual-access candidate, but its Baidu archive, label join, participant/view metadata, and license were not verified; no bypass or automatic download was attempted. V5 remains untrained and ANYA remains untrained. A separate experimental offline V4 + geometry demo was prepared and smoke-tested on 60 frames; it is not a replacement model or a clinical score.

Current overnight artifacts are `reports/overnight_run_summary.md`, `reports/overnight_external_dataset_search.md`, `reports/overnight_performance_benchmark.md`, `reports/v4_feature_audit.md`, `reports/reproducibility_audit.md`, `reports/overnight_progress.md`, and `outputs/validation/overnight/`. The full test suite is now `65 passed`. Raw data and V1–V4 artifacts remain unchanged. The current recommendation is to keep MediaPipe, defer YOLO, and obtain a legally traceable all-five-class RGB source or collect multi-participant office RGB data before any V5 training.

## MVSP manual-import pipeline (2026-09-06)

The manual-import area is prepared at `datasets/incoming/mvsp/`; the extraction and processed metadata areas are `datasets/raw/mvsp/` and `datasets/processed/mvsp/`. The incoming folder currently contains no archive, so the effective gate is **WAITING_FOR_MANUAL_MVSP_ARCHIVE**. An existing `datasets/raw/mvsp/official-repository/` directory was preserved and not included in any fabricated import.

Prepared tools are `scripts/inspect_mvsp_archive.py`, `scripts/import_mvsp_dataset.py`, `scripts/audit_mvsp_labels.py`, and `scripts/preflight_mvsp_rgb.py`. The inspector lists supported ZIP/TAR contents, hashes the archive, estimates extracted size, and blocks unsafe paths/links. The importer preserves the original archive, rejects traversal and collisions, and writes an extracted-file inventory plus SHA-256 manifest without training V5. The label audit does not promote directory or filename names to labels; it requires source-backed evidence for `EXACT` or `COMPATIBLE` mappings. The preflight is bounded and uses the existing MediaPipe World Landmark/V4-compatible path only after RGB media exists.

Reports are `reports/mvsp_archive_inspection.md`, `reports/mvsp_inventory.md`, and `reports/mvsp_inventory_th.md`. The full suite after adding the import gate is `70 passed`. No MVSP archive was downloaded, extracted, or modified in this run.
