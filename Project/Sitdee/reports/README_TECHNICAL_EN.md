# Office posture ML workspace

This workspace contains an auditable sitting-posture classification prototype. It uses source labels for posture research only; it is not a medical diagnostic or Office Syndrome detector.

Directory layout:

```text
datasets/raw/       Original downloaded datasets; excluded from version control
metadata/            Source metadata and MediaPipe task asset record
src/posture_ai/      Data validation, features, training, evaluation, inference
outputs/             Models, metrics, figures, and prediction logs
reports/             Dataset, training, compatibility, and status reports
scripts/             Reproducible inspection helpers
tests/               Unit and offline integration tests
```

## Environment

Create/use the local virtual environment in `.venv`. The verified environment and hardware are recorded in `reports/environment.md`.

## MultiPosture training

From this directory:

```powershell
& '.\.venv\Scripts\python.exe' -m src.posture_ai.train --dataset multiposture --seed 42 --cv-splits 5
```

The command performs subject-grouped holdout and cross-validation, compares available models and feature tracks, exports `outputs/models/multiposture_v1/`, and writes metrics/reports. XGBoost and LightGBM are optional and are skipped when unavailable.

## Offline inference

Use a prerecorded image, image directory, or video:

```powershell
& '.\.venv\Scripts\python.exe' -m src.posture_ai.inference --source <path> --model outputs/models/multiposture_v1 --pose-model metadata/pose_landmarker_heavy.task --output outputs/predictions/inference.jsonl
```

The offline runner never opens a webcam. The separate live prototype has
already verified camera index 0 on the current hardware, but live validation
still remains a limited pilot; exact cross-domain normalization equivalence is
not yet validated.

Run the ERGONOMIC POSTURE ENGINE V1 on offline media:

```powershell
& '.\.venv\Scripts\python.exe' -m src.posture_ai.inference --engine-v1 --source <image.jpg>
& '.\.venv\Scripts\python.exe' -m src.posture_ai.inference --engine-v1 --source <folder> --max-frames 100
& '.\.venv\Scripts\python.exe' -m src.posture_ai.inference --engine-v1 --source <video.mp4>
```

The V1 runner writes event logs under `outputs/logs/`, session summaries under
`outputs/session_reports/`, performance JSON under `outputs/metrics/`, and an
annotated video under `outputs/videos/` for video input. Its 15-second warning
and 60-second high-duration defaults are project settings, not medical rules.

## RGB-native V4 pilot

V4 uses one RGB-native contract for training and offline inference:
MediaPipe World Landmarks → hip-midpoint translation → V4 features →
classifier. The current official DriveNet pilot is deliberately limited to
the two compatible mappings `SS -> TUP` and `LB -> TLB`; `MB` and `RP` remain
unmapped. It is exported separately under `outputs/models/rgb_native_v4/`.

Train/rebuild the pilot from the six downloaded, hash-verified source videos:

```powershell
& '.\.venv\Scripts\python.exe' scripts\train_rgb_native_v4.py --samples-per-video 300
```

Run the same pipeline on offline media:

```powershell
& '.\.venv\Scripts\python.exe' scripts\infer_rgb_native_v4.py --source <image.jpg>
& '.\.venv\Scripts\python.exe' scripts\infer_rgb_native_v4.py --source <folder> --max-frames 100
& '.\.venv\Scripts\python.exe' scripts\infer_rgb_native_v4.py --source <video.mp4>
```

This V4 pilot is not a six-class replacement for MultiPosture and does not
claim office-camera or medical performance.

## Tests

The clean test command used for this Windows environment is:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q --basetemp .pytest-runtime
```

Tests cover MultiPosture and ANYA schema/feature ordering, grouped splits, MediaPipe adapter behavior, missing visibility, model-backed video-path execution, temporal smoothing, chunked ANYA loading, conservative target detection, raw hashing, schema-mismatch rejection, and external validation-manifest ontology. The latest clean verification run passed 29 tests.

The latest adapter robustness/V2, RGB-native V4, RGB-native V5 contract, local
collection, overnight benchmark, live hardware UX verification, lightweight
Acer workflow verification, room planner, multi-person contract, and offline
dashboard verification extends this to 100 tests. The RTSP source
abstraction, mock health/reconnect tests, bandwidth estimator, and source-health
dashboard compatibility extended the clean suite to 109 tests. The integrated
multi-person/seat-aware contracts, merge tests, and dashboard compatibility
now extend the full suite to 133 tests. The controlled RGB robustness
collection and frozen-V4 evaluation gates extend the full suite to 144 tests.
The CCTV multi-person contracts, manual-label assignment rules, and five-class
controlled collection path extend the current suite to 153 tests.
The experimental
upper-body model family is exported under `outputs/models/multiposture_upperbody_v2/`
and uses only five core landmarks for seated-posture robustness. The
coordinate-adapter V3 study supports a world-landmark/hip-centered candidate,
but real-image coordinate semantics remain only partially validated because
the released CSV and paper-model dimensionality/scaler provenance do not fully
agree. The overnight frozen V4 benchmark decoded all 12,395 frames from the six
verified DriveNet videos and pose-processed 4,134 dense 10-FPS frames. Pose and
five-core acceptance were 100%; no independent external accuracy claim was
made. See `reports/overnight_run_summary.md`,
`reports/overnight_performance_benchmark.md`, `reports/v4_feature_audit.md`,
and `reports/reproducibility_audit.md`.

## Acer lightweight development

The Desktop remains the source of truth for all raw datasets, long videos,
training outputs, and heavy experiments. A code-only bundle can be prepared
with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\create_laptop_bundle.ps1
```

The command now creates the new versioned `outputs/laptop_dev_bundle_v5.zip`
and leaves all earlier bundles untouched. The bundle
contains the frozen V4 runtime, MediaPipe pose model, source code,
tests, configs, RTSP profile templates, reports, and a sanitized `TEST_FIXTURE`; it contains no
dataset, validation video, live recording, cache, or `.venv`. On the Acer run
`setup_laptop_dev.ps1` once, then `start_laptop_dev.bat`. Use
`scripts\run_laptop_tests.py` for the lightweight test command. For older CPUs
use `scripts\run_webcam_demo.py --preset LOW_SPEC --source 0` only when a
camera is explicitly available; the preset processes every third frame and
marks skipped frames instead of pretending they were inferred.

See `reports/laptop_file_manifest_th.md`, `reports/acer_python_setup_th.md`,
`reports/two_machine_workflow_th.md`, and `reports/acer_daily_workflow_th.md`.

## Offline dashboard and room camera planning

The saved-session dashboard does not open a camera. It reads a session summary
and optional JSONL event log, keeps `AI POSTURE` separate from experimental
`GEOMETRY INDICATORS`, and shows `NOT RECORDED IN THIS SESSION` when an older
session lacks a field:

```powershell
& '.\.venv\Scripts\python.exe' scripts\dashboard_demo.py --use-fixture --output outputs\demo\test_fixture_dashboard.html
& '.\.venv\Scripts\python.exe' scripts\dashboard_demo.py --output outputs\demo\latest_session_dashboard.html
```

The room planner accepts only an explicit JSON/YAML room specification. The
included `fixtures/test_room_spec.json` is software-test data marked
`TEST_ROOM_ONLY`; it is not a real-room measurement and does not determine the
camera count:

```powershell
& '.\.venv\Scripts\python.exe' scripts\room_camera_planner.py `
  --spec path\to\room.json `
  --output-json outputs\room_plan\plan.json `
  --output-svg outputs\room_plan\plan.svg
```

All coverage results are labeled `GEOMETRIC PLANNING ESTIMATE`. The planner
must be reviewed with the real floor plan, camera FOV, obstacles and a later
visibility test for nose, shoulders and hips.

For a small prerecorded smoke check of the DESKTOP/BALANCED/LOW_SPEC frame
scheduling, run:

```powershell
& '.\.venv\Scripts\python.exe' scripts\benchmark_performance_presets.py
```

This accepts a video path only, never a camera index, and does not provide an
Acer-specific FPS or accuracy claim.

## Future Hikvision RTSP source

The Desktop has a secret-safe, source-agnostic preparation path for a USB
index, prerecorded file, direct RTSP URL, or the configurable Hikvision
profiles `HIKVISION_MAIN` (`/Streaming/Channels/101`) and `HIKVISION_SUB`
(`/Streaming/Channels/102`). The existing V4 model remains frozen and the
physical camera is not opened by setup or tests.

Create a local ignored `.env` from `.env.example` only when the camera is
available. Supply `POSTURE_CAMERA_HOST`, `POSTURE_CAMERA_USER`, and
`POSTURE_CAMERA_PASSWORD` (or a complete `POSTURE_RTSP_URL`) locally; never
commit or print credentials. The safe first check is:

```powershell
& '.\.venv\Scripts\python.exe' scripts\check_rtsp_source.py --source rtsp --profile HIKVISION_MAIN --frames 30 --seconds 10
```

This diagnostic does not run AI, record video, save screenshots, start a
listener, or change network settings. After the source health check passes,
the prepared V4 runtime can be started explicitly with:

```powershell
& '.\.venv\Scripts\python.exe' scripts\run_rtsp_posture.py --source rtsp --profile HIKVISION_MAIN --preset RTSP_DESKTOP
```

`RTSP_DESKTOP` keeps the source frame for display/debug and processes every
frame. `RTSP_BALANCED` uses a configurable inference copy and frame interval.
Both use real timestamps, capped reconnects, bounded latest-frame support, and
separate source-health states. Actual resolution, codec, bitrate, latency,
multi-person behavior, and camera reachability remain untested until the real
Hikvision stream is available. See `reports/rtsp_ai_server_preparation_th.md`,
`reports/hikvision_network_deployment_th.md`, and
`reports/hikvision_camera_test_checklist_th.md`.

## Person-centric multi-person and optional seat-aware prototype

The Acer handoff is now integrated on Desktop through the existing `VideoSource`
boundary. The Desktop V4 artifact remains the only model source of truth:
`TUP` and `TLB` are AI classes; Forward/Left/Right remain geometry indicators.

Generic person-centric mode is the recommended primary runtime. It supports
`--max-people 1` through `--max-people 4` with anonymous `PERSON_XX` tracks,
experimental SITTING/STANDING/UNKNOWN activity gating, and independent V4,
geometry, temporal, and lifecycle state. Standing pauses sitting analysis and
UNKNOWN pauses risk accumulation. Seat mode remains optional, uses
`--tracking-mode seat`, and **requires** an explicit real seat-layout YAML;
the example layouts are `TEST_ROOM_ONLY` and must not be treated as room data.

Example command shapes (do not run until a source is intentionally available):

```powershell
& '.\.venv\Scripts\python.exe' scripts\run_multiperson_demo.py --source rtsp --profile HIKVISION_MAIN --max-people 3 --tracking-mode generic --no-display
& '.\.venv\Scripts\python.exe' scripts\run_multiperson_demo.py --source rtsp --profile HIKVISION_MAIN --max-people 3 --tracking-mode seat --seat-layout path\to\approved_seat_layout.yaml
```

The multi-person launcher now uses `VideoSource`; it does not create a second
RTSP/OpenCV capture path. Telemetry keeps source health separate and redacts
credential-bearing URLs. A short real Hikvision source run is now complete, but
person-count, ID-switch, activity, and posture accuracy remain unvalidated.

For an attended controlled CCTV collection session, the operator must select
an anonymous `PERSON_XX` track and provide the label manually. Keys `1..5` map
to `TUP/TLB/TLF/TLL/TLR`; `0=IGNORE` and `U=UNCERTAIN` do not create samples.
Raw video is off by default, V4 predictions are diagnostic only, and no model
is trained by this command:

```powershell
& '.\.venv\Scripts\python.exe' scripts\collect_controlled_cctv.py `
  --source rtsp --profile HIKVISION_MAIN --max-people 3 `
  --expected-people-count 3 --preset RTSP_CCTV `
  --subject-id P001 --clip-id CCTV_STAGE1_01
```

Rows, rejection reasons, session metadata, and safe telemetry are written
under `data/controlled_cctv_v2/`. Use this only when a human operator is
present; do not treat the short RTSP benchmark as labelled data.

For future human-labelled activity data, use the prepared attended collector.
It stores SITTING/STANDING labels separately from experimental gate/V4 output
and keeps raw video disabled by default:

```powershell
& '.\.venv\Scripts\python.exe' scripts\collect_controlled_activity.py `
  --source rtsp --profile HIKVISION_MAIN --max-people 4 `
  --expected-people-count 3 --subject-id P001 --clip-id ACTIVITY_STAGE1_01
```

## Controlled RGB robustness lab

The AI Lab is a single-person interactive wrapper around the frozen V4 path.
It uses `HIKVISION_MAIN` and the `RTSP_LOW_LATENCY` preset by default and also supports `HIKVISION_SUB`; it shows
AI posture/confidence, the five core landmarks, source health, calibration
context, and geometry indicators separately. It does not save raw video by
default and does not train a model:

```powershell
& '.\.venv\Scripts\python.exe' scripts\run_rtsp_ai_lab.py --source rtsp --profile HIKVISION_MAIN --preset RTSP_LOW_LATENCY --max-people 1
```

For controlled human labels, use a new anonymous participant/clip code. In the
preview, press `1` for TUP, `2` for TLB, `SPACE` to start/pause accepted-frame
capture, and `Q`/`ESC` to finish:

```powershell
& '.\.venv\Scripts\python.exe' scripts\collect_controlled_posture.py --source rtsp --profile HIKVISION_MAIN --subject-id P001 --clip-id P001_TUP_01
```

Accepted feature rows and transparent rejection reasons are written under
`data/controlled_rgb_v5/`. `human_label` is never copied from the V4 result;
the V4 result is stored in a separate field. A frozen V4 baseline command is
ready for after at least two participant groups have been collected:

```powershell
& '.\.venv\Scripts\python.exe' scripts\evaluate_controlled_v4.py --data-root data\controlled_rgb_v5
```

The evaluator refuses to create metrics while verified human-labelled data is
missing. V5 remains a later, separately audited candidate and is not trained by
the lab. The measured low-latency behavior is documented in
`reports/rtsp_ai_lab_low_latency_v2_th.md`.

## Local RGB V1 collection system

The empty collection layout is ready under `datasets/raw/local_rgb_v1/` for 10
anonymous participants, 2 sessions each, and five postures (`TUP`, `TLF`, `TLB`,
`TLL`, `TLR`): 100 planned recording slots. Do not put participant videos in
Git. Use the Thai guide `reports/v5/local_rgb_collection_guide_th.md` and the
phone checklist `reports/v5/recording_checklist_th.md`.

Prepare or restore the empty layout:

```powershell
& '.\.venv\Scripts\python.exe' scripts/prepare_local_rgb_v1.py
```

After recordings are placed, run these tools in order:

```powershell
& '.\.venv\Scripts\python.exe' scripts/local_collection_status.py
& '.\.venv\Scripts\python.exe' scripts/validate_local_rgb_dataset.py
& '.\.venv\Scripts\python.exe' scripts/preflight_local_rgb_pose.py
& '.\.venv\Scripts\python.exe' scripts/extract_local_rgb_features.py
& '.\.venv\Scripts\python.exe' scripts/summarize_local_rgb_dataset.py
& '.\.venv\Scripts\python.exe' scripts/plan_local_v5_split.py
```

The tools do not train V5. They use the frozen RGB-native V4/V5 feature
definitions, keep raw videos unchanged, record SHA-256 hashes, and plan
participant-independent splits. Current status is `WAITING_FOR_LOCAL_RGB_DATA`.

## External RGB validation pack

The small external pack under `validation_data/external/` contains three
unchanged DriveNet v6 RGB videos from the official Mendeley Data record. Only
source classes with a safe semantic mapping are labeled (`SS -> TUP`,
`LB -> TLB`); `MB` is smoke-only. The first 300 frames per clip were processed
offline. MediaPipe detected a pose in all processed frames, but the strict
33-landmark visibility gate rejected the external side-view frames. The V2
five-landmark core gate accepts them, but no external classifier accuracy is
claimed because source coordinate and label semantics remain unvalidated. See
`reports/external_validation_media_search.md`,
`reports/external_rgb_domain_shift.md`, and `reports/adapter_robustness_v2.md`.

## Dataset status

MultiPosture is the complete local supervised baseline. ANYA is downloaded and fully audited, but its current release is classified `B — PSEUDO_LABEL_ONLY`: the four files contain shared observations plus algorithm-specific `Predicted_Label` outputs and no verified ground-truth target, so ANYA training is blocked pending the source label mapping. The paper-reported expert subset is not currently row-joinable to the release. See `reports/anya_ground_truth_investigation.md`, `reports/anya_pseudolabel_consensus.md`, `reports/anya_schema_audit.md`, and `reports/anya_file_level_audit.md`.

## Status reports

Start with `reports/final_project_status.md`, `reports/posture_engine_v1_design.md`, and `reports/posture_engine_v1_results.md`.

## MVSP manual import gate

The MVSP/LGCSPNet manual-import pipeline is prepared, but the incoming folder
is currently empty and no archive was downloaded automatically. Place the
original archive, without renaming or extracting it, under
`datasets/incoming/mvsp/`. Supported archive types are `.zip`, `.tar`, `.tgz`,
`.tar.gz`, `.tar.bz2`, and `.tar.xz`.

Run these checks from the project root after manual download:

```powershell
& '.\.venv\Scripts\python.exe' scripts\inspect_mvsp_archive.py
& '.\.venv\Scripts\python.exe' scripts\import_mvsp_dataset.py
& '.\.venv\Scripts\python.exe' scripts\audit_mvsp_labels.py
& '.\.venv\Scripts\python.exe' scripts\preflight_mvsp_rgb.py
```

The importer preserves the archive, refuses traversal/link entries and file
collisions, extracts only to `datasets/raw/mvsp/`, and writes inventories and
hashes without training V5. See `datasets/incoming/mvsp/README_th.md` and
`reports/mvsp_inventory.md`.

## Live Webcam Prototype V1

The hardware-ready live prototype uses the frozen RGB-native V4 pilot only:
`TUP -> UPRIGHT` and `TLB -> LEAN BACKWARD`. Forward, Left, and Right are
camera-relative geometry indicators, not V4 AI classes. No webcam is opened
until the user explicitly starts the live command.

Launchers:

```text
check_camera.bat
run_webcam_demo.bat
```

The first script checks camera indices 0, 1, and 2. The second starts the
timed beginner validation display for camera 0. Validation first waits for
stable core visibility, collects a short runtime-only neutral baseline, and
asks for `SPACE` before each step. Input loss pauses active-step timing. For
detailed calibration values, use `run_webcam_calibration.bat`. If discovery
reports camera 1 or 2, run:

```powershell
& '.\.venv\Scripts\python.exe' scripts\run_webcam_demo.py --source 1 --validation --calibration
```

Telemetry is written to `outputs/live/logs/`, session summaries to
`outputs/live/session_reports/`, and recordings are optional under
`outputs/live/videos/`. See `reports/webcam_setup_guide_th.md`,
`reports/live_webcam_v1_design.md`, and
`reports/live_webcam_validation_protocol_th.md`. The latest real-camera
calibration analysis is in `reports/live_hardware_calibration_analysis.md`.
