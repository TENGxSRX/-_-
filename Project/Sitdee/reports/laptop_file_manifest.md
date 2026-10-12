# Lightweight Acer file manifest

Audit date: 2026-09-07. Source of truth: the Desktop checkout at
`C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets`.

## Size audit

| Area | Measured size |
|---|---:|
| Entire working folder, including `.venv` | 14.43 GiB |
| `datasets/` | 13.35 GiB |
| `validation_data/` | 0.36 GiB |
| `outputs/` | 0.12 GiB |
| `.venv/` | 0.59 GiB |
| `src/` | 0.50 MiB |
| `scripts/` | 0.89 MiB |
| `tests/` | 0.28 MiB |
| `reports/` | 0.42 MiB |
| `metadata/` | 29.26 MiB |
| Proposed Acer bundle before ZIP compression | 29.93 MiB |

The generated code-only ZIP is 27.44 MiB.

The proposed Acer development checkout is far below the 5 GiB target. The
29.24 MiB pose model is the largest required runtime file.

## REQUIRED_ON_LAPTOP

- `src/`
- `scripts/` (including `laptop_health_check.py`, `run_laptop_tests.py`, and the offline inference tools)
- `configs/posture_engine.yaml`
- `configs/laptop_low_spec.yaml`
- `tests/`
- `README.md`, `pytest.ini`, `.gitignore`
- `requirements-laptop.txt`
- `setup_laptop_dev.ps1`, `start_laptop_dev.bat`
- `metadata/pose_landmarker_heavy.task`
- `outputs/models/rgb_native_v4/`
  - `model.joblib` (1,953 bytes)
  - `feature_config.json` (1,215 bytes)
  - `labels.json` (134 bytes)
  - `metadata.json` (6,451 bytes)
- `fixtures/` for sanitized UI/session-summary work
- the four laptop workflow reports

No raw video is required to load V4 or develop the core geometry/temporal
logic.

## OPTIONAL

- Selected small Markdown reports for context.
- Small provenance JSON/text files under `metadata/` when a report needs them.
- A copied saved session summary only when a UI needs a real example. The
  existing `outputs/live/` logs remain Desktop-only by default.
- A future small, provenance-recorded media fixture only after its license and
  redistribution terms are confirmed. None is copied in the current bundle.

## KEEP_DESKTOP_ONLY

- `datasets/raw/`, including ANYA CSVs, the ANYA archive, MultiPosture source
  CSV, MVSP material, SitPose/SitStraightNet, local participant recordings,
  and DriveNet source videos.
- `datasets/processed/` and `datasets/incoming/`.
- `validation_data/`, including DriveNet clips and external validation media.
- full `outputs/validation/`, `outputs/live/` logs/videos/session reports,
  `outputs/metrics/`, benchmark outputs, and training artifacts.
- `outputs/models/multiposture_v1/`,
  `outputs/models/multiposture_upperbody_v2/`, and future training models.
- `.venv/`, caches, `__pycache__/`, and temporary benchmark directories.

These exclusions are operational rules, not data deletion. The Desktop keeps
the primary copy and remains the only machine for full training and heavy
media processing.

## Sync rule

Synchronize code, configuration, tests, documentation, the frozen V4 runtime
artifact, and the pose model only. Never use `git add .` until the ignore rules
and the staged file list have been reviewed. No Git remote is configured in
the current checkout, so no push was performed.
