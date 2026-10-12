# Environment report

Verification date: 2026-09-05 (Asia/Bangkok)

- Host: `DESKTOP-ALH6CUB`
- OS: Windows 11 Home, build `10.0.26200`
- CPU: Intel Core i3-10105, 4 cores / 8 logical processors
- RAM: 15.92 GB
- Available RAM at the final pre-download check: approximately 1.26 GB; large-file work must use bounded chunks and must not load full CSVs into memory.
- GPU: NVIDIA GeForce GTX 1660 SUPER, driver 591.86, 6 GB VRAM
- `nvidia-smi`: available; this benchmark used scikit-learn CPU estimators and did not claim GPU training.
- Free `C:` space before download: approximately 38.08 GB; after the retained archive and four extracted ANYA CSVs: approximately 25.51 GB
- Python: 3.12.10 in `.venv`

Key packages:

| Package | Version |
|---|---:|
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| scipy | 1.18.1 |
| joblib | 1.6.0 |
| matplotlib | 3.11.1 |
| opencv-python | 5.0.0 |
| mediapipe | 1.0.1 |
| psutil | 7.2.2 |

Kaggle client: `2.2.4` in `.venv`; official authenticated file-list verification succeeded without exposing credentials.

Optional XGBoost and LightGBM were not installed and were skipped from the benchmark. The MediaPipe Tasks Pose Landmarker Heavy asset is recorded in `metadata/pose_landmarker_heavy.task` with SHA-256 `64437AF838A65D18E5BA7A0D39B465540069BC8AAE8308DE3E318AAD31FCBC7B`.
