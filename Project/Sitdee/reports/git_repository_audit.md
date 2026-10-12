# Git repository and sync audit

Audit date: 2026-09-07.

## Finding

- Project root is **not** a Git repository: no `C:\...\office-posture-datasets\.git`.
- Its parent `AnyaTrainModel` is also not a Git repository.
- Current branch: **not available**.
- Working-tree status: **not available**.
- Configured remotes / GitHub origin: **none can be verified**.
- Tracked/untracked file list: **not available**, because there is no repository.
- No push, reset, clean, or history-changing operation was performed.

## File-level observations

The working folder contains large local material, including `datasets/` at
13.35 GiB, `validation_data/` at 0.36 GiB, and a 0.59 GiB `.venv`. These are
not suitable for laptop sync. The largest source data are the ANYA CSV/archive
files and source videos. They remain in place.

The original ignore file already excluded raw CSV/archive patterns and Python
environments. It also had broad `*.joblib` and `models/` rules that would hide
the small frozen V4 runtime artifact. The ignore file was updated to exclude
large/local outputs while explicitly allowing `outputs/models/rgb_native_v4/`.
No existing tracked file was removed because no Git repository exists.

## Safe next step

If the user wants Git sync, create or connect a private repository manually on
Desktop, stage only the allow-listed code/config/test/report files, inspect
`git diff --cached --stat` and `git status --short`, and only then push. Do not
commit raw datasets, credentials, local recordings, temporary logs, or large
validation outputs. Git LFS is not needed for the current code-only bundle;
the pose model is about 29 MiB and can instead be distributed with the bundle
if repository policy does not allow binary model files.
