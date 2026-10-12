# Overnight progress checkpoint

## [START]

- Project root verified: `office-posture-datasets`.
- Frozen scope: preserve MultiPosture V1/V2/V3/V4, do not train V5 or ANYA, do not open webcam source `0`, do not install YOLO, do not modify raw data.
- C: free space before the run: approximately 19.97 GB.

## [EXTERNAL SEARCH]

- Completed official-source search and screening.
- Result: `NO_VALID_EXTERNAL_5CLASS_DATA`.
- MVSP/LGCSPNet remains a manual-access candidate because the official README points to a Baidu archive; no bypass or automatic download was attempted.
- No new raw archive was downloaded.

## [DRIVENET FULL BENCHMARK]

- Complete: all six verified LB/SS videos decoded end-to-end; deterministic dense 10 FPS MediaPipe/V4 pass.
- Source frames: 12,395; dense processed frames: 4,134; pose/core acceptance: 100%.
- Nested 1/2/5 FPS sensitivity summaries: complete; no POSE_LOST rows.

## [FEATURE AUDIT]

- Frozen V4 training/holdout feature audit and LogisticRegression coefficient audit prepared.

## [V5 DECISION]

- Gate remains closed: no valid external five-class RGB source was obtained; TLF/TLL/TLR remain missing. No V5 training.

## [TESTS]

- New overnight-contract tests added; full suite: 65 passed in 9.68 seconds.

## [FINAL]

- Complete: reports, hash checks, hybrid offline demo smoke test, and final status matrix written.
