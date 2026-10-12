# Overnight performance benchmark

## Contract

This report evaluates the frozen RGB-native V4 artifact on all six verified DriveNet videos. It is a same-source robustness/performance benchmark, not an independent external accuracy claim. All source frames were decoded; MediaPipe and V4 inference were run on a deterministic dense 10 FPS subsequence. 1/2/5 FPS rows are nested deterministic subsequences of that same pass.

| Rate | Selected frames | Pose detected | Core accepted | Pose rate | Core acceptance | Raw switches | Smoothed switches | POSE_LOST |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 417 | 417 | 417 | 100.00% | 100.00% | 1 | 1 | 0 |
| 2 | 830 | 830 | 830 | 100.00% | 100.00% | 1 | 1 | 0 |
| 5 | 2069 | 2069 | 2069 | 100.00% | 100.00% | 3 | 1 | 0 |
| 10 | 4134 | 4134 | 4134 | 100.00% | 100.00% | 3 | 1 | 0 |

## Per-video 5 FPS diagnostics

| Video | Participant | Source/mapped label | Decoded | Selected | Pose | Accepted | Prediction distribution | Mean confidence | OOD features >50% |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|
| `LB_1` | 1 | LB / TLB | 2131 | 356 | 356 | 356 | `{'TLB': 356}` | 0.9990 | 0 |
| `LB_2` | 2 | LB / TLB | 1058 | 177 | 177 | 177 | `{'TLB': 177}` | 0.9956 | 0 |
| `LB_3` | 3 | LB / TLB | 2226 | 371 | 371 | 371 | `{'TLB': 371}` | 0.9988 | 0 |
| `SS_1` | 1 | SS / TUP | 2131 | 356 | 356 | 356 | `{'TUP': 356}` | 0.9996 | 0 |
| `SS_2` | 2 | SS / TUP | 2126 | 355 | 355 | 355 | `{'TLB': 2, 'TUP': 353}` | 0.9916 | 0 |
| `SS_3` | 3 | SS / TUP | 2723 | 454 | 454 | 454 | `{'TUP': 454}` | 1.0000 | 3 |

## Performance and limits

- Dense pass wall time: `500.21` seconds; decoded throughput: `24.78` frames/s.
- RSS: `183.3` MB → `222.9` MB (delta `39.6` MB).
- Video labels are from the same DriveNet source used to train V4; therefore prediction counts are diagnostics and are not reported as independent accuracy.
- No webcam source 0 was opened. No YOLO package was installed. V1/V2/V3/V4 model artifacts were not overwritten.
- Feature OOD detail and train-vs-held-out feature statistics are in `reports/v4_feature_audit.md` and the JSON artifact.
