# Offline performance measurements

Verification date: 2026-09-05

These are local engineering measurements, not accuracy results.

## SitStraightNet image workload

Workload: 756 unlabeled JPEGs from the checked SitStraightNet repository
snapshot.

| Metric | Result |
|---|---:|
| Complete pipeline throughput | 59.89 images/s |
| Mean pose extraction | 14.32 ms/image |
| Adapter/classifier latency | not measured; 0 poses detected |
| Pose detections | 0 / 756 |
| Process RSS at start/end | 189.43 / 213.00 MB |
| RSS delta | +23.56 MB |

## Prerecorded video smoke workload

Workload: generated 64×64 blank AVI, 2 frames at 5 FPS. This verifies the video
reader, timestamp path, temporal engine, annotated writer, event log, and
session report without a webcam.

| Metric | Result |
|---|---:|
| Frames processed | 2 |
| Input FPS | 5.0 |
| Complete pipeline throughput | 4.59 frames/s |
| Mean pose extraction | 13.51 ms/frame |
| Pose detections | 0 / 2 |
| Process RSS at start/end | 189.25 / 223.74 MB |
| RSS delta | +34.49 MB |
| Annotated output | `outputs/videos/engine_blank_analyzed.mp4` |

The tiny blank clip includes file I/O and MediaPipe initialization overhead, so
it is not a representative production throughput benchmark. A future benchmark
should use a representative prerecorded sitting-person video and record peak
RSS/memory separately.
