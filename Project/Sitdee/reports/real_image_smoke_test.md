# Real-image domain smoke test

Verification date: 2026-09-05

## Scope

The local SitStraightNet JPEG snapshot was used only as an unlabeled domain
smoke test. No visual labels were inferred and no classification accuracy was
calculated.

Input: `datasets/raw/sitstraightnet/official-repository/dataset/`

- 756 JPEG images were processed.
- MediaPipe Pose was run with the pinned local task asset and the project
  visibility setting.
- The V1 pipeline created a structured event log and session summary.

## Measured result

| Measurement | Result |
|---|---:|
| Images attempted | 756 |
| MediaPipe pose detections | 0 / 756 |
| Pose detection success rate | 0.000% |
| Accepted poses after visibility/confidence | 0 / 756 |
| Adapter executions | 0 |
| Mean pose-extraction time | 14.32 ms/image |
| Complete pipeline throughput | 47.52 images/s |
| Mean landmark visibility | not available because no pose was returned |
| Missing-landmark frequency | not available because no pose was returned |
| Process RSS at start/end | 189.43 / 213.00 MB |
| RSS delta | +23.56 MB |

The images are very dark/silhouette-like in the checked snapshot. The zero
detection result is a limitation of this smoke workload and does not prove that
the MediaPipe adapter or the direction classifier works on future camera input.
It also does not prove classifier failure in general.

## Decision

- Technical file traversal and offline inference orchestration: **tested**.
- Real-image pose extraction on this SitStraightNet snapshot: **0 detections**.
- Real-image MultiPosture adapter semantic compatibility: **NOT VALIDATED**.
- SitStraightNet classification accuracy: **not measured by design**.

The generated telemetry is in `outputs/logs/dataset_events.jsonl`,
`outputs/session_reports/dataset_summary.json`, and
`outputs/metrics/dataset_performance.json`.
