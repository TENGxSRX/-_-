# ERGONOMIC POSTURE ENGINE V1 results

Verification date: 2026-09-05

## Implementation result

The V1 offline architecture is implemented around the existing supervised
MultiPosture direction model. It now has:

- canonical 33-landmark representation with x/y/z/visibility;
- strict MediaPipe-to-MultiPosture feature adapter with exact 126-feature order;
- deterministic geometry module and explicitly experimental indicators;
- configurable probability smoothing, discrete majority voting, minimum
  direction persistence, recovery, cooldown, and pose-lost state;
- structured JSONL events, session summaries, and annotated prerecorded-video
  output;
- a backend interface that can receive a future YOLO pose backend without
  rewriting the risk engine.

## Validated results

- Synthetic/unit geometry and temporal-state tests passed.
- Image input and video input integration tests passed.
- A 2-frame prerecorded blank-video smoke run wrote an annotated MP4, event log,
  and session summary.
- SitStraightNet's 756 unlabeled images were processed at 59.89 images/s, but
  MediaPipe returned 0 poses for this dark/silhouette workload. No accuracy was
  calculated.
- Full test suite: **23 passed**.

## Current statuses

| Component | Status |
|---|---|
| MultiPosture direction model | READY |
| MultiPosture feature adapter schema | READY for exact schema/order checks |
| MultiPosture real-image semantic adapter | NOT VALIDATED |
| Geometry engine | READY as deterministic indicators |
| Head-forward, shoulder-tilt, slouch indicators | EXPERIMENTAL |
| Temporal risk engine | READY as configurable project logic |
| Offline image pipeline | READY for file orchestration; real-image pose success remains workload-dependent |
| Offline video pipeline | READY for prerecorded-media orchestration and annotated output |
| ANYA ergonomic model | BLOCKED — NO VERIFIED GROUND TRUTH |
| YOLO Pose | NOT YET BENCHMARKED |
| Live webcam | NOT TESTED — HARDWARE UNAVAILABLE |

## Scientific boundary

The MultiPosture result supports the five direction states within its recorded
dataset and grouped evaluation. It does not validate clinical posture rules,
new-camera generalization, or unseen-person real-world performance. The
configured 15-second warning and 60-second high-duration values are product
settings, not medical thresholds.
