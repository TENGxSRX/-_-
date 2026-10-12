# RGB-native V4 pose-quality audit

The audit uses the same MediaPipe Pose Landmarker task asset and World Landmark extraction path that feeds V4 features. Core acceptance requires finite coordinates and image-landmark visibility ≥ 0.5 for nose, both shoulders, and both hips.

The audit workload is the six downloaded DriveNet videos whose source labels map to the supervised pilot (`LB_1..3`, `SS_1..3`). It samples 300 frames by deterministic linspace per video, so the 1,800 rows are not random frame duplicates. MB/RP were not included in this supervised workload because their source names are ambiguous for the target ontology.

| Source label | Sampled | Pose detected | Core accepted | Acceptance |
|---|---:|---:|---:|---:|
| LB | 900 | 900 | 900 | 100.00% |
| SS | 900 | 900 | 900 | 100.00% |

No webcam device was opened. This is an offline video audit only.
