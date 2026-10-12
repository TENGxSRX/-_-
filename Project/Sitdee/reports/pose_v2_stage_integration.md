# Optional Pose V2 stage

This milestone adds an off-by-default `PoseV2Stage` to the autonomous
long-session runner.  It runs after `CandidateSelector` and processes only
the selected ByteTrack boxes.  The stage uses a separate single-person
MediaPipe crop backend and writes `pose_telemetry_*.jsonl` records.

The rate limiter is global across all tracks.  Its default is
`max_pose_per_second=2.0`; a waiting track is selected by oldest
attempt/success time, with track ID only as a deterministic tie-breaker.
Unserved selected tracks receive `POSE_PENDING` records.  Each selected box
is checked against all current tracked boxes, including unselected boxes.
Ambiguous overlap is rejected with `POSE_AMBIGUOUS` and does not consume a
crop-attempt token; unselected boxes never produce pose records.

Pose records contain the pose state, visibility/core quality, schema version,
landmarks when available, and explicit `v4_prediction: null`.  Pose V2 never
loads or calls V4 and never creates a label.  A pose writer reaching capacity
raises the existing `TelemetryCapacityReached` path, producing a graceful
`TELEMETRY_CAPACITY_REACHED` session stop with finalized summary/manifest.

Validation covered disabled-by-default compatibility, global budget/fairness,
pending and invalid-pose states, writer-capacity shutdown, clean close, and
existing long-session capacity tests.  A real offline qualification using
`SS_1.mp4` completed with `DURATION_COMPLETED`, 9 pose attempts, 8
`POSE_OK`, 1 `POSE_UNAVAILABLE`, and 2 `POSE_PENDING`; all 11 pose JSONL
records parsed successfully.  No camera, RTSP, model training, or dataset
processing was used.
