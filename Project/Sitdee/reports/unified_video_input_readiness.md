# Unified video input readiness

Status: `WEBCAM_CODE_READY_NOT_LIVE_VALIDATED`.

The existing `VideoSource` remains the only frame-input contract for local
files, USB indexes, and RTSP profiles/URLs. It now accepts best-effort USB
width, height, and FPS preferences; source health reports the actual values
accepted by the backend. Open/read failure, file end-of-stream, reconnect,
latest-frame RTSP use, and cleanup retain their existing states.

`scripts/run_upper_body_posture_v2.py` is the shared experimental runner:

`file / USB / RTSP -> VideoSource -> YOLO/ByteTrack -> CandidateSelector -> padded-crop safety -> Pose V2 -> explicit upper-body V2 model`

`--source webcam --camera-index N` resolves to the USB source without opening
hardware during parsing. Files remain synchronous; RTSP alone uses the
bounded latest-frame reader. Camera preferences do not affect file/RTSP
frames. The selected model directory remains explicit and experimental.
Predictions are diagnostic `MODEL_PREDICTION` records, never ground truth.
The operator clicks an anonymous `TRACK` box to confirm seated context; no
face recognition or names are used.

Validation: 48 focused tests passed with no camera/network access. Tests use
fake captures for webcam open/read/preferences/failure/release, decode one
frame through a real local file source, verify EOF and bounded source state,
and retain RTSP redaction/reconnect, Pose V2 safety, model schema, V4
isolation, and long-session regressions.

An end-to-end offline smoke used
`validation_data/external/labeled/upright/DriveNet_v6_SS_1.mp4` through the
same runner. It ended cleanly after 8.97 seconds with one processed frame,
one anonymous track, one `POSE_OK`, one valid pose attempt, no image/video
recording, and V4 not called. Experimental V2 returned `UNKNOWN` with
`POSSIBLE_OOD_EXPERIMENTAL`; this is the correct refusal behavior and is not
an accuracy result. Output is under ignored `outputs/unified_video_smoke/`.

Tomorrow's webcam smoke command:

```powershell
.\.venv_yolo\Scripts\python.exe .\scripts\run_upper_body_posture_v2.py --source webcam --camera-index 0 --camera-width 1280 --camera-height 720 --camera-fps 30 --max-seconds 60
```

Expected display: the USB frame, anonymous `TRACK <id>` boxes, and
`UNKNOWN` until the operator clicks the seated target. Valid experimental
outputs then show `CORRECT`/`INCORRECT` plus the subtype; invalid, ambiguous,
stale, or OOD evidence remains `UNKNOWN`. Press `Q` or Escape to stop.

Still unverified: physical webcam discovery/index, backend-selected capture
resolution/FPS, device stability, live pose availability, live latency, and
live prediction behavior. No webcam, RTSP, Hikvision, or Tailscale source was
accessed in this milestone.

Frozen V4 SHA-256 remained
`af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd`.
