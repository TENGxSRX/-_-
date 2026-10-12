# Padded-crop cross-person safety

Root cause: raw-box IoU safety did not cover the padded pixels actually
passed to MediaPipe. A successful pose/core check did not establish that
the crop contained only the selected person.

The stage now computes the padded, frame-clamped, outward-rounded pixel
rectangle first and uses that same rectangle for safety and extraction.
Every current `all_boxes` entry is checked, including unselected and
untracked detections. Self is excluded by the selected anonymous track ID.
Any positive-area intersection with another person box is conservatively
rejected; edge contact alone is allowed. This is a bounding-box safety
proxy, not segmentation or proof of physical identity. It can reject crops
whose intersecting box areas contain background and cannot detect a person
missing from the detector output.

Rejection emits `POSE_AMBIGUOUS`, `PADDED_CROP_REJECTED`, and
`PADDED_CROP_PERSON_CONTAMINATION`. The rejected record includes the actual
pixel crop rectangle. It never calls MediaPipe, updates attempt/success
timestamps, or spends the global token. Only `selected_boxes` remain
eligible. Padding remains 0.10, the existing raw IoU gate remains 0.25,
and global rate/fairness scheduling is unchanged.

## Validation

50 focused/regression tests passed using the existing main environment:
pose stage, autonomous long-session shutdown/writers, upper-body V2
contracts, learning-pool integration, and YOLO tracking adapter.
The first default-temp invocation reached 47 passing checks but failed
pytest teardown on a Windows temp-folder permission error. Rerunning with
a new task-specific `--basetemp` completed normally with exit code 0.
New tests cover clean crops, disjoint raw boxes reached by padding,
unselected/untracked neighbors, self copies, multiple neighbors in either
order, pixel rounding, edge contact, budget retention, and fairness.

Real offline qualification used 16 evenly spaced decoded frames per video,
the existing local YOLO26n/ByteTrack and MediaPipe model, and resized each
frame to width 960 before both detection and cropping:

| Local video | Decoded frames | Maximum detected boxes | POSE_OK | Padded-crop rejections | Attempts |
|---|---:|---:|---:|---:|---:|
| validation_data/external/labeled/upright/DriveNet_v6_SS_1.mp4 | 16 | 1 | 16 | 0 | 16 |
| validation_data/external/unlabeled_smoke/DriveNet_v6_MB_1.mp4 | 16 | 2 | 11 | 5 | 11 |

Only one selected track per frame was eligible; all other boxes were safety
context. No contaminated selected crop produced POSE_OK. The five rejected
MB_1 observations produced no pose attempts. These are geometric/engineering
checks, not posture accuracy, identity continuity, or detector recall.
Results are in ignored `outputs/metrics/padded_crop_safety_offline.json`.
No camera, RTSP, classification, model training, raw-video recording, or
image saving was used.

V4 SHA-256 verified unchanged:
`af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd`.
Pre-existing unrelated untracked live-demo files remain outside this commit.
