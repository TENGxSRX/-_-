# RGB-native V5 dataset expansion status

Generated: 2026-09-05T10:51:46.711693+00:00

## Result

**V5 MULTI-CLASS MODEL: NOT TRAINED.**  The source-backed RGB inventory is
not sufficient for a scientifically defensible five-class direction model.
No pseudo-labels were created and no V5 model artifact was exported.

## Evidence classification

### SOURCE FACT

DriveNet v6 publishes RGB posture source labels and participant context; the
MMD-MSD resource page publishes a five-participant public subset. The source
pages and paper links are listed in `reports/v5/rgb_v5_label_mapping.md`.

### VERIFIED LOCAL FILE OBSERVATION

The local V5 manifest contains 452 rows: 417 sampled DriveNet video frames and
35 MMD-MSD images. All 452 rows have detected core pose landmarks and finite
V4-compatible features. Project-label counts are TUP 244 and TLB 183; 25 rows
are retained for audit without a project label. The hash inventory contains 44
source files and 1,146,521,497 bytes: six existing DriveNet videos plus 35
MMD-MSD JPGs and three MMD-MSD angle CSVs. The MMD-MSD V5 raw subset alone is
238,743,255 bytes.

### EXPERIMENT RESULT

The 1 fps deterministic sampling audit was completed, MediaPipe World Landmark
features were extracted, and the manifest/feature arrays were written. No V5
classifier was fitted because the required classes TLF, TLL, and TLR have zero
eligible source-backed rows. Grouped CV, holdout, LODO, model comparison, and
classifier error analysis are therefore **NOT RUN**.

### ENGINEERING DECISION

The current V5 result is a data-readiness gate, not a failed model. V4 remains
the frozen baseline. No pseudo-labels, webcam access, ANYA training, YOLO
installation, or raw-data mutation was performed.

## Dataset inventory

| Source | Download/use decision | Media | Notes |
|---|---|---|---|
| DriveNet v6 | Used | RGB video | six mapped videos; SS -> TUP and LB -> TLB only |
| MMD-MSD public subset | Used for TUP diversity/audit | RGB images | 35 images, five participants, corrected/active sitting can support TUP only |
| MVSP/LGCSPNet | Not used | RGB image dataset reported by paper | exact missing classes appear available, but official repo has no license statement and the only download is Baidu |
| CCTV Exam Monitor | Audit only | RGB JPG | no per-image label sidecar in the local archive |
| SitStraightNet | Smoke only | RGB-D images | no verified per-file labels in the local snapshot |
| SitPose | Not used | Kinect depth/skeleton | not normal RGB media |

## Manifest and extraction

- Manifest rows: **452**
- Source-backed supervised rows: **427**
- Project label counts: **{'TLB': 183, 'TUP': 244}**
- Participants by available class: **{TLB: ['1', '2', '3'], TUP: ['1', '19', '2', '20', '21', '22', '23', '3']}**
- Missing direction classes: **TLF, TLL, TLR**
- Pose pipeline: MediaPipe Pose World Landmarks -> hip-centered V4-compatible features
- Core landmarks: `nose, left_shoulder, right_shoulder, left_hip, right_hip`
- Feature families prepared: `normalized_world, geometry, hybrid`
- DriveNet sampling target: **1 fps**, with 1/2/5 fps counts retained for audit
- Pose detection and core acceptance are recorded per row in `manifest.csv`; no frame is silently imputed

## Leakage controls

The manifest keeps participant/session/video identifiers.  No random frame
split was performed.  A final five-class model is gated until all direction
classes have source-backed labels and enough independent participant/source
groups.  Current class/source confounding is severe for any attempted
direction benchmark because the available direction labels come from one
controlled source domain.

## Required next data

The immediate missing classes are **TLF, TLL, and TLR**.  The best candidate
identified is MVSP/LGCSPNet because its published category descriptions
explicitly include down/left/right, but it cannot be admitted until the
dataset license and downloadable archive are verified.  Otherwise collect a
small, explicitly labelled RGB desk dataset with at least 6--10 participants,
all five direction classes per participant, and separate recording sessions.

## Model and evaluation status

Models benchmarked: **none for V5**, because training was correctly blocked
before model fitting.  Grouped CV, participant holdout, leave-one-participant-
out, and leave-one-dataset-out metrics are therefore **NOT RUN**.  This is a
data-readiness decision, not a failed classifier result.

V4 was not retrained or overwritten.  ANYA was not used.  Webcam source 0 was
not opened.  YOLO was not installed.

## Artifacts

- Unified manifest: `datasets/processed/rgb_native_v5/manifest.csv`
- Prepared features: `datasets/processed/rgb_native_v5/features.npz`
- Source hashes: `outputs/metrics/rgb_native_v5_source_hashes.json`
- Sampling audit: `outputs/metrics/rgb_native_v5_sampling.json`
- Model gate: `outputs/models/rgb_native_v5/NOT_TRAINED.md`
- Label mapping: `reports/v5/rgb_v5_label_mapping.md`
- Confound audit: `reports/v5/dataset_class_confounds.md`

## Source hash count

The hash inventory contains **44** downloaded/source files.  Raw
DriveNet, MultiPosture, ANYA, and prior V4 artifacts were not modified.
