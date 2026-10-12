# Office-posture dataset collection summary

This is the collection-phase snapshot. Subsequent ML training, evaluation, offline inference, and the full ANYA audit are documented in `reports/final_project_status.md` and the training/audit reports.

Verification date: 2026-09-05 (Asia/Bangkok)

Scope: download, verify, organize, and document the source inventory. Raw downloads were not edited. Model and offline-inference status is maintained separately.

## Computer check

- Hostname: `DESKTOP-ALH6CUB`
- Windows: Microsoft Windows 11 Home, version `10.0.26200`
- Original working directory: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel`
- New workspace: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets`
- `C:` free space after the official ANYA download and audit: approximately `25.51 GB`
- Python: `3.12.10`
- Git: `2.55.0.windows.3`
- Node: `v22.14.0`
- Kaggle CLI/package: `2.2.4` installed in `.venv`
- Kaggle authentication: verified with the official file-list command; credentials were not printed or copied into the repository.

## Dataset status

| Dataset | Source | Status | Data type | Samples | Participants | Labels | Size | License | Best use |
|---|---|---|---|---:|---:|---|---:|---|---|
| ANYA | Kaggle | AUDITED_TARGET_BLOCKED | Four large CSVs with a shared core row sequence and model-specific predictions | 605,955 per file | Not available as an explicit field | No verified ground-truth target; `Predicted_Label` is output | 7,966,483,972 bytes of CSVs | Apache 2.0 | Most directly related ergonomic-posture source; target mapping required before training |
| MultiPosture | Zenodo | COMPLETE | 3D MediaPipe-style keypoint CSV with upper- and lower-body labels | 4,794 | 13 | 5 upper + 7 lower labels | 9,327,935 bytes | CC BY 4.0 | Easiest complete baseline for sitting-posture classification |
| SitPose | GitHub | PARTIAL | Joint-angle/feature CSV plus repository code | 5,746 local rows; paper reports 33,409 | Not present in local CSV; paper reports 36 | 7: `forward`, `healthy`, `lay`, `left`, `right`, `stand`, `wrong` | 601,411-byte CSV; 4,089,123-byte clone | Not stated in repository | Small feature-based baseline and completeness comparison |
| SitStraightNet | GitHub | PARTIAL | JPEG posture images plus model code | 756 local JPEGs; paper reports 17,596 frames | Not encoded locally | No local label/annotation file | 14,931,004 bytes of JPEGs; 14,936,143-byte clone | MIT | Image-modality example; not a complete benchmark release |
| MVSP / LGCSPNet | GitHub + Baidu Pan | MANUAL_ACTION_REQUIRED | Multi-view sitting-posture dataset; not locally downloaded | Not locally verified | Not locally verified | Not locally verified | 146-byte README only | Not stated in repository | Later multi-view extension after manual download |

## Verified local paths

- MultiPosture raw file: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\datasets\raw\multiposture\data.csv`
- SitPose raw public CSV: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\datasets\raw\sitpose\official-repository\Azure Kinect\data_all.csv`
- SitStraightNet public image directory: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\datasets\raw\sitstraightnet\official-repository\dataset`
- MVSP official instructions: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\datasets\raw\mvsp\official-repository\README.md`
- ANYA official files: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\datasets\raw\anya\official-download`

## Verification highlights

### ANYA

Kaggle’s current public metadata reports 7,966,483,972 bytes, Apache 2.0, and version 1 updated 2025-12-21. The four filenames (`ANN.csv`, `Random_Forest.csv`, `lgbm.csv`, and `xgb.csv`) were downloaded from the official client, hash-verified, and fully audited. They share the same core observations and differ in the final `Predicted_Label` output. No verified ground-truth target is present; see `reports/anya_schema_audit.md` and `reports/anya_four_file_comparison.md`.

### MultiPosture

The downloaded file is 4,794 rows × 102 columns. It has 13 subjects, 99 coordinate features (`_x`, `_y`, `_z` for 33 landmarks), two label columns, no missing values, and no exact duplicate rows. Its local MD5 `7efbc94d653acf32eefecec3ff26d6c6` matches Zenodo’s published checksum. The local file differs from the Zenodo prose description (4,800 frames and 11 joints/33 dimensions), so the local inventory is reported from the actual CSV.

### SitPose

The official shallow clone contains `Azure Kinect/data_all.csv` with 5,746 rows, 14 columns, 13 numeric/feature columns, and seven labels. No missing or exact duplicate rows were found. There are no participant/session/frame ID columns in this CSV. The local public release is smaller than the paper-reported 33,409 points from 36 participants, so it remains PARTIAL.

### SitStraightNet

The official shallow clone contains 756 decodable JPEGs, all 640×480, totaling 14,931,004 bytes. The dataset directory contains no local label/annotation sidecars or videos, and generic filenames do not carry labels. The paper reports 17,596 frames from 66 video sequences, so this public snapshot is PARTIAL.

### MVSP / LGCSPNet

The official repository contains only a README with a Baidu Pan download link and password. No dataset download was attempted. Manual login/verification is required and must be completed by the user.

## Recommendations

1. Most directly related to ergonomic posture: **ANYA**. It is explicitly an ergonomic-posture source, but the downloaded release still needs a source-backed ground-truth mapping before supervised training.
2. Easiest complete dataset to start future AI development: **MultiPosture**. It is small, structured, labeled, and already represented as pose keypoints rather than raw media.
3. Show the instructor first: **ANYA** as the high-priority target, **MultiPosture** as the complete local example, and **SitPose** as a clearly documented partial sitting-posture release.

## Storage and limitations

- Workspace storage now includes approximately 7.97 GB of extracted ANYA CSVs and the retained 3.61 GB official archive; current free C: space is approximately 25.51 GB.
- ANYA is locally downloaded and audit-complete, but training is blocked by the missing verified target.
- MVSP is not locally downloaded because Baidu Pan requires manual access.
- “Complete” for MultiPosture means the official Zenodo file was downloaded intact and checksum-verified; its local row count is 4,794, not the 4,800 stated in the record description.
- SitPose and SitStraightNet are repository snapshots, not proof that the full paper datasets are publicly included.
- No large unrelated datasets were downloaded; existing MultiPosture training was not unnecessarily repeated.

## Sources

- ANYA Kaggle dataset: https://www.kaggle.com/datasets/chanankornjandaeng/anya-a-dataset-for-ergonomic-posture-analysis
- Kaggle API documentation: https://www.kaggle.com/docs/api
- Kaggle CLI dataset commands: https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md
- MultiPosture Zenodo record: https://zenodo.org/records/14230872
- SitPose repository: https://github.com/TaylorDurden1114/SitPose
- SitPose paper: https://arxiv.org/abs/2412.12216
- SitStraightNet repository: https://github.com/realratchet/SitStraightNet
- SitStraightNet paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8022631/
- MVSP / LGCSPNet repository: https://github.com/LKW123800/multi-view-sitting-posture-dataset
