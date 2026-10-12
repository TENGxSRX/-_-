# Source inventory and verification notes

Verification date: 2026-09-05 (Asia/Bangkok)

## Live source metadata

- ANYA: https://www.kaggle.com/datasets/chanankornjandaeng/anya-a-dataset-for-ergonomic-posture-analysis
  - Public metadata endpoint: `https://www.kaggle.com/api/v1/datasets/view/chanankornjandaeng/anya-a-dataset-for-ergonomic-posture-analysis`
  - Title: `ANYA: A  Dataset for Ergonomic Posture Analysis`
  - Current version: 1
  - Last updated: 2025-12-21T01:45:48.213Z
  - Dataset size reported by Kaggle metadata: 7,966,483,972 bytes
  - License reported by Kaggle metadata: Apache 2.0
  - Local status: authenticated official download complete. Four CSVs and the retained ZIP are under `datasets/raw/anya/official-download/`; exact local hashes are in `metadata/anya_hashes.sha256` and `metadata/anya_archive.sha256`.
  - Version-history check: official metadata exposes version 1 (`Initial release`) only; no historical full release was downloaded. Details are in `metadata/anya_version_history.json`.

- MultiPosture: https://zenodo.org/records/14230872
  - Record JSON: https://zenodo.org/api/records/14230872
  - File: `data.csv`
  - Published file size: 9,327,935 bytes
  - Published MD5: `7efbc94d653acf32eefecec3ff26d6c6`
  - License identifier: `cc-by-4.0`
  - Local SHA-256: `d21579200db9a57dd01c24fa6ece961b0be73aa0d5d402ae8e629761c03ebe38`
  - Local MD5: `7efbc94d653acf32eefecec3ff26d6c6` (matches)

- SitPose: https://github.com/TaylorDurden1114/SitPose
  - Local shallow-clone commit: `b1f8e80642a3b534f207096bd11f7e0de509c9a4`
  - Public data file: `Azure Kinect/data_all.csv`
  - Local SHA-256: `55750a4d1deca910cb4573041ba4afc340009941cde77d9492c1025adc5ed009`

- SitStraightNet: https://github.com/realratchet/SitStraightNet
  - Local shallow-clone commit: `b928ab8804c200a3b413e88c31acb8004a5f7cc8`
  - Dataset image inventory: 756 JPEG files, all 640×480
  - Local JPEG byte-inventory SHA-256 (concatenated sorted file bytes): `a20e3d9c3299a0e415e186e001694a3b3ea393779da52b75c0ccb3a7293e32e2`
  - Repository license: MIT

- MVSP / LGCSPNet: https://github.com/LKW123800/multi-view-sitting-posture-dataset
  - Local shallow-clone commit: `0510efd289b7634afa78026ef6fee3005ef74b5b`
  - Repository contains only a 146-byte README with an official Baidu Pan link and extraction password.
  - No dataset was downloaded; manual access is required.

## Local file verification

### ANYA official download

- Four CSVs: `ANN.csv`, `Random_Forest.csv`, `lgbm.csv`, and `xgb.csv`.
- Each CSV is 1,991,620,993 bytes, 605,955 rows, and 176 columns; total CSV bytes are 7,966,483,972.
- The retained ZIP is `anya-a-dataset-for-ergonomic-posture-analysis.zip`, 3,607,095,951 bytes.
- The four CSVs share the same ordered core rows and differ only in their final `Predicted_Label` output column. No verified TRUE/FALSE ground-truth column is present. The current release is classified `B — PSEUDO_LABEL_ONLY`; the paper-reported expert subset is not row-joinable from the release. See `reports/anya_ground_truth_investigation.md` and `reports/anya_pseudolabel_consensus.md`.
- Full schema, target decision, temporal grouping evidence, and quality results are recorded in `reports/anya_schema_audit.md`, `reports/anya_four_file_comparison.md`, `reports/anya_leakage_analysis.md`, and `reports/anya_data_quality_audit.md`.

### MultiPosture `data.csv`

- 4,794 data rows and 102 columns.
- `subject` contains 13 IDs: `1` through `13`.
- `upperbody_label`: `TLB` 442, `TLF` 1,897, `TLL` 420, `TLR` 420, `TUP` 1,615.
- `lowerbody_label`: `LCL` 609, `LCR` 571, `LWA` 1,030, `LAP` 1,153, `LCS` 631, `LLR` 419, `LLL` 381.
- 99 coordinate columns follow the `_x`, `_y`, `_z` pattern, representing 33 named landmarks.
- No missing values, malformed rows, or exact duplicate rows were found.
- The Zenodo description says 4,800 frames and 11 joints/33 input dimensions; the local file itself is authoritative for this collection and contains 4,794 rows, 102 columns, and 99 coordinate features. This discrepancy is recorded rather than silently corrected.

### SitPose `data_all.csv`

- 5,746 data rows and 14 columns.
- 13 feature columns: `head_x`, `head_y`, `head_z`, `JAB`, `JAC`, `ABC`, `BCE`, `JAD`, `KDE`, `NIH`, `MGF`, `HIC`, `FGC`.
- `label`: `forward` 714, `healthy` 987, `lay` 649, `left` 820, `right` 664, `stand` 837, `wrong` 1,075.
- No missing values or exact duplicate rows were found.
- No participant, session, video, or frame identifier column was present in the public CSV.
- The public file is smaller than the paper-reported 33,409 points from 36 participants, so it is marked PARTIAL.

### SitStraightNet public repository

- `dataset/` contains 756 `.jpg` files totaling 14,931,004 bytes.
- All 756 images decoded successfully and are 640×480.
- No CSV, JSON, XML, TXT, MAT, YAML, MP4, AVI, or MOV sidecar/video files were present under `dataset/`.
- JPEG names are generic (`image001.jpg`, `image002.jpg`, ...); no label is encoded in the filename.
- The checked repository snapshot therefore cannot verify the full 17,596 labeled frames reported by the paper and is marked PARTIAL.

## Manual actions intentionally not performed

### ANYA

No additional manual download action was found to be safe or useful. The Kaggle token was used only by the official client and was not printed or copied into the repository. The current release is pseudo-label-only for this project; training remains blocked until the source-backed expert-label mapping or a ground-truth sidecar is available. A polite author request was drafted in `reports/anya_author_request.md` and was not sent.

### MVSP

The official README gives this manual download URL:

https://pan.baidu.com/s/1577RDm5dmdXtQqNUjwJusw?pwd=qm7x

Open it manually, complete any Baidu login or verification requested by the site, use the public extraction password shown in the official README (`qm7x`), and save the result under `datasets/raw/mvsp/manual-download/`. No CAPTCHA, password, or access restriction was bypassed.
