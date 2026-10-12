# ANYA schema audit

Status: **AUDITED; ergonomic classifier training blocked by missing verified ground truth**.

## Source fact

The official source is [ANYA: A Dataset for Ergonomic Posture Analysis](https://www.kaggle.com/datasets/chanankornjandaeng/anya-a-dataset-for-ergonomic-posture-analysis), Kaggle dataset version 1. The re-verified official inventory reports Apache-2.0 and four CSVs of `1,991,620,993` bytes each. The paper-level TRUE/FALSE ergonomic-label description is context only; it is not substituted for a downloaded target column.

## Verified local observation

- Downloaded unchanged under `datasets/raw/anya/official-download/`.
- Each CSV: `605,955` rows and `176` columns.
- Parser: UTF-8, comma delimiter, `pandas.read_csv(..., chunksize=20_000, low_memory=False)`.
- All four files: zero null cells, zero non-finite numeric cells, zero constant numeric columns, zero serialized duplicate rows.
- Exact CSV hashes are recorded in `metadata/anya_hashes.sha256`; the retained official ZIP is recorded in `metadata/anya_archive.sha256`.

## Exact schema

| Ordered column group | Exact columns | Count | Interpretation status |
|---|---|---:|---|
| Frame index | `frame` | 1 | VERIFIED numeric frame counter; resets at recording boundaries |
| Raw pose coordinates/visibility | `x0,y0,z0,vis0` through `x32,y32,z32,vis32` | 132 | VERIFIED pose-like numeric groups; anatomical names are not supplied |
| Normalized coordinates | `x0normalized,y0normalized,z0normalized` through `x12normalized,y12normalized,z12normalized` | 39 | VERIFIED normalized-looking numeric groups; exact transform and landmark mapping are not supplied |
| Recording/cluster/prediction metadata | `filename,clusterid,normclusterid,Predicted_Label` | 4 | VERIFIED metadata fields by name; semantic roles below |

The file therefore contains 33 indexed raw slots with XYZ plus visibility, and a separate 13-index normalized XYZ block. It does **not** contain named anatomical landmarks such as `nose`, `left_shoulder`, or `right_hip`; assigning those names to indices would be INFERRED, not VERIFIED.

## Landmark table

| Dataset column group | Landmark | x | y | z | visibility | Status |
|---|---|---|---|---|---|---|
| Raw slot `i`, `i=0..32` | Unnamed indexed pose slot | `xi` | `yi` | `zi` | `visi` | VERIFIED column pattern; anatomy/order NOT VERIFIED |
| Normalized slot `i`, `i=0..12` | Unnamed normalized pose slot | `xi normalized` | `yi normalized` | `zi normalized` | none | VERIFIED column pattern; anatomy/order/transform NOT VERIFIED |

Measured ranges in `ANN.csv` across all rows: raw x `[-0.338463, 1.289154]`, raw y `[-0.399686, 5.388710]`, raw z `[-4.194531, 2.799562]`, raw visibility `[2.19e-7, 0.999996]`; normalized x `[0.476872, 1.000000]`, normalized y `[0.162343, 0.932701]`, normalized z `[0.000000, 1.000000]`. These ranges do not prove hip centering or equivalence to the project's MultiPosture transform.

## Target audit

No downloaded column is a verified ground-truth ergonomic target:

- `clusterid` and `normclusterid` are binary-valued cluster identifiers, not labels by name or documentation in the CSV.
- `Predicted_Label` is explicitly a prediction output. Its values are integer `0/1`, and the four files disagree on it, so it cannot be promoted silently to ground truth.
- No `TRUE/FALSE`, `label`, `target`, or `class` ground-truth column is present.

The paper-level meaning of TRUE/FALSE may be used only after a source-backed mapping from the released columns is provided. The current files alone do not establish that mapping.

## Engineering decision

ANYA model training, metric reporting, error analysis, and model export are **not run** because doing so would train against an unverified model prediction or cluster ID. The raw files remain the source of truth. The conservative next input is the original expert-label mapping or a released ground-truth file/sidecar that can be tied to these rows/clusters.

