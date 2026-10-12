# ANYA ground-truth investigation

Verification date: 2026-09-05 (Asia/Bangkok)

## Final classification

**CURRENT KAGGLE RELEASE: B — PSEUDO_LABEL_ONLY**

The paper is source evidence that an expert-labeled subset existed in the ANYA
creation workflow. However, the downloaded Kaggle release does not expose that
subset, a TRUE/FALSE target, or a joinable cluster-to-label mapping. The current
files therefore cannot be used as a verified supervised target without an
additional source-backed artifact or author clarification.

This classification applies to the local release under
`datasets/raw/anya/official-download/`, not to the existence of the paper's
original annotation work.

## Sources checked

1. Official Kaggle dataset page and metadata/API endpoints:
   <https://www.kaggle.com/datasets/chanankornjandaeng/anya-a-dataset-for-ergonomic-posture-analysis>
   and
   <https://www.kaggle.com/api/v1/datasets/view/chanankornjandaeng/anya-a-dataset-for-ergonomic-posture-analysis>
2. ANYA paper, IEEE Access DOI:
   <https://doi.org/10.1109/ACCESS.2025.3633624>
3. Authors' public repository:
   <https://github.com/cjundang/ANYA-Dataset>
4. Local retained archive and extracted files, without changing either the
   archive or any CSV.

The checked author repository's main branch exposes a README and a pipeline
image, but no label sidecar, cluster mapping, supplementary data file, release,
or tag containing the expert subset. Kaggle currently exposes version 1 only,
noted as `Initial release`; no prior version was downloaded or used.

## Source-backed findings from the paper

The following are paper-level facts and are not silently promoted to local file
measurements:

- The data-collection workflow involved 30 student participants. The first 10
  recordings were used to create the expert-labeled subset.
- The paper reports 144,000 frames/records from those initial 10 recordings as
  manually or expert labeled, with `60,181` ergonomically correct and `83,819`
  ergonomically incorrect observations.
- The remaining approximately 288,000 observations were expanded using
  pseudo-labels from ANN, Random Forest, LightGBM, and XGBoost, for a paper-
  reported total of 432,000 fully labeled entries.
- The expert workflow sampled temporally contextual frames from K-means
  clusters and produced a conceptual mapping `L: C -> {TRUE, FALSE}`. The
  paper describes a threshold of at least 50% correct expert votes for TRUE.
- In the paper's semantic wording, TRUE means ergonomically correct and FALSE
  means ergonomically incorrect. These should be phrased as acceptable versus
  ergonomic-risk/non-acceptable posture in this project, not as a medical
  diagnosis.
- A second certified ergonomist reviewed 1,000 frames; the paper reports
  Cohen's kappa of 0.89 for that review.

The paper does not provide, in the checked accessible material, a file-level
mapping that identifies the exact local rows belonging to the 144,000 expert
observations, the storage location of `L`, or a numeric mapping from the local
`clusterid`/`normclusterid` values to TRUE/FALSE.

## Verified local observations

The four downloaded files are:

| File | Rows | Columns | Verified content |
|---|---:|---:|---|
| `ANN.csv` | 605,955 | 176 | Shared observation core plus ANN `Predicted_Label` output |
| `Random_Forest.csv` | 605,955 | 176 | Same core plus Random Forest `Predicted_Label` output |
| `lgbm.csv` | 605,955 | 176 | Same core plus LightGBM `Predicted_Label` output |
| `xgb.csv` | 605,955 | 176 | Same core plus XGBoost `Predicted_Label` output |

The full scan verified:

- All four files have the same ordered core row sequence; the final
  `Predicted_Label` sequences differ.
- No column named TRUE/FALSE, `label`, `target`, or other verified expert
  target is present. The only label-like field is `Predicted_Label`.
- `Predicted_Label` contains numeric `0`/`1` outputs. No official local artifact
  maps numeric 0/1 to the paper's TRUE/FALSE meanings.
- `clusterid` has values `0` and `1`; `normclusterid` also has values `0` and
  `1`. Their local names and counts do not prove that either field is an
  ergonomic label or encode the expert mapping.
- `filename` contains 32 recording-like names such as
  `sample_01_front.csv`, but there is no participant/session column. The
  release has gaps in the numeric filename sequence, so the paper's initial
  10 recordings cannot be identified safely as `sample_01` through
  `sample_10`.
- The paper-reported 432,000 fully labeled entries do not match the locally
  observed 605,955 rows per file. No evidence currently proves that the
  144,000 expert subset is present in, or directly traceable within, this
  release.

## Local archive and metadata search

The retained ZIP contains exactly the four CSV members and no README, JSON,
mapping, annotation, or label sidecar. The local ANYA directory contains only
those four CSVs and the retained ZIP. The local Kaggle record has an empty
description and reports only version 1. The local search found no hidden
metadata or alternate release containing TRUE/FALSE labels.

## Pseudo-label consensus diagnostic

The four `Predicted_Label` columns were compared in bounded-memory chunks. This
was an agreement diagnostic only; it was not training or validation against
ground truth.

- Four-way unanimous: `459,841 / 605,955` (`75.887%`).
- Not unanimous: `146,114 / 605,955` (`24.113%`).
- Exact 3-versus-1 split: `90,979` (`15.014%`).
- 2-versus-2 tie: `55,135` (`9.099%`).
- 3-of-4-or-unanimous, excluding 2-versus-2 ties: `550,820` (`90.901%`).
- Pairwise agreement ranged from `80.600%` (ANN/XGB) to `97.807%`
  (LightGBM/XGB). Cohen's kappa ranged from `0.612596` to `0.956122`.
- Agreement varies substantially by `filename`, `clusterid`, and
  `normclusterid`; therefore consensus cannot be treated as a universal
  correctness guarantee.

Full counts, pairwise confusion matrices, kappa values, and group breakdowns
are in [`anya_pseudolabel_consensus.md`](anya_pseudolabel_consensus.md) and
`outputs/metrics/anya_pseudolabel_consensus.json`.

## Answers to the investigation questions

### 1. Where is the original 144,000-row expert-labeled subset?

**Not identified in the current release.** The paper supports that the subset
was used during dataset creation, but the Kaggle ZIP and checked author
repository do not expose a traceable subset file or row-level mapping.

### 2. How were `clusterid` and `normclusterid` generated and what do they mean?

The paper supports a K-means/clustering-based annotation workflow and a
cluster-to-expert-label concept. It does not, in the checked source material,
document a local-file schema that proves which of these exact columns stores
the expert mapping, whether `normclusterid` is a globally normalized cluster
identifier, or how numeric values map to TRUE/FALSE. **Do not use either as a
target.**

### 3. Is the expert mapping stored elsewhere?

No mapping was found in the current Kaggle metadata, version 1 file list, ZIP,
local metadata, or checked public author repository. It may exist privately or
in an unlinked historical/supplementary artifact; this remains an author-
clarification request, not a verified fact.

### 4. Can the paper alone reconstruct ground truth?

**No.** It describes the procedure and aggregate counts, but not a row-joinable
mapping for this 605,955-row release. Reconstructing labels from
`Predicted_Label`, cluster IDs, row order, or guessed filenames would be
scientifically unsafe.

### 5. Which recordings are the initial 10?

**Not identified.** The paper says the first 10 recordings, while the local
release exposes 32 recording-like filenames with missing sequence numbers and
no authoritative initial-10 manifest.

### 6. Is the 144,000-frame subset present in this 605,955-row release?

**Unproven.** The local files have 605,955 rows each; no source-backed key or
metadata identifies a 144,000-row subset. The paper count is retained as a
paper-level fact only.

### 7. Can ANYA training begin now?

**No, not as a verified ergonomic classifier.** `Predicted_Label` can be used
for pseudo-label provenance, consensus, or a separately documented diagnostic,
but not as expert ground truth. Training is blocked pending a source-backed
expert subset or mapping.

## Engineering decision and next safe action

- Do not train the ANYA ergonomic classifier in this investigation.
- Do not translate numeric 0/1 or cluster IDs into TRUE/FALSE by guess.
- Keep the raw CSVs and archive immutable; hashes remain in the existing
  manifests.
- The prepared English author request in
  [`anya_author_request.md`](anya_author_request.md) asks for the expert
  subset, cluster mapping, initial-10 manifest, and release relationship. It
  has **not** been sent.
- If a response supplies an exact joinable mapping, create a separately
  versioned processed target, rerun the leakage-safe split design, and only
  then begin the planned model benchmark.

