# V5 missing-class license and access audit

Date: 2026-09-05

This report separates whether data can be accessed from whether its use terms
are verified. A public link or a repository is not treated as permission to
train on the images.

| Source | Data access | License/use classification | Evidence | V5 decision |
|---|---|---|---|---|
| MVSP/LGCSPNet | Official repo points to Baidu Pan; archive not obtained | **LICENSE_UNCLEAR** | Repository has a 3-line README and download link but no data license statement | Reject until archive/license are verified |
| DriveNet v6 | Mendeley files already retained | **LICENSE_VERIFIED** — CC BY 4.0 on the record | [Mendeley record](https://data.mendeley.com/datasets/v37nxdn2gh/6) | Existing TUP/TLB audit/training source only |
| MMD-MSD public subset | Official university page links public Google Drive subset | **RESEARCH_USE_ONLY / DATA-TERM REVIEWED** | [official resource page](http://isr.tu-varna.bg/ergo/index.php/resursi) and [paper](https://www.mdpi.com/1999-4893/17/5/187) | Existing TUP diversity/audit only; not a missing-class source |
| CCTV Exam Monitor | Kaggle archive accessible locally | **LICENSE_UNCLEAR** for this project | Kaggle source page has a public-domain/CC0 claim, but local archive has no per-image label join or independent participant metadata | Audit only |
| SitStraightNet | Public GitHub code/repository snapshot | **LICENSE_UNCLEAR** for image data | Repository code license does not establish image-data reuse terms; per-file label join missing | Smoke only |
| SitPose | Paper describes Kinect/depth/skeleton data | **NOT_USABLE for RGB V5** | [paper](https://arxiv.org/abs/2412.12216) | Reject by modality |
| Smart-cushion posture dataset | Paper reports pressure recordings | **NOT_USABLE for RGB V5** | [study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5421679/) | Reject by modality |

## MVSP manual access instructions

1. Open the [official MVSP repository](https://github.com/LKW123800/multi-view-sitting-posture-dataset).
2. Open the linked Baidu Pan page normally; sign in or complete any required
   CAPTCHA yourself.
3. Download the original archive without renaming it.
4. Place it under `datasets/raw/rgb_native_v5/mvsp/`.
5. Record archive size and SHA-256; inspect its README, labels, file layout,
   participant/session/video IDs and license terms.
6. Do not add rows to the supervised manifest until the label join and license
   are explicitly verified.

No access control was bypassed and no MVSP archive was downloaded in this run.
