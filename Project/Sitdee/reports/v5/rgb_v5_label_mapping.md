# RGB-native V5 label mapping audit

Generated: 2026-09-05T10:51:46.709186+00:00

## Decision rule

Only labels that are explicit in the source and semantically compatible with
the project direction ontology are assigned a project label.  Motion labels,
head-only actions, general slouch labels without a direction definition, and
unjoinable archive images remain unlabelled.

| Source | Verified source labels / descriptions | Project mapping | Status | Training decision |
|---|---|---|---|---|
| DriveNet v6 | `Sit_Straight`, `Lean_Backward`, `Move_Backward`, `Right_Position` | `Sit_Straight` -> `TUP`; `Lean_Backward` -> `TLB`; others blank | COMPATIBLE / AMBIGUOUS | Used for the current 2-class pilot inventory |
| MMD-MSD public subset | `Corrected Sitting`, `Uncorrected Sitting`, `Sitting with stability disk` | corrected/active sitting -> `TUP`; uncorrected sitting is not a direction label | COMPATIBLE / AMBIGUOUS | Used only for TUP diversity and audit |
| MVSP/LGCSPNet | `upright`, `back`, `down`, `left`, `right`, `cheek`, `leg` | likely `TUP`, `TLB`, `TLF`, `TLL`, `TLR` for the first five | COMPATIBLE if downloaded and license resolved | Not used: official repository gives Baidu-only download and no license statement |
| CCTV Exam Monitor | `Correct Posture`, `LeftSideMove`, `RightSideMove`, `ForwardMove`, `BackwardMove`, `Stand` | potentially compatible, but per-image join is unavailable locally | AMBIGUOUS | Audit-only |
| SitStraightNet | `sitting straight`, `lightly hunched`, `hunched over`, `extremely hunched`, `partially lying`, `lying down` | no defensible exact direction mapping; source image-to-label join unresolved | AMBIGUOUS | Smoke-only |
| SitPose | Kinect/depth sitting postures | not RGB-native | INCOMPATIBLE for this track | Not used |

## Current supervised coverage

The current manifest has verified project labels only for `TUP` and `TLB`.
No source-backed RGB samples for `TLF`, `TLL`, or `TLR` were eligible for
training.  `SLOUCH` is postponed because the available sources either lack a
per-file join or use semantics that are not identical to the V5 direction
ontology.

## Source references

- [DriveNet v6](https://data.mendeley.com/datasets/v37nxdn2gh/6)
- [MMD-MSD official resource page](http://isr.tu-varna.bg/ergo/index.php/resursi)
- [MMD-MSD paper](https://www.mdpi.com/1999-4893/17/5/187)
- [MVSP/LGCSPNet repository](https://github.com/LKW123800/multi-view-sitting-posture-dataset)
- [MVSP/LGCSPNet paper](https://www.sciencedirect.com/science/article/pii/S0957417425021050)
- [CCTV Exam Monitor](https://www.kaggle.com/datasets/cctvdataset/cctv-exam-monitor-dataset)
- [SitStraightNet repository](https://github.com/realratchet/SitStraightNet) and [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8022631/)
- [SitPose paper](https://arxiv.org/abs/2412.12216)
