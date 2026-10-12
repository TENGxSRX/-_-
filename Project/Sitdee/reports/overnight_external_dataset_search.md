# Overnight external RGB dataset search

## Decision

**NO_VALID_EXTERNAL_5CLASS_DATA** was obtained in this run. No new raw archive was downloaded. This is a deliberate storage and scientific-integrity decision: C: free space was approximately 19.97 GB before the run, while the known candidates either lacked a verified label join, were not normal RGB, were inaccessible through a manual-only host, or used a different posture ontology.

## Search scope and source-backed screening

| Candidate | Media / labels observed | People / grouping | License/access | Decision |
|---|---|---|---|---|
| MultiPosture / Zenodo | Processed pose coordinates only; TUP, TLF, TLB, TLL, TLR | 13 participants; subject column in local CSV | Zenodo record; processed data, no raw RGB | Keep as research five-class reference; cannot add RGB samples |
| MVSP / LGCSPNet | Paper describes 49,798 RGB images and seven sitting categories including upright/back/left/right/down/cheek/leg | Paper reports multi-view fixed/dynamic views; local label/media join not verified | Official GitHub README points to a Baidu archive; no local archive/license verification | **Manual-access candidate**; do not bypass Baidu login/CAPTCHA or download automatically |
| DriveNet v6 | RGB videos/images with source labels; locally verified mapped subset SS→TUP and LB→TLB | Three participants in the six verified videos; video IDs encode participant | Mendeley, CC BY 4.0 | Already used; does not add missing TLF/TLL/TLR |
| MMD-MSD | Local public subset is image/angle material with only a defensible TUP mapping in the current manifest | Five participant blocks inferred from ordered subset files; no per-file label sidecar | Public resource/subset; mapping remains limited | Keep as audit-only/TUP support; not a missing-class source |
| SitStraightNet | RGB-D video study with straight/hunched/lying-style classes; local repository snapshot lacks per-file ground-truth join | Paper reports 11 subjects and 66 labeled sequences | Public repository/paper; local snapshot not a joinable supervised release | Smoke/audit only; not MultiPosture five-class RGB |
| CCTV Exam Monitor | JPG files and source class vocabulary including forward/backward/left/right moves; local archive has hashed images without a verified sidecar | Train/valid/test archive structure; participant/session IDs not verified | Kaggle record; local label join absent | Audit only; no supervised transfer |
| SitPose | RGB-D/depth/skeleton sitting posture data; not ordinary RGB-only MediaPipe source | 36 participants reported | Paper/repository access | Reject for this RGB objective |
| Smart cushion / pressure studies | Pressure/sensor recordings with proper/lean-left/right/forward/back labels | Participant metadata exists in papers | Not RGB media | Reject; useful only as non-RGB context |
| MOPED25 / occupational posture resources | Motion/kinematics and synchronized video; workplace tasks rather than a verified seated five-class ontology | 11 published subjects; task/session grouping | Official GitHub/repository; substantial multimodal resource | Not downloaded; useful future research lead, not a drop-in missing-class source |

## Search conclusion

The search found source-level evidence for potential left/right classes in MVSP, but the actual archive, per-image label join, participant IDs, and license terms were not verified locally. MVSP's `down` category is also a compound “lean forward and bend over” label and is not silently equated with MultiPosture TLF. Therefore the V5 gate remains closed and no pseudo-labels or guessed class mappings are introduced.

## Manual-access candidates

1. **MVSP / LGCSPNet official archive** — inspect the Baidu download manually, confirm archive contents, label files, participant/view IDs, and license before any import.
2. **MOPED25 official repository** — investigate only if a seated RGB subset and legally usable frame-level/task mapping can be proven.

## Sources

- [MultiPosture Zenodo record](https://zenodo.org/records/14230872)
- [MVSP/LGCSPNet official repository](https://github.com/LKW123800/multi-view-sitting-posture-dataset)
- [LGCSPNet/MVSP paper](https://www.sciencedirect.com/science/article/pii/S0957417425021050)
- [DriveNet Mendeley record](https://data.mendeley.com/datasets/v37nxdn2gh/6)
- [SitStraightNet repository](https://github.com/realratchet/SitStraightNet)
- [SitPose paper](https://arxiv.org/abs/2412.12216)
- [MOPED25 official repository](https://github.com/human-systems-ise-ncsu/MOPED25)
- [CCTV Exam Monitor Kaggle record](https://www.kaggle.com/datasets/cctvdataset/cctv-exam-monitor-dataset)

## Integrity boundary

No source archive was downloaded in this track. Existing raw datasets, the verified notebook backup, and V1–V4 artifacts were left unchanged.
