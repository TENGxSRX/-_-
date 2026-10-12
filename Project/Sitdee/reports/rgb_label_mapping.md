# RGB-native V4 label mapping

## Source facts

- Source: DriveNet v6 official Mendeley record [https://data.mendeley.com/datasets/v37nxdn2gh/6](https://data.mendeley.com/datasets/v37nxdn2gh/6)
- License: CC BY 4.0
- Local subset: 6 source videos downloaded from the official per-file video folders; sampling is deterministic at 300 frames/video.

## Candidate-source audit

| Candidate | Local/source observation | Supervised V4 decision |
|---|---|---|
| DriveNet v6 | Official Mendeley record exposes per-class videos, `meta_data.csv`, 3 participant IDs, and CC BY 4.0. | Used for the two-class pilot below. |
| CCTV Exam Monitor | Official Kaggle archive is public-domain/CC0 and documents six class names, but the downloaded archive contains flat JPGs under train/valid/test with no row-level annotation or label sidecar. | Audit-only; not used because per-image labels cannot be joined without guessing. |
| SitStraightNet | Official repository is MIT, but the checked local image snapshot has generic filenames and no local label/participant sidecar. The paper taxonomy is not a usable image-to-label join in this release. | Smoke/preprocessing only; not supervised. |
| MVSP/LGCSPNet | Official repository points to a Baidu download and does not expose a locally verified archive/license in this run. | Not used; access/provenance remains unresolved. |

The absence of an annotation sidecar is treated as a scientific blocker for supervised use, not as permission to infer labels from appearance or directory order.

## Mapping decision

| DriveNet source label | Project label | Decision | Reason |
|---|---|---|---|
| SS (Sit Straight) | TUP | COMPATIBLE | Static straight sitting posture is directionally compatible with upright. |
| LB (Lean Backward) | TLB | COMPATIBLE | Source name explicitly states lean backward. |
| MB (Move Backward) | — | AMBIGUOUS / NOT USED | Move may denote an action or transition, not a sustained trunk posture. |
| RP (Right Position) | — | AMBIGUOUS / NOT USED | Source name does not define right lean versus seat position or action. |

Only SS and LB were allowed into supervised target training. No labels were invented for MB or RP.
