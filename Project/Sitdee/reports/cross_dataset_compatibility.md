# Cross-dataset compatibility

| Dataset | Local representation | Labels/IDs verified | Compatibility conclusion |
|---|---|---|---|
| MultiPosture | 33-landmark XYZ CSV, 13 subjects | 5 upper + 7 lower labels; subject IDs present | Primary supervised baseline; exact source scaling is still a live-adapter caveat |
| ANYA | 605,955-row pose-derived CSVs; four identical-core files | 33 indexed raw slots, 13 normalized slots; no verified target; filename groups only | File audit complete; ergonomic training and common adapter blocked pending label/mapping verification |
| SitPose | 13 derived joint/angle features, 5,746 local rows | 7 labels; no participant/session/frame IDs | Not feature-schema compatible with MultiPosture without a new mapping and grouping plan |
| SitStraightNet | 756 unlabeled JPEGs in the checked repository snapshot | No local annotation sidecars | Valid for MediaPipe smoke/preprocessing tests only; not supervised ground truth |
| MVSP/LGCSPNet | Repository instructions only | Not locally verified | No local comparison possible |

The common-looking phrase “sitting posture” is not enough to merge these datasets. MultiPosture's source labels describe upper-body direction (`upright`, forward/backward/left/right lean), while SitPose's labels and feature definitions are a different ontology. SitStraightNet images have no verified labels in the local snapshot. No pooled model or cross-dataset accuracy is claimed.

The next valid compatibility experiment is to obtain the source-backed ANYA label mapping plus a labeled, independently held-out RGB/video sample with known source coordinates or labels, run MediaPipe using the pinned task asset, verify the adapter's schema and normalization empirically, and report results separately by dataset.
