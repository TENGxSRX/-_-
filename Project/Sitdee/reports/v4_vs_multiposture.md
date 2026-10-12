# V4 versus MultiPosture

| Attribute | RGB-native V4 pilot | MultiPosture V1/V2 |
|---|---|---|
| Source modality | DriveNet RGB video, MediaPipe extracted at run time | Processed landmark CSV; raw RGB withheld |
| Labels used | TUP and TLB only; MB/RP unmapped | TUP/TLF/TLB/TLL/TLR |
| Coordinate path | MediaPipe World Landmarks → hip midpoint translation → features | Released processed coordinates; source/model-input provenance caveat |
| Grouping | participant ID encoded in official video IDs | subject column |
| Selected model | LogisticRegression + normalized_world | HistGradientBoosting + 30-feature hybrid V2 |
| Grouped CV macro F1 | 1.0000 ± 0.0000 | 0.9925 ± 0.0087 (five-class) |
| Holdout macro F1 | 1.0000 (participant 3) | 1.0000 (two-subject holdout; cautious) |
| Real-RGB compatibility | Same extractor/training path; compatible only for pilot labels | Semantic compatibility remains not fully validated |

Metrics are not directly comparable because the ontologies, modalities, number of classes, and data sources differ.
