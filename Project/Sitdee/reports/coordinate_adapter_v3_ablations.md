# Coordinate adapter V3 ablation study

Ablations are temporary controlled experiments using HistGradientBoosting and the existing subject-grouped split. No model is exported and V2 is not overwritten.

| Ablation | Features | Grouped CV macro-F1 | CV std | Validation F1 | Holdout F1 | World external V2 OOD features >50% | World external any-feature OOD rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| `all_30` | 30 | 0.9925 | 0.0087 | 0.9910 | 1.0000 | 5 | 100.0% |
| `remove_known_shifted` | 24 | 0.9934 | 0.0054 | 0.9910 | 1.0000 | 1 | 79.2% |
| `geometry_only` | 15 | 0.9889 | 0.0143 | 0.9903 | 1.0000 | 1 | 79.2% |
| `coordinate_only` | 15 | 0.9755 | 0.0237 | 0.9668 | 0.9955 | 4 | 100.0% |
| `robust_range_subset` | 25 | 0.9931 | 0.0065 | 0.9917 | 1.0000 | 0 | 33.7% |

The robust-range subset is defined by the source-supported world candidate's mechanical training-range diagnostic, not by DriveNet label accuracy. These results are not a reason to replace V2 without a frozen semantic validation.
