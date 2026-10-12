# MultiPosture model comparison

Model selection used grouped 5-fold cross-validation on the training participants only. The validation and final test participants were not used to select hyperparameters or the final model.

| Rank | Feature configuration | Model | CV macro F1 mean | CV macro F1 std | CV balanced accuracy mean | CV accuracy mean |
|---:|---|---|---:|---:|---:|---:|
| 1 | hybrid | HistGradientBoosting | 0.9891 | 0.0060 | 0.9859 | 0.9912 |
| 2 | engineered | RandomForest | 0.9879 | 0.0123 | 0.9857 | 0.9883 |
| 3 | hybrid | RandomForest | 0.9865 | 0.0112 | 0.9814 | 0.9887 |
| 4 | normalized | HistGradientBoosting | 0.9829 | 0.0074 | 0.9799 | 0.9835 |
| 5 | engineered | HistGradientBoosting | 0.9825 | 0.0154 | 0.9776 | 0.9867 |
| 6 | raw | HistGradientBoosting | 0.9805 | 0.0079 | 0.9759 | 0.9824 |
| 7 | engineered | GradientBoosting | 0.9788 | 0.0180 | 0.9776 | 0.9811 |
| 8 | raw | RandomForest | 0.9713 | 0.0099 | 0.9684 | 0.9733 |
| 9 | normalized | RandomForest | 0.9705 | 0.0142 | 0.9651 | 0.9718 |
| 10 | normalized | GradientBoosting | 0.9667 | 0.0281 | 0.9598 | 0.9740 |
| 11 | hybrid | GradientBoosting | 0.9666 | 0.0292 | 0.9621 | 0.9753 |
| 12 | hybrid | ExtraTrees | 0.9645 | 0.0209 | 0.9570 | 0.9686 |

Selected model: **HistGradientBoosting** with **hybrid** features.

Optional libraries detected: XGBoost=False, LightGBM=False. Models not available in the environment were not installed merely for this benchmark.
