# MultiPosture training summary

This is an engineering experiment, not a clinical validation. Labels remain the original source labels.

- Source rows: 4794; source columns: 102
- Source participants: 13; duplicate rows: 0; non-finite feature cells: 0
- Grouped split: train subjects `1, 10, 11, 12, 13, 2, 3, 5, 7`; validation subjects `4, 9`; final test subjects `6, 8`.
- Selection CV: GroupKFold with 5 folds over the 9 training subjects.
- Selected feature configuration: `hybrid` (126 features).
- Selected model: `HistGradientBoosting`.
- Export: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\outputs\models\multiposture_v1\model.joblib`.

## Validation metrics

- Accuracy: 0.9893
- Balanced accuracy: 0.9895
- Macro precision / recall / F1: 0.9925 / 0.9895 / 0.9910
- Weighted F1: 0.9893

## Final test metrics

- Accuracy: 1.0000
- Balanced accuracy: 1.0000
- Macro precision / recall / F1: 1.0000 / 1.0000 / 1.0000
- Weighted F1: 1.0000

The final test metrics are held out by participant and are reported once after model selection. They should not be interpreted as performance on unseen camera data.
