# MultiPosture upper-body V2 training

Experimental separate model family. V1 was not overwritten.

- Selected model/config: `HistGradientBoosting` + `hybrid` (30 features)
- Core required landmarks: `nose, left_shoulder, right_shoulder, left_hip, right_hip`
- Grouped CV macro F1: `0.9925 ± 0.0087`
- Validation macro F1: `0.9910`
- Final subject-held-out macro F1: `1.0000`
- Train subjects: `1, 10, 11, 12, 13, 2, 3, 5, 7`
- Validation subjects: `4, 9`
- Test subjects: `6, 8`
- Export: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\outputs\models\multiposture_upperbody_v2`
