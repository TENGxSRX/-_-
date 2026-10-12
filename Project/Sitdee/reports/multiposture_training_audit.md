# MultiPosture training audit

## Dataset integrity

The local official Zenodo file is `datasets/raw/multiposture/data.csv`: 4,794 rows, 102 columns, 13 subjects, 99 coordinate columns, no missing/non-finite feature cells, no malformed rows, and no exact duplicate rows. Its local MD5 matches the published Zenodo MD5 (`7efbc94d653acf32eefecec3ff26d6c6`); local SHA-256 is `d21579200db9a57dd01c24fa6ece961b0be73aa0d5d402ae8e629761c03ebe38`.

The five upper-body labels were trained: `TLB`, `TLF`, `TLL`, `TLR`, `TUP`. The seven lower-body labels were audited but not included in this first model. The CSV itself is authoritative for the local run; it differs from the Zenodo prose count of 4,800 frames.

## Leakage controls

- Group key: `subject`.
- Train subjects: `1, 10, 11, 12, 13, 2, 3, 5, 7`.
- Validation subjects: `4, 9`.
- Final test subjects: `6, 8`.
- Model selection: five-fold `GroupKFold` over training subjects only.
- Validation and final test subjects were not used for model selection.
- The final test was evaluated once after refitting the selected track on train plus validation subjects.

## Selected result

The selected track was `HistGradientBoosting` with `hybrid` features (126 features), chosen by grouped-CV macro F1, then balanced accuracy, then accuracy. Grouped-CV macro F1 was 0.9891 ± 0.0060; validation macro F1 was 0.9910; final subject-held-out test macro F1 was 1.0000.

These metrics are strong for this local release but are not evidence of generalization to new cameras, environments, or people beyond the 13 recorded subjects. The final test has only two subjects and should not be presented as real-camera validation.
