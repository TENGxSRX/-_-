# ANYA split strategy

Status: **PRE-REGISTERED; no supervised split used because no verified target exists**.

## Grouping decision

The release has 32 recording-like `filename` groups and no participant/session/video identifier. `filename` is the minimum defensible group key for a future experiment; it does not prove participant independence.

The 32 observed group values are: `sample_01_front.csv`, `sample_02_front.csv`, `sample_04_front.csv`, `sample_05_front.csv`, `sample_06_front.csv`, `sample_07_front.csv`, `sample_08_front.csv`, `sample_09_front.csv`, `sample_10_front.csv`, `sample_11_front.csv`, `sample_13_front.csv`, `sample_15_front.csv`, `sample_16_front.csv`, `sample_17_front.csv`, `sample_18_front.csv`, `sample_19_front.csv`, `sample_20_front.csv`, `sample_21_front.csv`, `sample_22_front.csv`, `sample_23_front.csv`, `sample_24_front.csv`, `sample_25_front.csv`, `sample_26_front.csv`, `sample_27_front.csv`, `sample_28_front.csv`, `sample_30_front.csv`, `sample_31_front.csv`, `sample_32_front.csv`, `sample_33_front.csv`, `sample_34_front.csv`, `sample_35_front.csv`, `sample_36_front.csv`.

## Future split rule

After a verified target is supplied, use a deterministic group split of approximately 70% train, 15% validation, and 15% final test, followed by grouped cross-validation inside the training groups. The exact group allocation must be recorded before model selection. The final test cannot be used for tuning, feature choice, threshold choice, or model choice. Random row metrics, if computed, are leakage diagnostics only.

No train/validation/test IDs are claimed in this run and no ANYA metric is reported.

