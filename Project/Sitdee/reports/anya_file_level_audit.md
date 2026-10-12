# ANYA file-level streaming audit

This report was generated from the official Kaggle download. Original CSVs were read sequentially and were not modified.

- Chunk size: 20,000 rows
- Complete scan: yes
- Same schema: True
- Same row count: True
- Same serialized core row sequence excluding final prediction field: True
- Same complete row sequence: False
- Same prediction sequence: False
- Canonical file candidate: **ANN.csv**

| File | Bytes | Rows | Columns | Null cells | Non-finite numeric cells | Duplicate full rows | Duplicate core rows | Parse seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ANN.csv | 1,991,620,993 | 605,955 | 176 | 0 | 0 | 0 | 0 | 33.0 |
| Random_Forest.csv | 1,991,620,993 | 605,955 | 176 | 0 | 0 | 0 | 0 | 32.5 |
| lgbm.csv | 1,991,620,993 | 605,955 | 176 | 0 | 0 | 0 | 0 | 35.7 |
| xgb.csv | 1,991,620,993 | 605,955 | 176 | 0 | 0 | 0 | 0 | 37.0 |

## Categorical/prediction observations

### `ANN.csv`
- `filename` unique values: 32; counts: `{'sample_01_front.csv': 20341, 'sample_02_front.csv': 18905, 'sample_04_front.csv': 17842, 'sample_05_front.csv': 17335, 'sample_06_front.csv': 19412, 'sample_07_front.csv': 18956, 'sample_08_front.csv': 16342, 'sample_09_front.csv': 20173, 'sample_10_front.csv': 18942, 'sample_11_front.csv': 20662, 'sample_13_front.csv': 19034, 'sample_15_front.csv': 17913, 'sample_16_front.csv': 17941, 'sample_17_front.csv': 18566, 'sample_18_front.csv': 20517, 'sample_19_front.csv': 18291, 'sample_20_front.csv': 19133, 'sample_21_front.csv': 18270, 'sample_22_front.csv': 18077, 'sample_23_front.csv': 18376, 'sample_24_front.csv': 19554, 'sample_25_front.csv': 19245, 'sample_26_front.csv': 19385, 'sample_27_front.csv': 19288, 'sample_28_front.csv': 18922, 'sample_30_front.csv': 19146, 'sample_31_front.csv': 19481, 'sample_32_front.csv': 19011, 'sample_33_front.csv': 19230, 'sample_34_front.csv': 19403, 'sample_35_front.csv': 19027, 'sample_36_front.csv': 19235}`
- Low-cardinality numeric `clusterid` unique values: 2; counts: `{'1': 475986, '0': 129969}`
- Low-cardinality numeric `normclusterid` unique values: 2; counts: `{'1': 365813, '0': 240142}`
- Low-cardinality numeric `Predicted_Label` unique values: 2; counts: `{'0': 322162, '1': 283793}`
- Serialized final-field values: `{'0': 322162, '1': 283793}`
- Filename blocks: 32; filename transitions: 31
- Frame decreases: 31; adjacent frame increments: 605,910
- Prediction blocks: 22,322; prediction transitions: 22,321
### `Random_Forest.csv`
- `filename` unique values: 32; counts: `{'sample_01_front.csv': 20341, 'sample_02_front.csv': 18905, 'sample_04_front.csv': 17842, 'sample_05_front.csv': 17335, 'sample_06_front.csv': 19412, 'sample_07_front.csv': 18956, 'sample_08_front.csv': 16342, 'sample_09_front.csv': 20173, 'sample_10_front.csv': 18942, 'sample_11_front.csv': 20662, 'sample_13_front.csv': 19034, 'sample_15_front.csv': 17913, 'sample_16_front.csv': 17941, 'sample_17_front.csv': 18566, 'sample_18_front.csv': 20517, 'sample_19_front.csv': 18291, 'sample_20_front.csv': 19133, 'sample_21_front.csv': 18270, 'sample_22_front.csv': 18077, 'sample_23_front.csv': 18376, 'sample_24_front.csv': 19554, 'sample_25_front.csv': 19245, 'sample_26_front.csv': 19385, 'sample_27_front.csv': 19288, 'sample_28_front.csv': 18922, 'sample_30_front.csv': 19146, 'sample_31_front.csv': 19481, 'sample_32_front.csv': 19011, 'sample_33_front.csv': 19230, 'sample_34_front.csv': 19403, 'sample_35_front.csv': 19027, 'sample_36_front.csv': 19235}`
- Low-cardinality numeric `clusterid` unique values: 2; counts: `{'1': 475986, '0': 129969}`
- Low-cardinality numeric `normclusterid` unique values: 2; counts: `{'1': 365813, '0': 240142}`
- Low-cardinality numeric `Predicted_Label` unique values: 2; counts: `{'0': 324652, '1': 281303}`
- Serialized final-field values: `{'0': 324652, '1': 281303}`
- Filename blocks: 32; filename transitions: 31
- Frame decreases: 31; adjacent frame increments: 605,910
- Prediction blocks: 22,622; prediction transitions: 22,621
### `lgbm.csv`
- `filename` unique values: 32; counts: `{'sample_01_front.csv': 20341, 'sample_02_front.csv': 18905, 'sample_04_front.csv': 17842, 'sample_05_front.csv': 17335, 'sample_06_front.csv': 19412, 'sample_07_front.csv': 18956, 'sample_08_front.csv': 16342, 'sample_09_front.csv': 20173, 'sample_10_front.csv': 18942, 'sample_11_front.csv': 20662, 'sample_13_front.csv': 19034, 'sample_15_front.csv': 17913, 'sample_16_front.csv': 17941, 'sample_17_front.csv': 18566, 'sample_18_front.csv': 20517, 'sample_19_front.csv': 18291, 'sample_20_front.csv': 19133, 'sample_21_front.csv': 18270, 'sample_22_front.csv': 18077, 'sample_23_front.csv': 18376, 'sample_24_front.csv': 19554, 'sample_25_front.csv': 19245, 'sample_26_front.csv': 19385, 'sample_27_front.csv': 19288, 'sample_28_front.csv': 18922, 'sample_30_front.csv': 19146, 'sample_31_front.csv': 19481, 'sample_32_front.csv': 19011, 'sample_33_front.csv': 19230, 'sample_34_front.csv': 19403, 'sample_35_front.csv': 19027, 'sample_36_front.csv': 19235}`
- Low-cardinality numeric `clusterid` unique values: 2; counts: `{'1': 475986, '0': 129969}`
- Low-cardinality numeric `normclusterid` unique values: 2; counts: `{'1': 365813, '0': 240142}`
- Low-cardinality numeric `Predicted_Label` unique values: 2; counts: `{'0': 296454, '1': 309501}`
- Serialized final-field values: `{'0': 296454, '1': 309501}`
- Filename blocks: 32; filename transitions: 31
- Frame decreases: 31; adjacent frame increments: 605,910
- Prediction blocks: 16,498; prediction transitions: 16,497
### `xgb.csv`
- `filename` unique values: 32; counts: `{'sample_01_front.csv': 20341, 'sample_02_front.csv': 18905, 'sample_04_front.csv': 17842, 'sample_05_front.csv': 17335, 'sample_06_front.csv': 19412, 'sample_07_front.csv': 18956, 'sample_08_front.csv': 16342, 'sample_09_front.csv': 20173, 'sample_10_front.csv': 18942, 'sample_11_front.csv': 20662, 'sample_13_front.csv': 19034, 'sample_15_front.csv': 17913, 'sample_16_front.csv': 17941, 'sample_17_front.csv': 18566, 'sample_18_front.csv': 20517, 'sample_19_front.csv': 18291, 'sample_20_front.csv': 19133, 'sample_21_front.csv': 18270, 'sample_22_front.csv': 18077, 'sample_23_front.csv': 18376, 'sample_24_front.csv': 19554, 'sample_25_front.csv': 19245, 'sample_26_front.csv': 19385, 'sample_27_front.csv': 19288, 'sample_28_front.csv': 18922, 'sample_30_front.csv': 19146, 'sample_31_front.csv': 19481, 'sample_32_front.csv': 19011, 'sample_33_front.csv': 19230, 'sample_34_front.csv': 19403, 'sample_35_front.csv': 19027, 'sample_36_front.csv': 19235}`
- Low-cardinality numeric `clusterid` unique values: 2; counts: `{'1': 475986, '0': 129969}`
- Low-cardinality numeric `normclusterid` unique values: 2; counts: `{'1': 365813, '0': 240142}`
- Low-cardinality numeric `Predicted_Label` unique values: 2; counts: `{'0': 295633, '1': 310322}`
- Serialized final-field values: `{'0': 295633, '1': 310322}`
- Filename blocks: 32; filename transitions: 31
- Frame decreases: 31; adjacent frame increments: 605,910
- Prediction blocks: 16,050; prediction transitions: 16,049

The core-row sequence equality is strong evidence that the four files share the same ordered observations and differ only in the final prediction field. It is still interpreted together with the schema and per-file audit, not from filenames alone.
