# ANYA file-level streaming audit

This report was generated from the official Kaggle download. Original CSVs were read sequentially and were not modified.

- Chunk size: 20,000 rows
- Complete scan: no; max rows=1000
- Same schema: True
- Same row count: True
- Same serialized core row sequence excluding final prediction field: True
- Same complete row sequence: True
- Same prediction sequence: True
- Canonical file candidate: **ANN.csv**

| File | Bytes | Rows | Columns | Null cells | Non-finite numeric cells | Duplicate full rows | Duplicate core rows | Parse seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ANN.csv | 1,991,620,993 | 1,000 | 176 | 0 | 0 | 0 | 0 | 2.2 |

## Categorical/prediction observations

### `ANN.csv`
- `filename` unique values: 1; counts: `{'sample_01_front.csv': 1000}`
- Low-cardinality numeric `clusterid` unique values: 1; counts: `{'1': 1000}`
- Low-cardinality numeric `normclusterid` unique values: 2; counts: `{'1': 714, '0': 286}`
- Low-cardinality numeric `Predicted_Label` unique values: 1; counts: `{'0': 1000}`
- Serialized final-field values: `{'0': 1000}`
- Filename blocks: 1; filename transitions: 0
- Frame decreases: 0; adjacent frame increments: 999
- Prediction blocks: 1; prediction transitions: 0

The core-row sequence equality is strong evidence that the four files share the same ordered observations and differ only in the final prediction field. It is still interpreted together with the schema and per-file audit, not from filenames alone.
