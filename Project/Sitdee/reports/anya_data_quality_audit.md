# ANYA data quality audit

Status: **COMPLETE from full bounded-memory scan**.

| Check | Result |
|---|---:|
| Rows per file | 605,955 |
| Columns per file | 176 |
| Null cells | 0 in all four files |
| Non-finite numeric cells | 0 in all four files |
| Constant numeric columns | none detected |
| Exact serialized duplicate rows | 0 in all four files |
| Raw CSV mutation | none; hashes still match `metadata/anya_hashes.sha256` |

No rows were removed and no raw CSV was rewritten. Near-duplicate distances, personal identity inference, and label-noise inference were not claimed because they cannot be established from these columns alone. The main quality limitation is semantic: the release contains prediction outputs and cluster IDs but no verified target column.

