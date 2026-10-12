# ANYA four-file comparison

Status: **COMPLETE from full streaming scan**.

## Verified file facts

All four official CSVs have the same `605,955` row count and the same ordered `176`-column schema. Their byte size is `1,991,620,993` each. The parser used UTF-8 and comma delimiter with 20,000-row chunks.

| File | Full CSV SHA-256 | Core-row SHA-256 (without final prediction field) | Prediction counts `0 / 1` |
|---|---|---|---:|
| `ANN.csv` | `23be4f23aa2ec6f1904e5b5c13d33f29fa6ce446740667fc374565d30bb38554` | `bee0497ecd63d28156bf0265ea61084c8d62e20065dfcafe2333ed6437c45c45` | 322,162 / 283,793 |
| `Random_Forest.csv` | `d4e5bc74a9e631da7a2cfff10dc44190408245f88c5bf854869bd51a1b7bb673` | same | 324,652 / 281,303 |
| `lgbm.csv` | `dba189298894e159a1803f3cb75b3a446dad327477939cd6e367488ab16d6412` | same | 296,454 / 309,501 |
| `xgb.csv` | `5e701ce9c6f6c763d3e385ec605fe47e76b729101a96d9e976da011cecb42714` | same | 295,633 / 310,322 |

The complete row sequence hashes and prediction sequence hashes differ across all files. Pairwise prediction disagreements are:

| Pair | Disagreeing rows |
|---|---:|
| ANN vs Random Forest | 83,370 |
| ANN vs LightGBM | 117,404 |
| ANN vs XGBoost | 117,555 |
| Random Forest vs LightGBM | 80,520 |
| Random Forest vs XGBoost | 81,341 |
| LightGBM vs XGBoost | 13,287 |

## Determination

The files are **the same ordered observations with different final model-prediction outputs**, not four independent labeled datasets. The core-row hash equality is direct content evidence; the prediction disagreement rules out treating their final columns as one common label.

They are not merely different feature subsets or preprocessing variants: the full core row sequence, including the metadata columns before the final field, is identical.

## Canonical file candidate

**CANONICAL FILE CANDIDATE: `ANN.csv`**, only as the deterministic representative of the shared core observation table. It is not a canonical ground-truth label source. Any of the four files can reproduce the shared features; none is approved as the ergonomic target source.

