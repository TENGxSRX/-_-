# ANYA pseudo-label consensus analysis

> This is a diagnostic of four model-output columns. It is **not** a ground-truth validation and no classifier was trained.

## Source and method

- Official local release: `datasets/raw/anya/official-download/`.
- Files scanned in lockstep: `ANN.csv, Random_Forest.csv, lgbm.csv, xgb.csv`.
- Rows: `605,955`; columns used: `filename`, `clusterid`, `normclusterid`, `Predicted_Label`.
- Bounded-memory pandas chunk reading; the source CSVs were not modified.
- A row is **unanimous** when all four predictions are equal; **strict 3/4** means exactly three models agree and one differs; **2/2 tie** means two models predict each class.

## Pairwise agreement

| Pair | Agreements | Disagreements | Agreement rate | Cohen's kappa |
|---|---:|---:|---:|---:|
| ANN_vs_Random_Forest | 522,585 | 83,370 | 86.242% | 0.723579 |
| ANN_vs_lgbm | 488,551 | 117,404 | 80.625% | 0.613027 |
| ANN_vs_xgb | 488,400 | 117,555 | 80.600% | 0.612596 |
| Random_Forest_vs_lgbm | 525,435 | 80,520 | 86.712% | 0.734646 |
| Random_Forest_vs_xgb | 524,614 | 81,341 | 86.576% | 0.731993 |
| lgbm_vs_xgb | 592,668 | 13,287 | 97.807% | 0.956122 |

## Four-model consensus

- Unanimous: **459,841 / 605,955 (75.887%)**.
- Not unanimous (disagreement): **146,114 (24.113%)**.
- Exact 3-of-4 split: **90,979 (15.014%)**.
- 2-vs-2 ties: **55,135 (9.099%)**.
- 3-of-4-or-unanimous (binary majority excluding ties): **550,820 (90.901%)**.
- Unanimous label counts: `0=243,182`, `1=216,659`.

## Group breakdown

The following tables are diagnostic only. `filename`, `clusterid`, and `normclusterid` are not asserted to be participant IDs or expert labels.

### By filename

| Filename | Rows | Unanimous | Unanimous rate | Exact 3/4 | 2/2 tie | Majority 0 | Majority 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sample_01_front.csv` | 20,341 | 20,187 | 99.243% | 105 | 49 | 20,272 | 20 |
| `sample_02_front.csv` | 18,905 | 5,092 | 26.935% | 7,171 | 6,642 | 614 | 11,649 |
| `sample_04_front.csv` | 17,842 | 17,175 | 96.262% | 345 | 322 | 9,008 | 8,512 |
| `sample_05_front.csv` | 17,335 | 16,637 | 95.973% | 508 | 190 | 16,615 | 530 |
| `sample_06_front.csv` | 19,412 | 16,224 | 83.577% | 741 | 2,447 | 12,631 | 4,334 |
| `sample_07_front.csv` | 18,956 | 9,736 | 51.361% | 5,509 | 3,711 | 303 | 14,942 |
| `sample_08_front.csv` | 16,342 | 15,844 | 96.953% | 349 | 149 | 31 | 16,162 |
| `sample_09_front.csv` | 20,173 | 8,878 | 44.009% | 8,218 | 3,077 | 5,923 | 11,173 |
| `sample_10_front.csv` | 18,942 | 18,215 | 96.162% | 692 | 35 | 18,841 | 66 |
| `sample_11_front.csv` | 20,662 | 17,824 | 86.265% | 2,524 | 314 | 10,527 | 9,821 |
| `sample_13_front.csv` | 19,034 | 15,320 | 80.488% | 1,229 | 2,485 | 883 | 15,666 |
| `sample_15_front.csv` | 17,913 | 7,947 | 44.364% | 6,487 | 3,479 | 6,066 | 8,368 |
| `sample_16_front.csv` | 17,941 | 15,452 | 86.127% | 1,632 | 857 | 2,746 | 14,338 |
| `sample_17_front.csv` | 18,566 | 15,608 | 84.068% | 1,587 | 1,371 | 9,498 | 7,697 |
| `sample_18_front.csv` | 20,517 | 19,702 | 96.028% | 342 | 473 | 62 | 19,982 |
| `sample_19_front.csv` | 18,291 | 5,875 | 32.120% | 9,901 | 2,515 | 108 | 15,668 |
| `sample_20_front.csv` | 19,133 | 18,273 | 95.505% | 664 | 196 | 16,399 | 2,538 |
| `sample_21_front.csv` | 18,270 | 15,062 | 82.441% | 2,684 | 524 | 16,763 | 983 |
| `sample_22_front.csv` | 18,077 | 17,228 | 95.303% | 475 | 374 | 9,888 | 7,815 |
| `sample_23_front.csv` | 18,376 | 15,963 | 86.869% | 2,033 | 380 | 14,621 | 3,375 |
| `sample_24_front.csv` | 19,554 | 13,974 | 71.464% | 2,997 | 2,583 | 14,322 | 2,649 |
| `sample_25_front.csv` | 19,245 | 17,526 | 91.068% | 1,252 | 467 | 15,803 | 2,975 |
| `sample_26_front.csv` | 19,385 | 13,684 | 70.591% | 2,537 | 3,164 | 552 | 15,669 |
| `sample_27_front.csv` | 19,288 | 12,301 | 63.775% | 5,138 | 1,849 | 6,938 | 10,501 |
| `sample_28_front.csv` | 18,922 | 18,921 | 99.995% | 1 | 0 | 18,922 | 0 |
| `sample_30_front.csv` | 19,146 | 14,043 | 73.347% | 2,643 | 2,460 | 12,301 | 4,385 |
| `sample_31_front.csv` | 19,481 | 13,194 | 67.728% | 2,496 | 3,791 | 409 | 15,281 |
| `sample_32_front.csv` | 19,011 | 11,697 | 61.528% | 6,760 | 554 | 97 | 18,360 |
| `sample_33_front.csv` | 19,230 | 14,486 | 75.330% | 959 | 3,785 | 332 | 15,113 |
| `sample_34_front.csv` | 19,403 | 11,096 | 57.187% | 5,200 | 3,107 | 12,904 | 3,392 |
| `sample_35_front.csv` | 19,027 | 10,219 | 53.708% | 6,500 | 2,308 | 16,396 | 323 |
| `sample_36_front.csv` | 19,235 | 16,458 | 85.563% | 1,300 | 1,477 | 4,869 | 12,889 |

### By `clusterid`

| clusterid | Rows | Unanimous | Unanimous rate | Exact 3/4 | 2/2 tie | Majority 0 | Majority 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0` | 129,969 | 87,485 | 67.312% | 24,625 | 17,859 | 4,775 | 107,335 |
| `1` | 475,986 | 372,356 | 78.228% | 66,354 | 37,276 | 270,869 | 167,841 |

### By `normclusterid`

| normclusterid | Rows | Unanimous | Unanimous rate | Exact 3/4 | 2/2 tie | Majority 0 | Majority 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0` | 240,142 | 195,503 | 81.411% | 29,104 | 15,535 | 149,749 | 74,858 |
| `1` | 365,813 | 264,338 | 72.260% | 61,875 | 39,600 | 125,895 | 200,318 |

## Engineering decision

- `Predicted_Label` remains a model output/pseudo-label field, not verified expert ground truth.
- Agreement, including unanimous agreement, cannot establish correctness without the expert-labeled subset or an official cluster-to-TRUE/FALSE mapping.
- These results may support pseudo-label provenance analysis and a future model-output comparison, but not a production ANYA ergonomic classifier in this investigation.
