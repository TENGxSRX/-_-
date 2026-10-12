# Reproducibility and raw-data integrity audit

- Audit timestamp UTC: `2026-09-06T03:16:48.735263+00:00`
- Host: `DESKTOP-ALH6CUB`
- Python: `3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]`
- C: free space at audit: `23.72 GB`
- Available RAM at audit: `1.50 GB`

## Raw-file hashes

| File | Bytes | SHA-256 |
|---|---:|---|
| `data.csv` | 9327935 | `d21579200db9a57dd01c24fa6ece961b0be73aa0d5d402ae8e629761c03ebe38` |
| `ANN.csv` | 1991620993 | `23be4f23aa2ec6f1904e5b5c13d33f29fa6ce446740667fc374565d30bb38554` |
| `lgbm.csv` | 1991620993 | `dba189298894e159a1803f3cb75b3a446dad327477939cd6e367488ab16d6412` |
| `Random_Forest.csv` | 1991620993 | `d4e5bc74a9e631da7a2cfff10dc44190408245f88c5bf854869bd51a1b7bb673` |
| `xgb.csv` | 1991620993 | `5e701ce9c6f6c763d3e385ec605fe47e76b729101a96d9e976da011cecb42714` |
| `LB_1.mp4` | 152696410 | `80f7218aa828bedc3f1fa1f605237a5aeb750794112ab3cfb2b0515f5ada668e` |
| `LB_2.mp4` | 77562475 | `baa2dbb0c1d4dec761389ab4941114e27c61928ac8132c6eae0abcd84ce677ac` |
| `LB_3.mp4` | 156875573 | `670099e4ab5595c72f32eb9e6350388da6d5b9dddeaccdbcc47c0b48b083892b` |
| `MB_2.mp4` | 158716341 | `29c44212543e5c3efc5e2fd12e2b0bf4ac2e57a276b100b578bb9d32784bfc85` |
| `SS_1.mp4` | 153589862 | `f4161e3bb9faf61831422e0d88c253f9746423a86be1deac8f997212a4426f79` |
| `SS_2.mp4` | 162121606 | `f063f2e6b5a7bd8fb56376520835ff852d7b6ed6647e32fcddd48d4e73aa03d2` |
| `SS_3.mp4` | 204932316 | `8bab893c4ad8c6d1546f04a6270e4e6aa74856640eb800b9c66e65d493725196` |

## ANYA prior-hash checks

| File | Matches prior hash |
|---|---|
| `ANN.csv` | `True` |
| `lgbm.csv` | `True` |
| `Random_Forest.csv` | `True` |
| `xgb.csv` | `True` |

The audit only reads and hashes files. It does not modify, rename, delete, or replace anything under `datasets/raw/`. No new large external archive was downloaded because free space is below a comfortable margin for multi-gigabyte experiments.
