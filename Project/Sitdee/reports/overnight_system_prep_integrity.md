# Overnight system-preparation integrity check

ตรวจเมื่อ 2026-09-07 บน `DESKTOP-ALH6CUB`

## Scope

การทำงานคืนนี้เป็นการอ่าน ตรวจ และสร้าง code/report/demo outputs เท่านั้น ไม่มีคำสั่ง train, model export, raw-data conversion, raw-file rename/delete หรือ webcam access

## Raw dataset hashes

| ไฟล์ | Bytes | SHA-256 ปัจจุบัน | เทียบกับ manifest |
|---|---:|---|---|
| `datasets/raw/anya/official-download/ANN.csv` | 1,991,620,993 | `23be4f23aa2ec6f1904e5b5c13d33f29fa6ce446740667fc374565d30bb38554` | MATCH |
| `datasets/raw/anya/official-download/Random_Forest.csv` | 1,991,620,993 | `d4e5bc74a9e631da7a2cfff10dc44190408245f88c5bf854869bd51a1b7bb673` | MATCH |
| `datasets/raw/anya/official-download/lgbm.csv` | 1,991,620,993 | `dba189298894e159a1803f3cb75b3a446dad327477939cd6e367488ab16d6412` | MATCH |
| `datasets/raw/anya/official-download/xgb.csv` | 1,991,620,993 | `5e701ce9c6f6c763d3e385ec605fe47e76b729101a96d9e976da011cecb42714` | MATCH |
| `datasets/raw/multiposture/data.csv` | 9,327,935 | `d21579200db9a57dd01c24fa6ece961b0be73aa0d5d402ae8e629761c03ebe38` | MATCHES recorded project hash |

The four ANYA rows were rehashed in this run and matched `metadata/anya_hashes.sha256`. No ANYA training was performed.

## Frozen model artifact snapshot

These are current SHA-256 snapshots after the run. The run did not write to these paths; V1/V2/V4 remain isolated.

| Family | Artifact | SHA-256 |
|---|---|---|
| MultiPosture V1 | `outputs/models/multiposture_v1/model.joblib` | `1db2dc98ddbfd8b27ed0827dfc6df320f9c9f714fc71e388c4f0cc09910f0ec9` |
| MultiPosture upper-body V2 | `outputs/models/multiposture_upperbody_v2/model.joblib` | `8c4b77327d91f136ebfe706385668f0d0c61d07cc1eab0ed82a24c2b1aef4070` |
| RGB-Native V4 | `outputs/models/rgb_native_v4/model.joblib` | `af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd` |
| RGB-Native V5 | `outputs/models/rgb_native_v5/NOT_TRAINED.md` | documentation only; no model |

## Acer bundles

- Original `outputs/laptop_dev_bundle.zip` remains unchanged: SHA-256 `b7fc2466ec72fbeb496dfea6dbf29b1c55167634c2cd7a7f30c0c0608a668a5f`
- Intermediate `outputs/laptop_dev_bundle_v2.zip` remains unchanged: SHA-256 `946cb94cfb571e138e13e966f753583c185b126705999d116352ced2f9c8d218`
- Final `outputs/laptop_dev_bundle_v3.zip` includes the final overnight docs/dashboard additions: SHA-256 `9d3432c50039ea49a00b021b358184c763f50af5e506759fa85e891d384b93ab`
- V3 bundle contains 136 entries and the ZIP inspection found no dataset, video, `.venv`, cache or forbidden large-data entry.
- New `outputs/laptop_dev_bundle_v4.zip` contains the RTSP source abstraction, secret-free profiles, diagnostic tools and updated docs: SHA-256 `266bab5fd2aac64631cd93b8c6f3912e422a93cf40a99b36cfcb72d63c97b543`.
- V4 bundle contains 153 entries and the ZIP inspection found no dataset, validation video, `.venv`, cache, credential file or forbidden large-data entry. The earlier bundles remain untouched.
- Integrated v5 bundle `outputs/laptop_dev_bundle_v5.zip` contains 179 entries, size 27.61 MiB, SHA-256 `04d740daef4c8fb879f7ff875e5dd902caca659e0285e62b2e9cd0b9bf221a78`; it contains the RTSP, multi-person and seat-aware code/config/tests but no dataset, video, `.venv`, cache, real `.env` or credential file.

## Decision

**INTEGRITY PASS for this preparation run.** Generated dashboard, test-room plan, benchmark JSON, reports and the new versioned bundle are derived outputs. They do not replace any raw data or existing model.
