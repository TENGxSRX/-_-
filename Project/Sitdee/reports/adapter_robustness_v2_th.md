# รายงาน Adapter Robustness และ Upper-body V2

## ผลหลัก

V1 adapter เดิมรับได้ 0/900 เพราะบังคับ landmark ทั้ง 33 จุดให้ visibility ≥ 0.5 ไม่ใช่เพราะ MediaPipe หา pose ไม่ได้ โดย MediaPipe ตรวจ pose ได้ 900/900 เฟรม สาเหตุหลักคือเข่า ข้อเท้า ส้นเท้า เท้า และข้อศอก/นิ้วบางส่วนถูกบังจากมุมกล้องด้านข้างของท่านั่ง

V2 เป็น model family แยก ใช้เฉพาะ `nose, left_shoulder, right_shoulder, left_hip, right_hip` และไม่แก้ V1

## ผล policy

| Policy | รับ | ปฏิเสธ | อัตรารับ | quality เฉลี่ย |
|---|---:|---:|---:|---:|
| `strict` | 0 | 900 | 0.0% | 0.674 |
| `core-only` | 900 | 0 | 100.0% | 1.000 |
| `tiered` | 900 | 0 | 100.0% | 1.000 |
| `quality` | 900 | 0 | 100.0% | 1.000 |

แกน core ที่จำเป็นต่อ office posture คือ nose, ไหล่ซ้าย/ขวา และสะโพกซ้าย/ขวา ส่วน ears/elbows เป็น optional และ knees/ankles/feet เป็น low-value สำหรับเป้าหมายนี้ รายละเอียดอยู่ใน `reports/core_sitting_landmarks.md`

## V2 training

- model/config: `HistGradientBoosting` + `hybrid` (30 features)
- grouped-CV macro F1: `0.9925 ± 0.0087`
- holdout macro F1: `1.0000`
- external adapter acceptance: `900/900 (100.0%)`

V1 และ V2 ใช้ holdout แบบแยก subject เดียวกัน แต่ผลนี้ยังไม่ใช่ real-camera generalization

## Semantic และ YOLO decision

CSV จริงมี 102 columns, 99 coordinate columns และชื่อครบ 33 กลุ่ม แต่ metadata ของ Zenodo ระบุ 11 joints จึงยืนยันได้เฉพาะ name/index mapping ไม่ใช่ coordinate-scale/z semantic แบบสมบูรณ์ รายละเอียดอยู่ใน `reports/multiposture_landmark_mapping_verified.md` และ `reports/domain_shift_root_cause.md`

ผล gate: **KEEP MEDIAPIPE as the primary backend; YOLO Pose benchmarking is not justified by this study.**

V2 ทำให้ operational acceptance เหมาะกับ seated occlusion มากขึ้น แต่ real-image semantic compatibility ยัง **NOT VALIDATED** ยังต้องตรวจด้วยวิดีโอ office-view ของผู้ใช้เองเมื่อมี hardware, โดยไม่เปิด webcam ในเครื่องนี้และไม่ฝึก ANYA
