# Hierarchical posture architecture

## ชั้นตัดสินใจ

```text
pose valid?
  └─ no → UNCERTAIN
activity: SITTING / STANDING
  └─ missing/low confidence → UNCERTAIN
sitting quality: CORRECT / INCORRECT
  └─ missing/low confidence → UNCERTAIN
subtype: FORWARD / BACKWARD / LEFT / RIGHT / MIXED
  └─ missing/low confidence → SITTING_INCORRECT_UNCERTAIN
```

สถานะที่มีใน contract คือ `STANDING`, `SITTING_CORRECT`, `SITTING_INCORRECT_<SUBTYPE>` และ `UNCERTAIN`.

## สถานะจริงของโมเดล

- Activity model: **NOT TRAINED — STANDING_DATA_INSUFFICIENT**
- Sitting quality model: **NOT TRAINED — SITTING_POSTURE_DATA_INSUFFICIENT**
- Subtype model: **NOT TRAINED**; ใช้ Geometry เป็น indicator/diagnostic ได้เท่านั้น
- Temporal engine: มี per-person persistence contract และเปิดเผย raw decision

ทุก track มี state แยกกัน. หากหลักฐานไม่พอ ระบบต้องแสดง `UNCERTAIN` ไม่ยกระดับ V4 prediction ให้เป็น sit/stand label.
