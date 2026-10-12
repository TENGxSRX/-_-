# CCTV Feature Experiments

## สถานะ

รอบนี้เป็นการเตรียม schema และ evaluation gate ยังไม่มีการ train candidate CCTV_V5 เพราะไม่มี trusted multi-person CCTV labels

## Feature families

| Schema | สถานะ | รายละเอียด |
|---|---|---|
| `FEATURE_V4_15` | FROZEN BASELINE | normalized MediaPipe World Landmark features ของ V4; TUP/TLB เท่านั้น |
| `MULTIPOSTURE_126_HYBRID_BASELINE` | EXISTING BASELINE | MultiPosture direction model; ไม่ใช่ CCTV RGB semantic guarantee |
| `FEATURE_CCTV_V5_A` | DESIGN ONLY | hip-centered coordinates + body-relative geometry; ยังไม่ fit/เลือกจาก accuracy |
| `FEATURE_CCTV_V5_B` | DESIGN ONLY | shoulder-axis/torso normalized representation; ต้องพิสูจน์ว่า z/axis มีความหมายร่วมกันก่อน |
| `CCTV_V5_BINARY` | BLOCKED | NORMAL/NON_NORMAL labels แบบ trusted ยังไม่พอ |
| `CCTV_V5_MULTICLASS` | BLOCKED | TLF/TLL/TLR จาก CCTV ที่ join ได้ยังไม่พอ |

## Canonical representation ที่เสนอ

```text
PoseLandmarks
→ explicit coordinate_type
→ hip midpoint translation (เมื่อ source รองรับ)
→ optional scale/torso normalization (ต้องมีหลักฐาน/ablation)
→ relative torso/head/shoulder geometry
→ feature_schema_version
→ model only after grouped validation
```

ห้ามทำ arbitrary transform เพื่อให้คะแนนดีขึ้นบน DriveNet และห้าม clip feature เพื่อซ่อน domain shift

## Metrics ที่ต้องใช้ในอนาคต

- subject/video/session grouped macro-F1 และ balanced accuracy
- cross-dataset และ cross-view performance
- per-class recall, confusion matrix
- core valid rate และ throughput ที่ 3 คน
- frame age, inference latency และ temporal switch rate

## Engineering decision

เริ่มเก็บ controlled CCTV data ที่มีหลายคน/หลายที่นั่ง/หลายระยะด้วย human labels ก่อน แล้วค่อย freeze schema และ train candidate รุ่นใหม่ การเพิ่ม dataset ที่ label/coordinate ไม่เท่ากันจะไม่ถูกถือว่าเป็น improvement โดยอัตโนมัติ
