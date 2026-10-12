# Dataset Label Harmonization

## เป้าหมาย

ทดสอบ taxonomy แบบสองระดับโดยรักษา source label เดิมไว้:

```text
NORMAL
NON_NORMAL
  ├─ BACKWARD
  ├─ FORWARD
  ├─ LEFT
  └─ RIGHT
```

นี่เป็น design/experiment ไม่ใช่ผลว่า dataset ทั้งหมดเข้ากันแล้ว

## Mapping ที่มีหลักฐานเพียงพอ

| Dataset/source label | Canonical | สถานะ |
|---|---|---|
| MultiPosture `TUP` | `NORMAL` | ใช้ได้สำหรับ direction ontology ของโครงการ |
| MultiPosture `TLB` | `NON_NORMAL` + `BACKWARD` | ใช้ได้ใน source ontology |
| MultiPosture `TLF` | `NON_NORMAL` + `FORWARD` | ใช้ได้ใน source ontology |
| MultiPosture `TLL` | `NON_NORMAL` + `LEFT` | ใช้ได้ใน source ontology |
| MultiPosture `TLR` | `NON_NORMAL` + `RIGHT` | ใช้ได้ใน source ontology |
| RGB-native V4 `TUP`/`TLB` | `NORMAL`/`NON_NORMAL` | ใช้ได้เฉพาะ frozen two-class pilot |
| ANYA `Predicted_Label`, `clusterid`, `normclusterid` | ไม่ map | pseudo/unknown; reject |
| SitPose `healthy`, `wrong`, `forward`, ... | ไม่ map อัตโนมัติ | ontology/coordinate ต่างกัน |
| SitStraightNet image names | ไม่ map | ไม่มี verified label join |
| MVSP repository label names | ยังไม่ map | archive/media/license ยังไม่ตรวจ |

## Canonical adapter rule

โค้ด `src/posture_ai/data/canonical.py` จะ raise `UnsupportedLabelError` เมื่อไม่มี mapping ที่อนุมัติ และจะไม่แปลง label โดยอาศัยชื่อที่คล้ายกันเท่านั้น `label_original` ต้องถูกเก็บคู่กับ `label_canonical` เสมอ

## สถานะ hierarchical experiment

ยังไม่ train และยังไม่รายงาน metric ของ NORMAL/NON_NORMAL เพราะการรวม source ที่มีเพียง direction บางส่วนหรือมี coordinate/domain ต่างกันจะทำให้ source confounding สูง การทดลองที่ถูกต้องต้องมี subject/video grouping และมี trusted samples ของแต่ละ class จากมากกว่าหนึ่ง source หรือ controlled CCTV collection

## ข้อห้าม

- ไม่เรียก `NON_NORMAL` ว่าโรคหรือ Office Syndrome
- ไม่ใช้ V4 prediction เป็น human label
- ไม่ใช้ ANYA pseudo-label เป็น ground truth
- ไม่ลบ source label เดิม
