# แผนเชื่อม Multi-person จาก Acer เข้ากับ Desktop

สถานะ Desktop ปัจจุบัน: **ยังเป็น single-person runtime**

Acer มีต้นแบบ `MULTI_PERSON_SOFTWARE_PROTOTYPE_READY` แต่โค้ดนั้นยังไม่ได้ถูกนำมารวมใน Desktop รอบนี้ และผมไม่สร้าง tracker ชุดที่สองแข่งกับ Acer โดยเดาเอง

## Contract ที่ Desktop เตรียมไว้

```text
RTSP frame + source health
        ↓
PoseDetection[]
        ↓
anonymous subject layer
  PERSON_01, PERSON_02, ...
        ↓
per-person V4 + geometry + temporal state
        ↓
multi-person telemetry / dashboard
```

แต่ละ frame ควรมีข้อมูล non-secret:

| Field | ความหมาย |
|---|---|
| `source_type` | USB / FILE / RTSP |
| `camera_id` | เช่น `CAM01` |
| `timestamp_seconds` | เวลาจริงของ frame |
| `subjects` | รายการ pose ที่ตรวจพบ |
| `subject_id` | anonymous runtime ID เช่น `PERSON_01` |
| `pose_quality` | คุณภาพ landmark เชิงเทคนิค |
| `v4_result` | ผล V4 ของ subject นั้น |
| `geometry` | ตัวบ่งชี้เชิงเรขาคณิต |
| `temporal_state` | สถานะเตือนของ subject นั้น |

ไม่มีชื่อจริง ใบหน้า หรือ biometric identity ใน contract

## สิ่งที่ต้องตกลงก่อน merge Acer

1. ชื่อ field และ timestamp ต้องตรงกัน
2. ID lifecycle ต้องกำหนดกรณีคนเข้า/ออก/บังกัน/ID switch
3. ต้องมี temporal engine และ neutral calibration แยกต่อ subject
4. source health (`SOURCE_STALE`, `SOURCE_RECONNECTING`) ต้องไม่ถูกแปลงเป็น `POSE_LOST` โดยอัตโนมัติ
5. UI ต้องแยก camera health จาก posture risk
6. ต้องทดสอบ synthetic multi-pose ก่อนต่อกล้องจริง

## Seat-aware ต่อในอนาคต

`configs/seat_zones.example.yaml` เป็น schema เท่านั้น มี `CAM01`, `SEAT_01`, `SEAT_02` และ `roi_polygon` ว่างไว้เพื่อไม่สร้างพิกัดห้องปลอม เมื่อมีภาพกล้องจริงจึงค่อยกรอก polygon ใน image pixels

Seat ID ไม่ผูกกับชื่อครูหรือบุคคล และไม่ควรใช้ ROI เป็นการยืนยันตัวตน

## ขอบเขตคืนนี้

Desktop ยังใช้ V4 สองคลาสเดิมและ single-person processor เดิม โมดูล `PoseSubject`/`MultiPersonFrameResult` ที่มีอยู่เป็น data contract แยก ไม่ได้เปลี่ยน production runtime และไม่ได้อ้างว่า multi-person deployment พร้อมแล้ว
