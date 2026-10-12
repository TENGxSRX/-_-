# Baseline ก่อนเก็บข้อมูล RGB แบบมีคนกำกับ

วันที่ตรวจ: 2026-09-09  
เครื่อง: `DESKTOP-ALH6CUB`  
โครงการ: `office-posture-datasets`

## สถานะสรุป

`READY_FOR_CONTROLLED_DATA_COLLECTION`

สถานะนี้หมายถึงซอฟต์แวร์พร้อมให้เก็บข้อมูลใหม่แบบมีมนุษย์เลือกป้ายกำกับ
แต่ยังไม่มีข้อมูลใหม่ที่ตรวจสอบแล้วสำหรับฝึก V5 จึงยัง **ไม่ฝึก V5**

| ส่วน | สถานะที่ตรวจได้ |
|---|---|
| RTSP source abstraction | `RTSP_SOFTWARE_READY` |
| AI Lab | `RTSP_AI_LAB_READY_FOR_LIVE_VIEW` |
| V4 | Frozen, `TUP/TLB` เท่านั้น |
| Geometry | ใช้เป็นตัวชี้วัดแยกจาก AI |
| การเก็บข้อมูลควบคุม | พร้อมใช้ แต่ยังไม่มี sample ใหม่ |
| V5 candidate | ยังไม่ train |
| กล้อง Hikvision จริงในรอบเตรียมนี้ | ยังไม่เปิด/ยังไม่ validate |
| Raw video | ปิดเป็นค่าเริ่มต้น |
| ANYA | ไม่แตะและไม่ train |

## V4 ที่เป็นแหล่งอ้างอิง

- artifact: `outputs/models/rgb_native_v4/`
- model: Logistic Regression
- input: MediaPipe Pose World Landmarks ผ่าน schema `normalized_world`
- core landmarks: nose, left/right shoulder, left/right hip
- feature count: 15
- labels: `TUP` (Upright), `TLB` (Lean backward)
- V4 confidence/probabilities ถูกเก็บแยกจากป้ายกำกับมนุษย์
- Forward/Left/Right เป็น `GEOMETRY INDICATORS` เชิงทดลอง ไม่ใช่คลาส AI ของ V4

V4 เป็น pilot ที่ train จาก DriveNet และไม่ใช่หลักฐานว่า generalize กับทุก
กล้อง/ทุกห้องได้ การเก็บรอบนี้จึงใช้เพื่อวัดความทนทานของระบบและเตรียมข้อมูล
สำหรับการพิจารณา V5 เท่านั้น

## สิ่งที่มีอยู่แล้วและนำกลับมาใช้

`VideoSource` รองรับ USB, video file และ RTSP/profile โดยแยก health ออกจาก
posture state มี reconnect จำกัด, backoff, stale detection, low-latency buffer
request และ latest-frame buffer แบบ bounded

`LiveFrameProcessor` เดิมยังเป็นเจ้าของ pipeline:

```text
frame → MediaPipe image/world landmarks → core gate
      → V4 15 features → frozen V4
      → geometry → temporal risk → telemetry/UI
```

AI Lab ใหม่ห่อ pipeline นี้โดยตั้งค่า single-person เป็นค่าเริ่มต้น และเพิ่ม
source health card พร้อมบันทึก session แบบไม่เก็บวิดีโอ

## กติกาข้อมูลใหม่

ข้อมูลที่เครื่องมือใหม่จะเขียนอยู่ใต้ `data/controlled_rgb_v5/` ซึ่งแยกจาก
`datasets/raw/` โดยมี `samples.csv`, `rejections.jsonl` และ session metadata
รายครั้ง

- ป้าย `human_label` ได้จากการกด `1` หรือ `2` เท่านั้น
- `1 = TUP`, `2 = TLB`
- `v4_prediction` เป็นเพียงผลทดลองของโมเดล ไม่ใช่ ground truth
- ต้องผ่าน pose/core/finite-feature gates ก่อนจึงจะเป็น accepted sample
- เฟรมที่ไม่ผ่านเก็บเหตุผลเพื่อ audit แต่ไม่ใช้เป็น sample ฝึก
- ไม่บันทึก raw video โดยค่าเริ่มต้น
- ใช้รหัสผู้เข้าร่วมไม่ระบุตัวตน เช่น `P001`

## สิ่งที่ยังไม่มีสิทธิ์สรุป

ยังไม่มีการรายงาน V4 accuracy บนข้อมูล controlled ใหม่ และยังไม่มี V5
candidate เพราะในรอบนี้ยังไม่ได้เก็บข้อมูล human-labelled ใหม่ การเปิด AI Lab
หรือดู prediction สดไม่ใช่การ validate ความแม่นยำ

## การตรวจที่ทำในรอบนี้

- ตรวจ host/project/path และ artifact ปัจจุบัน
- ตรวจว่าไม่มี `.env`/process RTSP setting ที่พร้อมใช้ใน session นี้ โดยไม่อ่าน
  หรือแสดง secret
- compile ตรวจ syntax ของเครื่องมือใหม่
- ไม่เปิด webcam, ไม่เชื่อม RTSP, ไม่ train model, ไม่แก้ raw data

## คำสั่งที่เตรียมไว้

เปิด AI Lab เมื่อ local RTSP configuration พร้อม:

```powershell
& '.\.venv\Scripts\python.exe' scripts\run_rtsp_ai_lab.py --source rtsp --profile HIKVISION_MAIN --preset RTSP_DESKTOP --max-people 1
```

เก็บคลิป TUP/TLB แบบมีคนกดป้าย:

```powershell
& '.\.venv\Scripts\python.exe' scripts\collect_controlled_posture.py --source rtsp --profile HIKVISION_MAIN --subject-id P001 --clip-id P001_TUP_01
```

การใช้ `HIKVISION_SUB` ทำได้โดยเปลี่ยน profile เท่านั้น ไม่เปลี่ยน schema V4

## Decision

คืนนี้ระบบหยุดที่ `READY_FOR_CONTROLLED_DATA_COLLECTION` อย่างถูกต้อง
ยังไม่สร้างผลฝึก V5 จากข้อมูลเทียม และไม่ถือผล prediction เป็น label มนุษย์
