# CCTV Multi-Person Collection Protocol

เอกสารนี้เป็นแผนเก็บข้อมูลแบบ controlled เท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์ และไม่อนุญาตให้ใช้ prediction ของระบบเป็น label

## กติกาพื้นฐาน

- ใช้รหัสไม่ระบุตัวตน เช่น `P001`, `P002`
- human operator เป็นผู้เลือก label
- ไม่ใช้ใบหน้า ชื่อครู หรือ biometric identity
- ปิดการบันทึก raw video เป็นค่าเริ่มต้น; บันทึกเฉพาะเมื่อได้รับอนุญาตและกำหนด retention
- เก็บ `camera_id`, `view_id`, `distance_band`, seat/zone และ quality metadata แยกจาก label

## ลำดับการเก็บ

1. คนเดียว 3 คน แยก seat/distance เพื่อเช็ก core landmarks
2. สองคนในเฟรมเดียว โดย label แต่ละคนแยกกัน
3. สามคนในเฟรมเดียว
4. mixed posture เช่น `P001=NORMAL`, `P002=BACKWARD`, `P003=NORMAL`
5. passer-by และ temporary occlusion

แต่ละคลิปควรสั้นและทำซ้ำหลายมุม: front-ish, left/right oblique, elevated, near/far และตำแหน่งซ้าย/ขวาของภาพ

## ต่อหนึ่ง sample/telemetry

ต้องรักษา:

- `subject_id` anonymous
- `human_label_original`
- `label_canonical` เฉพาะเมื่อ mapping ถูกอนุมัติ
- `camera_id`, `view_id`, `distance_band`
- `frame_age_ms`, `inference_time_ms`
- `core_visibility`, `core_valid`, `pose_quality`
- `tracking_state`

ระบบต้องทำเครื่องหมาย `UNCERTAIN`, `POSE_PARTIAL` หรือ `IGNORE` เมื่อ label/pose ไม่ชัด และห้ามฝืนเลือก NORMAL/NON_NORMAL

## เครื่องมือเก็บข้อมูลที่เตรียมไว้

ใช้คำสั่งนี้เฉพาะตอนมีผู้ควบคุมอยู่หน้ากล้องจริง และตั้งค่า local RTSP
secret ไว้แล้ว ค่า `EXPECTED PEOPLE` ต้องเป็นจำนวนคนจริงในฉาก ไม่ใช่ค่า
ที่ระบบตรวจพบ:

```powershell
& '.\.venv\Scripts\python.exe' scripts\collect_controlled_cctv.py `
  --source rtsp --profile HIKVISION_MAIN --max-people 3 `
  --expected-people-count 3 --preset RTSP_CCTV `
  --subject-id P001 --clip-id CCTV_STAGE1_01
```

ในหน้าต่าง preview ให้กด `[`/`]` ตั้ง `EXPECTED PEOPLE`, กด `TAB` หรือ `N`
เลือก `PERSON_XX`, กด `1..5` เลือก `TUP/TLB/TLF/TLL/TLR`, แล้วกด `SPACE` เพื่อเริ่ม/หยุดการบันทึก
เฉพาะ frame ที่ผ่าน core-quality gate ส่วน `0=IGNORE`, `U=UNCERTAIN`,
`R=reset`, `Q/ESC=จบ` จะไม่สร้าง supervised sample ป้ายที่เขียนลงไฟล์
มาจาก operator เท่านั้น และ prediction ของ V4 ถูกเก็บเป็น diagnostic แยกฟิลด์

ไฟล์จะอยู่ใน `data/controlled_cctv_v2/` โดยมี `samples.csv`, เหตุผลของ
frame ที่ไม่ผ่าน, session metadata และ telemetry JSONL; ค่าเริ่มต้นไม่บันทึก
วิดีโอดิบและคำสั่งนี้ไม่ train โมเดล

## ห้ามทำ

- สุ่มแยก frame จากคลิปเดียวกันไป train/test
- ใช้ current V4 prediction เป็น ground truth
- เริ่ม train V5 ก่อนตรวจ label join และ grouping
- ใช้ผู้เข้าร่วมคนเดิมข้าม train/test โดยไม่ตั้งใจ
- เก็บภาพดิบเกินความจำเป็น
