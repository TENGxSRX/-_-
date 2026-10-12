# ตรวจความพร้อมสำหรับหลายคน

ตรวจเมื่อ 2026-09-07

## สถานะปัจจุบัน

**SINGLE_PERSON_READY** — ระบบปัจจุบันพร้อมสำหรับคนหนึ่งคนต่อกล้องเท่านั้น
ยังไม่ใช่ระบบติดตามหลายคน และไม่มีการระบุตัวบุคคล

## หลักฐานจากโค้ดจริง

1. `create_pose_landmarker` ตั้ง `num_poses=1`
2. `extract_pose_landmarks` คืนค่า `pose_landmarks[0]`
3. `LiveFrameProcessor` ใช้ image/world pose ที่ตำแหน่ง `[0]`
4. V4 รับ landmark ชุดเดียวและให้ prediction เดียว
5. `TemporalRiskEngine` มี history, duration และ cooldown ชุดเดียวต่อ instance
6. telemetry ไม่มี `person_id` หรือ `subject_id`

## คำตอบแบบง่าย

- รองรับกี่คน: หนึ่ง pose ต่อกล้อง
- คนที่สอง: ไม่ได้ประมวลผลแยก และระบบอาจเลือก/สลับคนที่ติดตามโดยไม่แจ้งชัด
- V4: สมมติว่ามี subject เดียว
- เวลาเตือน: เป็น state เดียว ไม่ได้แยกต่อคน
- หลายกล้อง: มักง่ายกว่าใน prototype เพราะแบ่งเป็นโซนหรือ workstation ได้
- สิ่งที่นำกลับมาใช้ได้: landmark schema, core gate, V4, geometry และ
  temporal engine แต่ต้องสร้าง instance แยกต่อคน

## สิ่งที่ต้องเพิ่มในอนาคต

- รับ pose หลายชุดจาก MediaPipe
- เพิ่ม anonymous runtime ID เช่น `PERSON_01`, `PERSON_02`
- มี tracker และจัดการกรณีคนหาย/คนบังกัน/ID สลับ
- มี temporal engine, calibration และ summary แยกต่อคน
- เพิ่ม `camera_id` และ `subject_id` ใน telemetry
- ทดสอบด้วย synthetic multi-pose ก่อนใช้กล้องจริง

ห้ามใช้ใบหน้า ชื่อจริง หรือ biometric identity ระบบควรใช้ anonymous pose
เท่านั้น รายละเอียด schema ทดลองอยู่ที่ `src/posture_ai/multiperson/schema.py`

ยังไม่เลือกจำนวนกล้องจนกว่าจะเห็น floor plan และรู้ FOV/ตำแหน่งติดตั้งจริง
