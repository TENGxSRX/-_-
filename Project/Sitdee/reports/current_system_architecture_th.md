# สถาปัตยกรรมระบบปัจจุบัน

ตรวจเมื่อ 2026-09-07 รายงานนี้อธิบายของที่มีอยู่จริง ไม่ได้ retrain หรือแทนที่
V1/V2/V4

## เส้นทาง live

```text
ภาพจากกล้อง
  → run_webcam_session
  → MediaPipe Pose
  → image landmarks + world landmarks
  → ตรวจ 5 จุดหลักและคุณภาพท่า
  → แปลงเป็น V4 features
  → โมเดล V4 ที่ freeze ไว้
  → geometry engine
  → neutral calibration
  → geometry indicators
  → temporal risk engine
  → telemetry / summary / หน้าจอ
```

V4 ใช้ `TUP = Upright` และ `TLB = Lean Backward` เท่านั้น ส่วน Forward,
Left, Right เป็น geometry indicators ทดลอง ไม่ใช่คลาส AI ของ V4

## ไฟล์และหน้าที่

| ขั้นตอน | ไฟล์/ฟังก์ชัน | หน้าที่ |
|---|---|---|
| เริ่มโปรแกรม | `scripts/run_webcam_demo.py:main` | รับคำสั่งกล้อง, validation, calibration และ preset |
| อ่านกล้อง | `src/posture_ai/inference/webcam_v1.py:run_webcam_session` | อ่านภาพจากกล้องที่ผู้ใช้ระบุ และ cleanup |
| ตั้ง MediaPipe | `src/posture_ai/inference/pipeline.py:create_pose_landmarker` | ใช้ Pose Landmarker แบบ video และตั้ง `num_poses=1` |
| แปลงจุด | `extract_pose_result`, `PoseLandmarks.from_any` | รักษาลำดับจุดร่างกาย 33 จุด |
| ตรวจ input | `framing_assessment`, `evaluate_visibility_policy` | ตรวจ nose, ไหล่ 2 ข้าง และสะโพก 2 ข้าง |
| V4 adapter | `RGBNativeV4Runner`, `build_rgb_native_v4_features` | สร้าง schema 15 features และปฏิเสธ schema ผิด |
| โมเดล | `outputs/models/rgb_native_v4/model.joblib` | โมเดลสองคลาสที่ freeze แล้ว |
| geometry | `calculate_geometry` | คำนวณทิศทางลำตัว ศีรษะ และไหล่ |
| calibration | `NeutralBaselineCalibrator`, `relative_geometry` | เก็บท่านั่งกลางของ session แล้วเทียบแบบ relative |
| temporal | `TemporalRiskEngine` | เก็บประวัติ prediction, เวลา และ warning ของคนหนึ่งคน |
| บันทึกผล | `LiveSessionAccumulator`, `write_live_outputs` | เขียน JSONL และ session summary |
| หน้าจอ | `draw_live_overlay` | แยก AI posture, geometry, visibility และ risk state |

## เส้นทาง source เดียวสำหรับ RTSP ในอนาคต

เตรียม source abstraction เพิ่มโดยไม่แก้เส้นทาง webcam เดิมที่ใช้ตรวจ V4:

```text
USB / video file / RTSP URL หรือ profile
  → parse_source_spec
  → VideoSource
  → health + reconnect + stale protection
  → run_source_posture_session
  → MediaPipe + V4 ที่ freeze + geometry + temporal
  → telemetry ที่มี metadata ของ source
```

ไฟล์สำคัญ:

| ส่วน | ไฟล์/ฟังก์ชัน | หน้าที่ |
|---|---|---|
| แยกชนิด source | `video_source.py:parse_source_spec` | รองรับ USB, file, RTSP URL และ Hikvision profile ที่ไม่มี secret |
| อ่านภาพ | `VideoSource` | เปิด/อ่าน/ปิด source ที่ผู้ใช้สั่งเท่านั้น |
| health | `SourceHealth` | นับ frame, decode failure, FPS, resolution, อายุ frame และ reconnect |
| reconnect | `VideoSource.read` | retry/backoff แบบมีเพดาน ไม่ให้หลุดไปเป็น posture warning |
| latest frame | `AsyncVideoSourceReader` | buffer แค่ frame ล่าสุดและ mark stale ได้ |
| runtime | `stream_runtime.py:run_source_posture_session` | ใช้ V4/geometry/temporal เดิมกับ source เดียวกัน |
| diagnostic | `scripts/check_rtsp_source.py` | ตรวจ source แบบจำกัดเวลา ไม่ทำ AI ไม่อัดวิดีโอ ไม่เปิด server |

เส้นทางนี้ยังไม่ได้ทดสอบกับ Hikvision จริง เพราะกล้องยังไม่พร้อม และไม่ได้
แก้ route, firewall, NAT หรือ IP forwarding

## เส้นทาง multi-person/seat-aware ที่ merge แล้ว

```text
VideoSource
  → frame + source health
  → MediaPipe num_poses=max_people
  → anonymous PERSON_XX tracker
  → generic person-centric หรือ optional seat mode
  → activity SITTING/STANDING/UNKNOWN
  → เมื่อ SITTING: V4 เดิม + geometry + temporal แยกต่อคน
  → telemetry ที่ตัด secret + dashboard
```

รองรับ `max_people` ตั้งแต่ 1 ถึง 4 ระบบยังไม่ใช้ใบหน้า/ชื่อจริง. Activity gate เป็น
`GEOMETRY_EXPERIMENTAL` และยังไม่ใช่ Sit/Stand AI ที่ train แล้ว. Seat mode
ต้องใช้ YAML layout ของห้องจริงที่ได้รับอนุมัติ ตัวอย่างที่มีพิกัดเป็น
`TEST_ROOM_ONLY` เท่านั้น ไม่ใช่พิกัดห้องจริง

## เส้นทาง offline

`src/posture_ai/inference/pipeline.py:infer_source` รองรับ image file,
image directory และ prerecorded video โดยคืนนี้ไม่เปิด webcam source 0

## ขอบเขตสำคัญ

- ระบบ live ปัจจุบันรองรับ pose หนึ่งคนและ temporal state หนึ่งชุด
- geometry/risk เป็นค่าทางวิศวกรรม ไม่ใช่คะแนนสุขภาพหรือการวินิจฉัยโรค
- คืนนี้ไม่มีการ train, export model ใหม่, แก้ raw data, รันกล้อง หรือใส่ขนาดห้องจริง
