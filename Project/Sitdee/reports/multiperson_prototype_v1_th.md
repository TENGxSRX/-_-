# Multi-Person Posture Monitoring Prototype V1

วันที่: 8 กันยายน 2026  
สถานะซอฟต์แวร์: `MULTI_PERSON_SOFTWARE_PROTOTYPE_READY`  
สถานะฮาร์ดแวร์: `MULTI_PERSON_HARDWARE_NOT_VALIDATED`

เอกสารนี้อธิบาย prototype เชิงสถาปัตยกรรม ไม่ใช่ผลการทดลองจริง และไม่ใช่การวินิจฉัยทางการแพทย์

## 1. เปลี่ยนอะไร

เพิ่มชั้น multi-person แยกจาก runtime เดิม โดยทำงานตามลำดับ:

```text
กล้องหนึ่งตัว
→ MediaPipe Tasks Pose Landmarker
→ pose หลายชุดในเฟรมเดียว
→ anonymous tracker
→ V4 ต่อคน
→ geometry ต่อคน
→ temporal/risk state ต่อคน
→ telemetry และ overlay ต่อคน
```

ระบบรองรับ `PERSON_01`, `PERSON_02`, ... ซึ่งเป็น ID ชั่วคราวของ runtime/session เท่านั้น

## 2. อะไรยังเหมือนเดิม

- V4 ไม่ได้ retrain และไม่ถูกแก้ไข
- V4 ยังมี 2 AI classes เท่านั้น: `TUP` = Upright และ `TLB` = Lean Backward
- Forward, Left และ Right ยังคงเป็น geometry indicators ไม่ใช่ AI classes
- single-person runtime เดิมยังอยู่และไม่ได้ถูกแทนที่
- ไม่มีการใช้ใบหน้า ชื่อบุคคล facial embedding หรือการระบุตัวตนจริง
- ไม่มีการเปลี่ยน raw dataset หรือสร้างข้อมูลฝึกใหม่

## 3. MediaPipe รองรับกี่คนจริง

ติดตั้งและตรวจสอบ MediaPipe `1.0.1` แล้ว พบว่า `mp.solutions` แบบเก่าไม่มี แต่ `mediapipe.tasks.vision.PoseLandmarkerOptions` มี `num_poses` จริง และผลลัพธ์มีรายการ `pose_landmarks` กับ `pose_world_landmarks` แยกตาม pose

CLI กำหนด `--max-people` ได้ `1` ถึง `4` โดยค่าเริ่มต้นคือ `2` ตัว adapter ภายในรับค่าจำนวนเต็มอย่างน้อย `1` แต่ยังไม่มี benchmark ว่า Acer จะประมวลผล realtime ได้กี่คน

โค้ดเดิมยังคงใช้ `num_poses=1` และเลือก pose แรก เพื่อรักษา single-person behavior ส่วน adapter ใหม่จึงเป็นคนละเส้นทาง

## 4. PERSON_01 tracking ทำงานอย่างไร

tracker ใช้ระยะห่างของตำแหน่งกึ่งกลางร่างกายที่ normalize แล้ว ซึ่งคำนวณจากจุดไหล่และสะโพก หรือใช้กึ่งกลาง bbox เมื่อจำเป็น การจับคู่เป็น nearest-neighbor แบบ deterministic และมีค่า:

- `max_association_distance`: 0.20
- `max_misses`: 3 เฟรม
- `reacquisition_window_seconds`: 1.0 วินาที

สถานะของ track คือ `NEW`, `ACTIVE`, `TEMPORARILY_LOST`, `EXPIRED` เมื่อหายไปชั่วคราว ID เดิมยังรอการกลับมา เมื่อหมดอายุ คนใหม่จะได้เลขใหม่ ไม่ reuse ID เก่าใน session เดียวกัน

## 5. ID อยู่รอดเมื่อลำดับ detector เปลี่ยนหรือไม่

อยู่รอดในกรณี synthetic test ที่ตำแหน่งยังใกล้เดิม แม้ detector จะคืนลำดับ Person B ก่อน Person A เพราะจับคู่ด้วยตำแหน่ง ไม่ใช่ลำดับรายการ

ข้อจำกัด: หากคนเดินตัดกันมาก บังกันนาน หรือขยับไกลเกิน threshold อาจเกิด track ใหม่ได้ นี่เป็น runtime association ไม่ใช่การยืนยันตัวบุคคลจริง

## 6. V4 ต่อคน

แต่ละ subject ที่ core landmarks ผ่านจะส่ง World Landmarks เข้า `RGBNativeV4Runner` เดิม โดยไม่เปลี่ยน feature family, feature order หรือ model artifact ผลลัพธ์ของแต่ละคนเก็บ label และ probabilities ของตนเอง

ถ้า subject คนหนึ่งประมวลผลไม่ได้ จะไม่ทำให้ subject คนอื่นหยุดทำงาน คนที่ข้อมูลหายจะได้สถานะ `POSE_LOST` แยกของตนเอง

## 7. Geometry และ calibration ต่อคน

geometry ถูกคำนวณแยกเป็นรายคน โดยเก็บ Forward, Left, Right, shoulder tilt และตัวชี้วัดศีรษะ/ลำตัวในชุดเดียวกับ engine เดิม แต่มีข้อความกำกับว่าเป็น experimental indicators

ระบบไม่แชร์ neutral baseline ของคนหนึ่งให้คนอื่น ถ้ามี baseline จะเก็บใน dictionary ตาม `subject_id`; ถ้ายังไม่มี จะใช้ absolute geometry fallback และแสดงว่า `neutral_baseline_applied=false`

## 8. Temporal state ต่อคน

ทุก subject มี `TemporalRiskEngine` ของตัวเอง จึงมี warning timer, poor-posture duration, recovery และ `POSE_LOST` แยกกัน ตัวอย่างเช่น PERSON_01 อาจเป็น `WARNING` ขณะที่ PERSON_02 ยังเป็น `NORMAL` หรือ `POSE_LOST` โดยไม่รับ timer ของกันและกัน

ค่าระยะเวลาเป็น engineering/product settings ไม่ใช่ medical threshold

## 9. Telemetry และ dashboard model

เพิ่ม event แบบ JSON ที่มี `timestamp_seconds`, `camera_id`, `subjects[]` และแต่ละ subject มี:

- `subject_id`
- `core_valid` และ `pose_quality`
- V4 prediction/probabilities
- geometry indicators
- state และ duration
- tracking state และ last seen time

มี `legacy_single_person_to_event()` สำหรับห่อ record เก่าโดยไม่แก้ field เดิม และ `SessionAccumulator` เดิมยังไม่ถูกแก้

overlay และ `dashboard_cards()` แยกการแสดงผลเป็น AI POSTURE, GEOMETRY, STATE, DURATION และ POSE QUALITY ต่อคน โดยไม่ใช้ชื่อจริง

## 10. TEST_FIXTURE

ไฟล์ `fixtures/multiperson_prototype_v1.json` เป็น synthetic `TEST_FIXTURE` จากผู้สังเคราะห์สองตำแหน่งเท่านั้น ไม่ใช่ภาพคนจริง ไม่ใช่ dataset และไม่ใช่หลักฐานด้าน accuracy ใช้สาธิตโครงสร้าง telemetry/dashboard เท่านั้น

## 11. LOW_SPEC compatibility

ค่า `LOW_SPEC` ใช้ capture เป้าหมาย 640×480 และประมวลผลทุกเฟรมที่ 3 preview ยังสามารถแสดงทุกเฟรมได้ แต่เฟรมที่ข้ามจะถูกทำเครื่องหมาย `inference_skipped=true` และไม่ถูกนับเป็น inference ใหม่ timestamp ของเฟรมจริงยังคงใช้ ไม่ได้นำจำนวนเฟรมที่ข้ามไปคูณเวลา

ยังไม่มีการอ้างว่า realtime บน Acer เพราะไม่ได้ benchmark กล้องจริงแบบหลายคนโดยอัตโนมัติ

## 12. ผลทดสอบ

- MediaPipe API smoke test: ขอ `num_poses=3` สำเร็จ และรัน blank image ได้ โดยคืน 0 pose ตามที่คาดจากภาพว่าง
- tests ใหม่ของ multi-person: ผ่าน `10/10`
- focused regression suite: ผ่าน `42` รายการ, skip `3`, fail `1` เพราะไม่มี `datasets/raw/multiposture/data.csv` ใน lightweight checkout
- full existing suite: ผ่าน `95`, fail `9`, skip `3`; failures เป็นไฟล์ dataset/model/manifest ของงานเดิมที่ไม่ได้อยู่ใน checkout นี้ ไม่ใช่ multi-person code
- laptop health check: `LAPTOP_DEV_READY`
- กล้องจริง: ยังไม่ทดสอบอัตโนมัติ

## 13. ไฟล์ที่เพิ่มหรือแก้ไข

- `src/posture_ai/multiperson/schema.py`
- `src/posture_ai/multiperson/tracking.py`
- `src/posture_ai/multiperson/mediapipe_tasks.py`
- `src/posture_ai/multiperson/runtime.py`
- `src/posture_ai/multiperson/telemetry.py`
- `src/posture_ai/multiperson/overlay.py`
- `src/posture_ai/multiperson/demo.py`
- `src/posture_ai/multiperson/__init__.py`
- `scripts/run_multiperson_demo.py`
- `scripts/multiperson_fixture_demo.py`
- `tests/test_multiperson_prototype_v1.py`
- `reports/multiperson_mediapipe_audit_th.md`
- `fixtures/multiperson_prototype_v1.json`

## 14. Acer และการทดสอบเต็มเครื่อง

ผล health check ล่าสุด: free disk ประมาณ `19.42 GiB`, RAM ประมาณ `7.86 GiB`, 4 logical cores, Python `3.12.10`, Windows 10

ไม่ดาวน์โหลด dataset/model/video เพิ่ม และไม่เปิดกล้องอัตโนมัติ เพื่อไม่ใช้พื้นที่และ CPU เกินความจำเป็น

ต้องทดสอบบนเครื่องที่มีกล้องจริงหรือไฟล์วิดีโอหลายคนเพื่อยืนยัน:

- จำนวนคนที่ตรวจได้จริง
- ID เมื่อคนเดินตัดกันหรือถูกบัง
- FPS และ latency ที่ `--max-people 2/3/4`
- ความหมายของ left/right ตามทิศกล้องจริง
- ความเหมาะสมของ camera placement ในห้อง instructor

## 15. ขั้นตอนถัดไปหนึ่งอย่าง

บนเครื่องที่มีกล้อง ให้รัน bounded manual test นี้โดยระบุ source อย่างชัดเจน:

```powershell
python scripts\run_multiperson_demo.py --source 0 --max-people 2 --preset LOW_SPEC --max-frames 300 --telemetry outputs\multiperson_test_fixture.jsonl
```

ให้ถือผลนี้เป็น hardware validation log เท่านั้น ไม่ใช่ accuracy claim และไม่ใช่ medical diagnosis
