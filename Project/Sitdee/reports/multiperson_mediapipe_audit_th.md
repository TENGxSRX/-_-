# MediaPipe Multi-Person Audit — ANYA / Ergonomic Posture Prototype

วันที่ตรวจสอบ: 8 กันยายน 2026  
เครื่อง: Acer Aspire 4741G, Windows 10, Python 3.12.10

## ผลตรวจสั้น ๆ

MediaPipe ที่ติดตั้งจริงคือ `1.0.1` และใช้ MediaPipe Tasks API รุ่นใหม่ ไม่ใช่ API เก่า `mp.solutions.pose`:

- `mediapipe.__version__`: `1.0.1`
- `mediapipe.solutions`: ไม่มี (`has_solutions=False`)
- `mediapipe.tasks`: มี
- `mediapipe.tasks.vision.PoseLandmarker`: มี
- `mediapipe.tasks.vision.PoseLandmarkerOptions`: มี
- `RunningMode`: `IMAGE`, `VIDEO`, `LIVE_STREAM`

จาก signature ของแพ็กเกจที่ติดตั้งจริง:

```text
PoseLandmarkerOptions(
    base_options,
    running_mode=IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False,
    result_callback=None,
)
```

ดังนั้น API มีตัวเลือก `num_poses` และสามารถออกแบบให้ขอหลาย pose ได้ เช่น `num_poses=2` หรือ `num_poses=3` โดยต้องสร้าง landmarker ด้วย `PoseLandmarkerOptions` ของ Tasks API รุ่นนี้

ผลลัพธ์ `PoseLandmarkerResult` มีโครงสร้างหลัก:

- `pose_landmarks`: รายการของ pose แต่ละคน
- `pose_world_landmarks`: รายการของ World Landmark ของ pose แต่ละคน
- `segmentation_masks`: optional

## สิ่งที่โค้ดปัจจุบันทำ

ไฟล์ `src/posture_ai/inference/pipeline.py` ใน checkout นี้ใช้ Tasks API แล้ว แต่ฟังก์ชัน `create_pose_landmarker` กำหนด `num_poses=1` แบบตายตัว นอกจากนี้ `extract_pose_landmarks` คืนเฉพาะ `result.pose_landmarks[0]` ดังนั้นระบบปัจจุบันจึงเป็น single-person จริง แม้ API จะรองรับผลลัพธ์หลายคน

`src/posture_ai/inference/webcam_v1.py` เป็น runtime ของ V4 สำหรับคนเดียว และใช้ state/calibration/temporal engine ชุดเดียว จึงไม่ควรนำ state ชุดเดิมไปแชร์ให้หลายคนโดยตรง

## สิ่งที่ MediaPipe ไม่ได้ให้มาโดยอัตโนมัติ

การมีผลลัพธ์หลาย pose ไม่ได้หมายความว่า MediaPipe ให้ runtime ID ที่คงที่แก่คนแต่ละคน โค้ดที่ใช้ผลลัพธ์ตามลำดับรายการอาจสลับคนได้เมื่อ detector คืนลำดับใหม่ นอกจากนี้ layer นี้ยังต้องจัดการเองเรื่อง:

- การจับคู่ pose ระหว่างเฟรม
- `PERSON_01`, `PERSON_02` ซึ่งเป็น ID ชั่วคราวของ session
- การหายไปชั่วคราวและการหมดอายุของ track
- bbox/center และ pose quality
- V4 prediction, geometry และ timer แยกต่อคน
- telemetry ที่มีหลาย subject

ไม่ใช้ใบหน้า, facial embedding หรือชื่อบุคคล

## แผน adapter ที่ปลอดภัย

จะคง single-person production path เดิมไว้ และเพิ่ม adapter แยกสำหรับ multi-person ดังนี้:

1. สร้าง `PoseLandmarkerOptions` ด้วย `num_poses=max_people` ตามค่าที่ผู้ใช้ระบุ
2. อ่าน `pose_landmarks[i]` คู่กับ `pose_world_landmarks[i]` ทุก pose ที่คืนมา
3. ตรวจ core landmarks 5 จุด: nose, shoulders ทั้งสอง และ hips ทั้งสอง
4. สร้าง anonymous geometry-based tracker โดยใช้ center/bbox และระยะ normalized
5. ส่ง World Landmarks ของแต่ละคนเข้า frozen V4 เดิมโดยไม่ retrain และไม่เปลี่ยน feature semantics
6. สร้าง `TemporalRiskEngine` แยกหนึ่งตัวต่อ `PERSON_XX`
7. เก็บ neutral baseline แยกต่อคน หรือระบุชัดว่าใช้ absolute geometry ก่อน calibration

## ข้อจำกัดและสถานะวิทยาศาสตร์

- การรองรับ `num_poses` ใน API ถูกยืนยันจาก installed signature แล้ว
- การตรวจสอบด้วยภาพกล้องจริงหลายคนยังไม่ได้ทำ
- ยังไม่มี claim ว่า realtime บน Acer
- `num_poses=2/3/4` เป็นค่าที่ configurable สำหรับ prototype ไม่ใช่ผล benchmark
- Geometry indicators ยังคงเป็น indicators ไม่ใช่ AI classes และไม่ใช่ medical diagnosis
- V4 ยังคงเป็น pilot 2 classes: `TUP` และ `TLB`

สถานะที่เหมาะสมหลังสร้างซอฟต์แวร์คือ `MULTI_PERSON_SOFTWARE_PROTOTYPE_READY` และ `MULTI_PERSON_HARDWARE_NOT_VALIDATED` จนกว่าจะมีการทดสอบกล้องจริง
