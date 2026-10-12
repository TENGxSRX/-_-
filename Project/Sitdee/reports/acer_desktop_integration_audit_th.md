# Acer → Desktop integration audit (pre-merge)

ตรวจบน `DESKTOP-ALH6CUB` ก่อนแก้ production files

## Transfer verification

- พบไฟล์ต้นฉบับที่ `C:\Users\ADMIN\Downloads\`
- ZIP SHA-256 ตรงกับค่าที่ผู้ใช้กำหนด:
  `35980111ba65383bcb4cf405f45fcdc620d19e9be2fb0d30d960033cce12a76a`
- คัดลอกแบบไม่ย้ายไป `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\transfers\`
- manifest ระบุ 28 files และ ZIP มี 28 entries ตรงกันทั้งหมด
- extract อยู่เฉพาะใน `outputs/integration_staging/acer_seat_aware_v2/`
- ไม่พบ absolute path, `..` path traversal, dataset, model binary, `.env`, credential หรือวิดีโอใน archive

## Classification

| กลุ่ม | ไฟล์/ขอบเขต | การตัดสินใจ |
|---|---|---|
| `ALREADY_PRESENT` | `requirements-laptop.txt`, `configs/posture_engine.yaml` | hash ตรงกัน ไม่ต้องคัดลอกซ้ำ |
| `SAFE_ADD` | `multiperson/demo.py`, `mediapipe_tasks.py`, `overlay.py`, `runtime.py`, `seating.py`, `telemetry.py`, `tracking.py`, scripts/tests/fixtures/reports/evidence ใน archive | เพิ่มได้หลังแก้ source I/O ให้ใช้ Desktop `VideoSource` |
| `SAFE_MERGE` | `src/posture_ai/multiperson/schema.py` | รักษา contract เดิม แล้วเพิ่ม tracking/bbox/world-landmark fields ที่ validate แล้ว |
| `SAFE_MERGE` | `src/posture_ai/multiperson/__init__.py` | เพิ่ม exports ของ modules ใหม่ โดยไม่ลบ exports เดิม |
| `SAFE_MERGE` | `README.md` | เพิ่มคำอธิบาย ไม่แทนที่ RTSP/วิทยาศาสตร์เดิม |
| `SAFE_ADD_TEST_ONLY` | `configs/seat_layout_example.yaml`, `fixtures/seat_layout_example.png` | ใช้ synthetic/test fixture เท่านั้น ไม่ใช่พิกัดห้องจริง |
| `DO_NOT_IMPORT_AS_MODEL_DATA` | `evidence/multiperson_test.jsonl` | เก็บเป็นหลักฐาน/fixture ได้ แต่ห้ามใช้ train, tune หรือประเมินโมเดล |
| `CONFLICT / REWORK_REQUIRED` | `demo.py` และ CLI source handling | Acer เดิมเรียก `cv2.VideoCapture` ตรง จึงต้องเปลี่ยนให้ผ่าน Desktop `VideoSource` ก่อน |
| `DO_NOT_IMPORT` | ไม่มี model artifact, MediaPipe task asset, raw dataset หรือ credentials ใน archive | ไม่สร้างไฟล์ทดแทนจาก Acer |

## จุดเสี่ยงที่ต้องรักษา

1. Desktop V4 จาก `outputs/models/rgb_native_v4/` เป็น source of truth ห้าม copy/overwrite จาก Acer
2. V4 มีเฉพาะ `TUP` และ `TLB`; Forward/Left/Right เป็น geometry indicators
3. RTSP reconnect, stale state, URL redaction และ source metadata ต้องอยู่ใน `VideoSource` ไม่ซ้ำใน AI layer
4. generic mode และ seat mode ต้องเป็น opt-in; seat mode ต้องรับ layout จริงที่ผู้ใช้กำหนด
5. `max_people` อนุญาตเฉพาะ 1–4 และต้องไม่ทำลาย single-person workflow
6. telemetry ใหม่ต้องมี camera/source metadata แต่ห้ามมี password หรือ credential-bearing URL
7. current Desktop มี `schema.py` อยู่แล้ว จึงต้อง merge แบบ additive ไม่ใช่ copy ทับ

## ผล audit ก่อน merge

**SAFE TO PROCEED WITH CONTROLLED MERGE**

การ merge ที่อนุญาตคือเพิ่ม multi-person/seat-aware code และ adapters ที่ผ่าน tests
เท่านั้น ไม่รวมการเปิดกล้อง, RTSP จริง, retraining, V5, ANYA หรือ YOLO
