# Acer → Desktop Transfer Merge Plan

วันที่ตรวจสอบ: 8 กันยายน 2026  
ต้นทาง: `C:\Users\sun\Documents\laptop_dev_bundle_v3`  
Git baseline: `3bc2e7f` — `Acer laptop baseline before multi-person development`

เอกสารนี้เป็นแผนก่อน merge; หลังจาก audit ตามเอกสาร
`reports/acer_desktop_integration_audit_th.md` ได้ทำ controlled merge แล้ว
โดยยังคงข้อจำกัดเรื่องกล้องจริงและโมเดลเดิม

## หลักการ

- คัดลอกเฉพาะ multi-person/seat-aware implementation, tests, fixture, config, reports และ real telemetry ขนาดเล็ก
- Desktop คง RTSP/Hikvision source abstraction, source health, CLI และ dashboard ของตนเองไว้ก่อน
- `PERSON_XX` เป็น short-lived anonymous debug track; `SEAT_XX` เป็น workstation context ไม่ใช่ identity
- V4 frozen: ไม่ส่ง patch เพื่อแก้ feature semantics หรือ retrain model

## File-level classification

| Acer path | Category | Decision | เหตุผล/จุดต้องระวัง |
|---|---|---|---|
| `src/posture_ai/multiperson/tracking.py` | MULTIPERSON_CORE | `SAFE_COPY` | tracker ใหม่แยกจาก single-person path แต่ desktop ต้อง map `PoseLandmarks` ให้ตรง |
| `src/posture_ai/multiperson/schema.py` | MULTIPERSON_CORE | `REVIEW_BEFORE_MERGE` | แก้ schema scaffold เดิมและเพิ่ม fields ที่ downstream อาจใช้ |
| `src/posture_ai/multiperson/mediapipe_tasks.py` | MULTIPERSON_CORE | `REVIEW_BEFORE_MERGE` | สร้าง MediaPipe Tasks landmarker เอง; ต้องเทียบ desktop initialization/source lifecycle |
| `src/posture_ai/multiperson/runtime.py` | MULTIPERSON_CORE | `REVIEW_BEFORE_MERGE` | เรียก frozen V4, geometry และ per-person temporal state |
| `src/posture_ai/multiperson/seating.py` | SEAT_AWARE_CORE | `SAFE_MERGE` | seat model/ROI/Hungarian assignment แยกเป็นโมดูลใหม่; ตรวจชื่อ state กับ desktop |
| `src/posture_ai/multiperson/telemetry.py` | MULTIPERSON_CORE | `LIKELY_CONFLICT` | desktop มี telemetry contract ของตนเอง; ควร merge schema โดยรักษา legacy fields |
| `src/posture_ai/multiperson/overlay.py` | MULTIPERSON_CORE | `LIKELY_CONFLICT` | desktop dashboard/overlay อาจมี renderer และ field layout ต่างกัน |
| `src/posture_ai/multiperson/demo.py` | MULTIPERSON_CORE | `LIKELY_CONFLICT` | จัดการ `VideoCapture`, timestamp, frame skipping และ `imshow`; ห้ามแทน RTSP abstraction ทันที |
| `src/posture_ai/multiperson/__init__.py` | MULTIPERSON_CORE | `REVIEW_BEFORE_MERGE` | export surface เปลี่ยนและ eager-import demo/dependencies |
| `scripts/run_multiperson_demo.py` | TOOLS | `LIKELY_CONFLICT` | argparse มี `--source`, `--preset`, `--tracking-mode`, `--seat-layout`; desktop CLI ต้องเป็นเจ้าของ entrypoint เดียว |
| `scripts/replay_multiperson_log.py` | TOOLS | `SAFE_COPY` | read-only log replay ไม่ผูกกับกล้อง |
| `scripts/configure_seat_zones.py` | TOOLS | `SAFE_COPY` | รับ reference image/พิกัด pixel และบันทึก normalized YAML; ไม่เปิดกล้อง |
| `scripts/visualize_seat_layout.py` | TOOLS | `SAFE_COPY` | วาด ROI บนภาพหรือ synthetic fixture; ไม่ผูก source backend |
| `scripts/multiperson_fixture_demo.py` | TOOLS | `SAFE_COPY` | synthetic architecture fixture เท่านั้น |
| `configs/seat_layout_example.yaml` | CONFIG | `SAFE_COPY` | ทำเครื่องหมาย `TEST_ROOM_ONLY`; ห้ามใช้แทน layout ห้องจริง |
| `tests/test_multiperson_prototype_v1.py` | TESTS | `SAFE_COPY` | ตรวจ generic tracking/telemetry/V4 isolation |
| `tests/test_seat_aware_multiperson_v2.py` | TESTS | `SAFE_COPY` | ตรวจ 12 deterministic seat scenarios; run หลัง map dependencies |
| `fixtures/multiperson_prototype_v1.json` | OPTIONAL_EVIDENCE | `SAFE_COPY` | synthetic only; ไม่มี participant/camera data |
| `fixtures/seat_layout_example.png` | OPTIONAL_EVIDENCE | `SAFE_COPY` | synthetic visualization only |
| `outputs/multiperson_test.jsonl` | OPTIONAL_EVIDENCE | `SAFE_COPY` | real telemetry 2,757,902 bytes; ไม่มี raw video/identity; ใช้ตรวจ fragmentation |
| `reports/multiperson_mediapipe_audit_th.md` | REPORTS | `SAFE_COPY` | ผลตรวจ MediaPipe 1.0.1 และข้อจำกัด API |
| `reports/multiperson_prototype_v1_th.md` | REPORTS | `SAFE_COPY` | V1 implementation/validation report |
| `reports/multiperson_real_test_analysis_th.md` | REPORTS | `SAFE_COPY` | ตัวเลขจาก real log และข้อจำกัด ground truth |
| `reports/seat_aware_multiperson_v2_th.md` | REPORTS | `SAFE_COPY` | V2 seat-aware design/limitations |
| `reports/acer_to_desktop_merge_plan_th.md` | REPORTS | `SAFE_COPY` | เอกสารนี้ |
| `README.md` | REPORTS | `REVIEW_BEFORE_MERGE` | มีทั้ง historical single-person และ Acer launcher docs; merge เฉพาะ section ที่เกี่ยวข้อง |
| `configs/posture_engine.yaml` | CONFIG | `REVIEW_BEFORE_MERGE` | runtime ต้องใช้ threshold/config นี้ แต่ desktop อาจมี policy ของตนเอง |
| `requirements-laptop.txt` | CONFIG | `SAFE_COPY` | runtime compatibility context เท่านั้น ไม่ใช่คำสั่งให้เปลี่ยน desktop environment |

## จุดเสี่ยงหลัก

### `run_multiperson_demo.py` และ source abstraction

Acer demo รับ camera index หรือ video path โดยตรงและใช้ OpenCV `VideoCapture` ส่วน desktop มี RTSP/Hikvision source abstraction แล้ว จึงไม่ควรคัดลอก launcher ไปทับ desktop ให้ map detector/runtime เข้า source interface ของ desktop แทน

### MediaPipe initialization

Acer adapter สร้าง Tasks `PoseLandmarkerOptions` ด้วย `num_poses=max_people` และอ่าน `pose_landmarks`/`pose_world_landmarks` ทุก pose Desktop ต้องตรวจ running mode, timestamp units, model asset ownership และ lifecycle ของ landmarker ก่อนใช้ร่วมกัน

### Telemetry/dashboard

Acer event มี `subjects[]` และ seat-aware event มี `seats[]` พร้อม debug `PERSON_XX` การ merge ต้องรักษา legacy single-person event และอย่าแสดง teacher names หรือเปลี่ยน dashboard field ของ desktop โดยอัตโนมัติ

### Presets/performance

Acer preset คือ `DESKTOP`, `BALANCED`, `LOW_SPEC` และใช้ frame skipping ทุก N frames โดยรักษา timestamp จริง Desktop ควรให้ source abstraction เป็นเจ้าของ capture FPS/resize แล้วส่ง scheduler settings ที่ compatible เข้า runtime

## ไฟล์ที่ตั้งใจไม่ส่ง

- `.venv`, caches, `__pycache__`, `.pyc`
- `datasets/` และ raw datasets ทุกชุด รวม ANYA, DriveNet, MultiPosture, MVSP
- raw video, recordings และ archives
- `metadata/pose_landmarker_heavy.task` (30,664,242 bytes) — desktop ควรใช้ model asset ที่ตนจัดการเอง
- `outputs/models/rgb_native_v4/` — V4 binary/config เป็น artifact ของ desktop/runtime เดิม ไม่ใช่การเปลี่ยนแปลงของ seat-aware layer
- credentials, `.env`, API keys, passwords และ camera URLs

## ขั้นตอน merge ที่แนะนำภายหลัง

1. Desktop owner ตรวจ archive/manifest และ review `REVIEW_BEFORE_MERGE`/`LIKELY_CONFLICT`
2. คัดลอก `seating.py`, tests และ config example ก่อน โดยยังไม่ผูก RTSP
3. สร้าง adapter ระหว่าง desktop source abstraction กับ `PoseDetection`
4. Merge telemetry/dashboard contract แบบ explicit และทดสอบ legacy logs
5. สร้าง real Hikvision seat layout แยกจาก `TEST_ROOM_ONLY`
6. ทำ bounded hardware validation แล้วค่อยพิจารณา integration commit

## ผลหลัง controlled merge

- `demo.py` และ `run_multiperson_demo.py` ถูกปรับให้ใช้ Desktop `VideoSource`
  แทนการเปิด `cv2.VideoCapture` ตรง
- `max_people` ถูกจำกัดไว้ที่ 1–4 และมี `generic`/`seat` tracking modes
- seat mode ไม่มี default real layout; ต้องส่ง `--seat-layout` เอง
- source health/credential redaction ของ Desktop ยังคงเป็นเจ้าของ RTSP
- Desktop V4 artifact ไม่ถูกแทนที่ และ evidence ไม่ถูกใช้เป็น training data
- full tests ผ่าน `133 passed`

สถานะปัจจุบัน: `CONTROLLED_MERGE_COMPLETE_READY_FOR_HARDWARE_TEST`
