# Seat-aware Multi-person Posture Monitoring Prototype V2

วันที่: 8 กันยายน 2026  
สถานะซอฟต์แวร์: `SEAT_AWARE_SOFTWARE_READY`  
สถานะฮาร์ดแวร์: `HIKVISION_HARDWARE_PENDING`  
สถานะห้องจริง: `REAL_ROOM_VALIDATION_PENDING`

เอกสารนี้เป็น prototype สำหรับงานวิจัย/สาธิตเชิงวิศวกรรม ไม่ใช่ production readiness และไม่ใช่การวินิจฉัยทางการแพทย์

## 1. ทำไม PERSON IDs จึง fragment

จาก real log `outputs/multiperson_test.jsonl` จำนวน 300 telemetry frames มี:

- unique IDs ตั้งแต่ `PERSON_01` ถึง `PERSON_21` รวม 21 IDs
- maximum simultaneous detections = 3
- 21 first-appearance/new-ID events
- 7 IDs ที่มีสถานะ `TEMPORARILY_LOST`
- 200 frames เป็น telemetry ของผลเดิมที่ถูกทำเครื่องหมาย `inference_skipped`

ค่าเหล่านี้เป็นหลักฐานของ track fragmentation ใน runtime run นี้ แต่ไม่มี ground-truth identity จึงไม่ใช่ identity accuracy หรือ IDF1

สาเหตุเชิงสถาปัตยกรรมคือ MediaPipe ให้ pose detections แต่ไม่ได้ให้ persistent biological identity ที่เชื่อถือได้ ระบบ generic V1 จับคู่ด้วยตำแหน่งกึ่งกลางและระยะ normalized เมื่อคนเข้าออก บังกัน หรือเคลื่อนที่เร็ว จึงอาจสร้าง `PERSON_XX` ใหม่ได้ นอกจากนี้ log เดิมไม่ได้เก็บ raw detector order จึงไม่ควรใช้ลำดับ `[A,B]` เป็น identity

รายละเอียดตัวเลขอยู่ใน [รายงานวิเคราะห์ real log](multiperson_real_test_analysis_th.md)

หากต้องการ replay log เดิมแบบ read-only ให้ใช้ `scripts/replay_multiperson_log.py` ระบบจะรายงาน field ที่ไม่มีอยู่จริงแทนการเติมข้อมูลที่ขาด

## 2. ทำไม fixed seats เหมาะกับห้องครู

สภาพแวดล้อมเป้าหมายเป็นโต๊ะ/เก้าอี้ประจำ ไม่ใช่ CCTV ที่ต้องติดตามคนทั่วพื้นที่ ดังนั้น persistent context ที่เหมาะกว่าคือ workstation:

```text
SEAT_01  ← current short-lived PERSON_07
SEAT_02  ← current short-lived PERSON_11
SEAT_03  ← empty / candidate / occupied
```

`SEAT_XX` ไม่ใช่ชื่อบุคคลและไม่ใช่การจำแนกตัวตน หาก PERSON ID แตกแต่ตำแหน่งใน seat และหลักฐานความต่อเนื่องยังสอดคล้อง ระบบสามารถคง session ของ seat ได้โดยไม่อ้างว่าเป็นคนเดิมทางชีวภาพ

## 3. SeatZone และพิกัด normalized

แต่ละ `SeatZone` มี:

- `seat_id`
- `roi` แบบ rectangle หรือ `polygon`
- `anchor_x`, `anchor_y`
- `expected_facing`
- `enabled`
- runtime fields: `assignment_state`, `current_track_id`, `last_valid_subject`, `occupied_duration`

พิกัดเป็น normalized image coordinates ในช่วง `[0.0, 1.0]`:

- `x=0` อยู่ซ้าย, `x=1` อยู่ขวา
- `y=0` อยู่ด้านบน, `y=1` อยู่ด้านล่าง
- ใช้ได้กับ 640×480, 1280×720 และ 1920×1080
- `roi: [left, top, right, bottom]`

ตัวอย่างปัจจุบันเป็น `TEST_ROOM_ONLY` เท่านั้น อยู่ใน [seat_layout_example.yaml](../configs/seat_layout_example.yaml) และยังไม่ใช่ตำแหน่งห้องจริง

## 4. วิธีที่ SEAT_01 ทำงาน

ตำแหน่งของ detection เลือกตามลำดับที่ conservative:

1. hip midpoint เมื่อสะโพกทั้งสองจุดใช้ได้
2. shoulder midpoint เมื่อสะโพกใช้ไม่ได้
3. detector center หรือ bbox center เป็น fallback

การจับคู่ seat/detection ใช้ deterministic global assignment ของ SciPy Hungarian algorithm โดย cost พิจารณา:

- point-in-polygon ของ seat ROI
- ระยะจากตำแหน่งไปยัง seat anchor
- ความชอบ detection ที่เป็น current/last track ของ seat
- pose quality

หนึ่ง detection ถูกจับคู่ได้ไม่เกินหนึ่ง seat และ order ของ detector ไม่ควรเปลี่ยน seat assignment

## 5. เมื่อมีคนเดินผ่าน

ระบบไม่เปลี่ยน occupant ทันทีจาก detection เดียว:

- คนใหม่ใน seat ROI จะเริ่มเป็น `CANDIDATE`
- ต้องมี core landmarks และคุณภาพขั้นต่ำต่อเนื่องอย่างน้อย `1.5` วินาที
- ถ้า seat มี occupant เดิม คนที่เข้ามาใหม่จะได้ `PASSERBY_DETECTED`
- การแทนที่ occupant ต้องผ่าน replacement dwell `1.5` วินาที และสร้าง occupant-session ใหม่
- `MULTIPLE_CANDIDATES` ถูกแสดงเมื่อหลาย detection อยู่ใน seat เดียวกัน

กฎนี้เป็น heuristic ของ prototype ไม่ใช่การรู้ว่าใครเป็นใคร

## 6. Seat occupancy states

| State | ความหมาย |
|---|---|
| `EMPTY` | ยังไม่มี occupant ที่ผ่านเงื่อนไข |
| `CANDIDATE` | detection อยู่ใน seat แต่ยังไม่ครบ dwell |
| `OCCUPIED` | ผ่าน dwell และมี seat session |
| `OCCLUDED` | occupant หาย/ถูกบังชั่วคราวภายใน grace period |
| `LOST` | ไม่เห็น occupant เกิน grace แต่ยังไม่ reset seat ทันที |

ค่าเริ่มต้นของ prototype คือ occlusion grace `1.0` วินาที และ empty timeout `3.0` วินาที

เมื่อเป็น `OCCLUDED` ระบบไม่ให้ passerby สืบทอด timer ของ occupant เดิมโดยอัตโนมัติ posture engine จะได้รับ `POSE_LOST` แบบ conservative ขณะที่ seat session ยังถูกเก็บไว้ชั่วคราว หากเป็นคนใหม่จริงและอยู่ต่อจนผ่าน replacement dwell จึง reset occupant session และ per-seat posture history

## 7. Per-seat temporal engine

`TemporalRiskEngine` ถูกสร้างแยกตาม `SEAT_XX` ไม่ได้ keyed เฉพาะ `PERSON_XX` ดังนั้น timer ของ `SEAT_01` ไม่ปนกับ `SEAT_02`

เมื่อ occupant replacement ถูกยืนยัน:

- reset posture engine ของ seat
- ล้าง runtime neutral baseline ของ seat
- สร้าง session ID ใหม่ เช่น `SESSION_SEAT_02_002`
- ไม่ถ่ายโอน posture history เดิมไปยัง occupant ใหม่

เมื่อ tracker ID แตกแต่ตำแหน่งเดิมใกล้เดิมภายใน continuity window `1.0` วินาทีและระยะไม่เกิน `0.12` normalized units ระบบอาจเปลี่ยน current debug track โดยคง seat session เดิมไว้ นี่เป็น continuity heuristic ไม่ใช่ face/identity recognition

## 8. V4 เปลี่ยนหรือไม่

ไม่เปลี่ยน:

- ไม่ retrain
- ไม่แก้ `model.joblib`, feature order หรือ feature semantics
- ยังคงใช้ 15 `normalized_world` features
- AI labels ยังมี `TUP` และ `TLB` เท่านั้น
- Forward/Left/Right และ shoulder/head indicators ยังเป็น geometry indicators ไม่ใช่ AI classes

เมื่อ detection ของ seat ใช้ core landmarks ได้ ระบบส่ง world landmarks ของคนนั้นเข้า frozen V4 เดิม หาก landmark ไม่พอ ระบบไม่สร้าง prediction ใหม่และแสดงสถานะ input/pose uncertainty แยกต่างหาก

## 9. ใช้ใบหน้าหรือไม่ และ 3 คน validate แล้วหรือยัง

ไม่ใช้:

- face recognition
- facial embeddings
- teacher names
- biological identity matching

ระบบใช้เพียง `CAM01`, `SEAT_XX`, `PERSON_XX` ชั่วคราว และ runtime session IDs

โครงสร้างรองรับ 3 คนและผ่าน simulated 3-seat scenarios แล้ว แต่ยังไม่มีการ validate กล้องจริง 3 คนบน Acer จึงยังห้ามอ้างว่า realtime หรือ production-ready

## 10. สิ่งที่ต้องทำเมื่อ Hikvision มาถึง

1. จับหรือโหลด reference frame หนึ่งภาพจากมุมกล้องจริง
2. รัน `scripts/configure_seat_zones.py` กับไฟล์ภาพนั้น
3. วาด/คลิก ROI ของ `SEAT_01`, `SEAT_02`, ... และตรวจทาน normalized coordinates
4. บันทึก YAML ใหม่ที่ระบุ `layout_status` ว่าอยู่ระหว่างตรวจห้องจริง
5. ใช้ `scripts/visualize_seat_layout.py` ตรวจ ROI บน reference frame
6. ทดสอบ bounded run แบบ manual ด้วย `--tracking-mode seat --max-people 3`
7. บันทึกผลเรื่อง seat continuity, passerby, occlusion, FPS และ left/right ตามมุมกล้องจริง

ตัวอย่างสร้าง ROI จาก pixel rectangle โดยไม่เปิดกล้อง:

```powershell
python scripts\configure_seat_zones.py --image room_reference.jpg --seat SEAT_01:40,80,250,430 --seat SEAT_02:280,80,490,430 --output configs\seat_layout_hikvision.yaml
```

ตัวอย่าง bounded manual validation หลังตั้งค่าแล้ว:

```powershell
& '.\.venv\Scripts\python.exe' scripts\run_multiperson_demo.py --source 0 --tracking-mode seat --seat-layout configs\seat_layout_hikvision.yaml --max-people 3 --preset LOW_SPEC --max-frames 300 --telemetry outputs\seat_aware_validation.jsonl
```

คำสั่งเหล่านี้ต้องเริ่มโดยผู้ปฏิบัติงานเอง ไม่เปิดกล้องอัตโนมัติ

## 11. Generic V1 เทียบกับ Seat-aware V2

| Metric | Generic V1: real log | Seat-aware V2: current evidence |
|---|---:|---|
| unique PERSON IDs | 21 | ใช้เป็น short-lived debug IDs; ไม่ใช่ persistent context |
| max simultaneous detections | 3 | config/test รองรับ 3+ seats |
| ID creation events | 21 | seat session เปลี่ยนเฉพาะหลัง replacement dwell |
| detector-order robustness | raw order ไม่ได้ถูกเก็บใน log | deterministic global assignment + explicit order-swap test |
| walk-by disruption | log ไม่มี seat/ROI จึงวัดไม่ได้ | one-frame passerby ไม่แทน occupant; ได้ `PASSERBY_DETECTED` |
| short occlusion | generic track มี `TEMPORARILY_LOST` 21 records | seat คง context เป็น `OCCLUDED`/`LOST` และไม่ถ่ายโอน timer แบบทันที |
| seat continuity | ไม่มี seat context | synthetic fragmentation continuity คง occupant session เมื่อ evidence ผ่าน |
| test evidence | observed runtime log | 12/12 deterministic seat-aware tests ผ่าน |

คอลัมน์ V2 เป็นผลจาก synthetic fixture/tests ไม่ใช่ผลการทดลองห้องจริง และไม่ควรนำไปแทน hardware validation

## 12. สถานะสุดท้าย

```text
SEAT_AWARE_SOFTWARE_READY
HIKVISION_HARDWARE_PENDING
REAL_ROOM_VALIDATION_PENDING
```

ระบบพร้อมสำหรับการตรวจสอบโค้ดและการเตรียม layout แต่ยังไม่พร้อมอ้าง production readiness, accuracy ทางการแพทย์ หรือ real-time performance บน Acer จนกว่าจะทดสอบด้วย reference frame/กล้อง Hikvision จริงและเก็บผลอย่างเป็นระบบ
