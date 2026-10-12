# แผนผสานข้อมูลภายนอกอย่างปลอดภัย

สถานะ ณ 10 กันยายน 2026: **ไม่มี dataset ภายนอกใหม่ได้รับอนุญาตให้เข้า training set**

## หลักการตัดสินใจ

```text
source record / archive
        ↓
license + provenance check
        ↓
file inventory and raw checksum
        ↓
label ↔ media manifest join
        ↓
participant / session / camera grouping check
        ↓
MediaPipe extraction audit
        ↓
feature-schema and domain-shift audit
        ↓
frozen split design
        ↓
only then: controlled training experiment
```

ถ้าด่านใดไม่ผ่าน ให้ dataset อยู่ใน `AUDIT_ONLY`, `SMOKE_ONLY`, หรือ `BLOCKED` ไม่เข้าสู่ training table

## Integration tracks ที่ออกแบบไว้

### Track 1 — MCD-MSD (ต้องมีสิทธิ์และ audit ก่อน)

เป้าหมาย: ตรวจว่า media จริง join กับ protocol `SK`, `EK`, `EKC` ได้หรือไม่

1. ยืนยัน research-use permission และเก็บ URL/เงื่อนไขใน registry
2. ดาวน์โหลด metadata/manifest ก่อนวิดีโอ
3. ตรวจ file name → participant → session → protocol label
4. ยืนยันว่า `EKC` หมายถึง instructed-correct ตาม source
5. ห้ามรวม `SK` หรือ `EK` เป็น “incorrect” โดยอัตโนมัติ
6. สุ่มวิดีโอ 1–2 คลิปต่อ protocol เพื่อวัด MediaPipe detection และ core landmark quality
7. ถ้า label/media/group ผ่าน จึงออกแบบ subject-grouped split ที่แยกทั้งคนและ session

ผลที่เป็นไปได้:

- ผ่าน: ทดลอง **instructed-correct vs usual posture** เป็น task ใหม่ที่เรียกชื่อตรง source
- ไม่ผ่าน: เก็บเพื่อ qualitative RGB/pose audit เท่านั้น

### Track 2 — MMD-MSD (metadata-first และ full release by request)

1. อย่าดาวน์โหลด full corpus 100 คนจนได้รับเงื่อนไขการใช้และ inventory
2. ตรวจ subset 5 คนเฉพาะเมื่อ terms อนุญาต
3. ตรวจ whether video/camera file มี timestamp/participant/session และ label schema
4. ห้ามย้าย `improper neck posture` จาก use case ใน paper ไปเป็น label ทุกเฟรม
5. เลือกเทียบ distribution และ camera viewpoint ก่อนเลือกการใช้งาน

### Track 3 — MOPED25 / AnDy (biomechanics reference)

ทั้งสองแหล่งมีประโยชน์สูงสำหรับนิยาม anatomical/ergonomic geometry แต่ไม่ควรถูกนำไปสร้าง label ของ V4/V5 ทันที

- MOPED25: หากเข้าถึง full video ได้โดยสิทธิ์ชัด ให้ใช้ตรวจ pose recovery เทียบ marker kinematics; task label ไม่ใช่ TUP/TLF/TLB/TLL/TLR
- AnDy: EAWS taxonomy ต้องเก็บเป็น label แบบ EAWS; avatar replay ไม่ใช่ raw-RGB domain ของกล้อง Hikvision

### Track 4 — action / sit-stand references

MSR DailyActivity3D และ UTD-MHAD ใช้ได้เฉพาะเมื่อ file-level media, action label และ person ID ถูกตรวจจริง. เป้าหมายที่ถูกต้องคือ action/transition research ไม่ใช่ ergonomic-quality classifier.

## รูปแบบ manifest ที่ต้องมี ก่อน data เข้า experiment

```json
{
  "dataset_id": "source_name",
  "source_file": "immutable/original/path",
  "sha256": "...",
  "sample_id": "source-native-id",
  "participant_id": "anonymous-source-id",
  "session_id": "source-session-id",
  "camera_id": "source-camera-id",
  "frame_or_time_range": "...",
  "original_label": "never rewritten",
  "canonical_label": "null until mapping has written evidence",
  "label_evidence": "manifest / annotation file and row reference",
  "coordinate_representation": "unknown until measured"
}
```

## rules สำหรับ MediaPipe และ feature adapter

1. re-extract MediaPipe จาก raw RGB เท่านั้นเมื่อ RGB ได้รับสิทธิ์และมี manifest
2. บันทึก MediaPipe version/model, `image_landmarks` หรือ `world_landmarks`, visibility, timestamp, frame size และ mirror state
3. ห้ามส่ง CSV/marker/Kinect skeleton ผ่าน V4 feature adapter เสมือนว่าเป็น MediaPipe
4. ตรวจ feature order กับ V4 schema แบบ exact; mismatch ต้อง fail loudly
5. วัด OOD กับ training range แยกเป็น diagnostic; ห้าม clip ค่าทิ้งเพื่อให้ model รับ
6. external labels ใช้หลัง adapter design freeze แล้วเท่านั้น

## rules สำหรับ split และ leakage

- primary split: participant-independent
- หาก participant ไม่มี: session/video-independent
- หากไม่มีทั้งคู่: data ไม่ใช้สร้าง primary metric
- frame จากคลิปเดียวกันต้องอยู่ split เดียวกัน
- random-row split ใช้ได้เพียง diagnostic และห้ามรายงานเป็น real-world generalization

## สถานะงานที่ไม่ได้ทำตามเจตนา

- ไม่ได้ train V5
- ไม่ได้ retrain V4 หรือ MultiPosture
- ไม่ได้ train ANYA
- ไม่ได้สร้าง YOLO/PyTorch pipeline
- ไม่ได้เชื่อม camera หรือแก้ seat-aware/RTSP pipeline
