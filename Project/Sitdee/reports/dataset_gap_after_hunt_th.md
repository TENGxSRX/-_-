# ช่องว่างของข้อมูลหลังการค้นหารอบ 2

## คำตอบตรง ๆ

**ยังไม่ควรเทรน V5 จาก external dataset ใดในรอบนี้**

ไม่ใช่เพราะหา dataset ไม่ได้เลย แต่เพราะแหล่งที่พบยังขาดอย่างน้อยหนึ่งข้อจาก: label ที่ join กับ media ได้, participant/session grouping, license/permission, RGB domain ที่ใกล้กล้องจริง, หรือ semantic mapping ที่ตรงกับคลาสโครงการ

## ช่องว่างที่ยังเหลือ

| ความสามารถที่ต้องการ | หลักฐานที่มี | ช่องว่าง |
|---|---|---|
| `TUP/TLF/TLB/TLL/TLR` จาก RGB จริง | MultiPosture มี label ดีแต่ไม่ให้ raw RGB; DriveNet มี TUP/TLB จำกัดและ domain คนขับรถ | ต้องมี RGB office/room พร้อม label direction และ participant groups |
| acceptable / ergonomic-risk | MCD-MSD protocol น่าสนใจ; ANYA เป็น pseudo-label only | ต้องมี file-level verified ergonomic annotation หรือ protocol ที่นิยาม/permission ชัด |
| sit / stand | MSR/UTD มี activity/protocol ที่เกี่ยวข้อง | ต้องมี per-video label ที่ join ได้และ person grouping จาก raw RGB |
| หลายคน/seat-aware ในห้องจริง | ระบบ software พร้อม แต่ dataset external ไม่ครอบคลุม camera/desk/occlusion ของห้อง | ต้องเก็บ multi-person CCTV/RTSP-like ground truth อย่างควบคุม |
| real RGB MediaPipe adapter | MediaPipe ตรวจ pose ได้ดี แต่ MultiPosture coordinate mismatch เคยพบ | ต้อง validate บน video ที่ถ่ายแบบเดียวกับ deployment |

## สิ่งที่ “มี” กับสิ่งที่ “ยังไม่ควรอ้าง”

### มีแล้ว

- V4 frozen 2-class pilot (`TUP`, `TLB`)
- MultiPosture subject-grouped direction model จาก skeleton data
- MediaPipe / core landmark / temporal / multi-person / seat-aware software architecture
- code สำหรับเก็บ behavior และ telemetry แบบไม่เอา raw video เป็นค่าเริ่มต้น
- policy ที่กัน pseudo-label, frame leakage และ feature-schema mismatch

### ยังไม่ควรอ้าง

- real-room accuracy
- ergonomic/medical diagnosis
- 5-class RGB generalization
- correct-vs-incorrect RGB classifier
- sit/stand production classifier
- multi-person camera performance ในห้องจริง

## ทางเลือกที่แนะนำ

### 1. เก็บข้อมูลเองแบบเล็กแต่ถูกต้อง — แนะนำที่สุด

เริ่มกับ participant 8–15 คน, อย่างน้อย 2 sessions/คน, มุมกล้องและระยะใกล้ Hikvision/ห้องจริง, เก็บ label ตาม protocol ที่เขียนก่อนถ่าย. ไม่ต้องเก็บภาพบุคคลเป็นค่าเริ่มต้น: สร้าง manifest, pose telemetry และเก็บ raw video เฉพาะเมื่อผู้เข้าร่วมยินยอมและ policy อนุญาต.

minimum label set ที่ควรเริ่ม:

```text
activity: SITTING / STANDING / TRANSITION / OUT_OF_SEAT
trunk_direction: TUP / TLF / TLB / TLL / TLR / UNCERTAIN
quality: VALID / OCCLUDED / OUT_OF_VIEW
ergonomic_protocol: neutral_reference / instructed_forward / instructed_backward
```

คำว่า `instructed_*` ไม่ใช่ medical diagnosis และปลอดภัยกว่าเรียก good/bad ถ้ายังไม่มี ergonomic expert rubric

### 2. ขอสิทธิ์/manifest ก่อนจาก MCD-MSD หรือ MMD-MSD

เป้าหมายคือ metadata/label file ก่อน video ไม่ใช่ขอไฟล์ 100 GB ทันที. หากเจ้าของยืนยัน license และ file map ได้ จึงทำ small media preflight ต่อ

### 3. ใช้ external data เป็น evaluation/smoke track

MOPED25, MSR, UTD-MHAD, SitStraightNet และ DriveNet มีประโยชน์ตรวจ input quality, landmark visibility, transition และ domain shift แต่ไม่ควรเปลี่ยนเป็น training labels เพียงเพื่อเพิ่ม sample count

## เกณฑ์เปิด V5 ในอนาคต

V5 จะเริ่มได้เมื่อชุดข้อมูลหนึ่งชุดหรือ controlled local collection ผ่านทุกข้อ:

1. มี source-native label ต่อ clip/frame และ manifest join ได้
2. มี `participant_id` หรืออย่างน้อย `session_id`/`video_id`
3. เลือก split ก่อน train และ train/test groups ไม่ทับกัน
4. label definition ไม่ถูกแปลเกินหลักฐาน
5. raw RGB ผ่าน MediaPipe preflight และ feature pipeline ที่ versioned
6. license/consent/retention ชัด
7. V4 และ raw data เดิมตรวจ hash แล้วไม่เปลี่ยน

## คำตัดสินสุดท้ายรอบนี้

`EXTERNAL_DATASET_HUNT_COMPLETE — NO_NEW_TRAINING_ELIGIBLE_RGB_CORPUS`

ผลลัพธ์นี้เป็นสิ่งที่ควรเกิดเมื่อใช้ scientific gate อย่างเคร่งครัด: ป้องกันการสร้าง V5 ที่ตัวเลขสวยจาก label ไม่ตรงหรือ frame leakage แล้วใช้ไม่ได้จริงในห้อง
