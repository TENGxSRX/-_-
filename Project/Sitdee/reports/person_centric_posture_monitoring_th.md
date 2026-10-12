# Person-centric posture monitoring

อัปเดต 10 กันยายน 2026 — โหมดหลักของโครงการเปลี่ยนจากการผูกกับที่นั่งเป็นการติดตามคนแบบนิรนาม

## เส้นทางระบบหลัก

```text
RTSP / USB / VIDEO
        ↓
VideoSource
        ↓
MediaPipe Multi-Pose
        ↓
Anonymous PERSON_XX tracker
        ↓
SITTING / STANDING / UNKNOWN
        ├─ STANDING → POSTURE MONITOR PAUSED
        ├─ UNKNOWN  → ACTIVITY UNCERTAIN / PAUSE
        └─ SITTING  → frozen V4 + Geometry
                         ↓
                    per-person Temporal Risk
```

## แยกหน้าที่ให้ชัด

| ส่วน | สถานะ | ทำหน้าที่ |
|---|---|---|
| PERSON tracking | SOFTWARE READY | ตามตำแหน่ง/การเคลื่อนที่ ไม่รู้ชื่อและไม่ใช้ใบหน้า |
| SITTING/STANDING/UNKNOWN | `SIT_STAND_GEOMETRY_GATE_EXPERIMENTAL` | ใช้สะโพก-เข่า-ข้อเท้าเมื่อ visibility พอ; ไม่ใช่ AI ที่ train แล้ว |
| V4 AI | `V4_TUP_TLB_FROZEN` | ตรวจเฉพาะ TUP/TLB และทำงานเมื่อ activity เป็น SITTING |
| Geometry | EXPERIMENTAL AUXILIARY | Forward/Left/Right เป็น indicator แยกจาก AI |
| Temporal | READY PER PERSON | สะสม poor-posture duration แยกคน และไม่สะสมตอนยืน/UNKNOWN |
| Seat-aware | OPTIONAL | ยังมีไว้สำหรับห้องที่ต้องการผูก zone/seat แต่ไม่ใช่ default |

## Activity gate

ค่า threshold อยู่ใน `configs/posture_engine.yaml` และติดป้าย experimental. ต้องเห็น landmark lower-body ทั้งสองข้างและมี knee geometry สอดคล้องกันจึงจะยอมรับ SITTING/STANDING. ถ้าขาดเพราะโต๊ะบัง, occlusion, pose lost หรือ geometry ก้ำกึ่ง จะเป็น `UNKNOWN`.

การเปลี่ยน SITTING ↔ STANDING ใช้ hysteresis จำนวนเฟรมที่กำหนดได้. ระหว่างรอยืนยัน transition ระบบจะ pause posture monitoring เพื่อไม่สะสม warning จากสถานะที่ยังไม่แน่ใจ.

## Tracking lifecycle

มี event แยก ได้แก่ `PERSON_APPEARED`, `PERSON_ACTIVE`, `PERSON_TEMPORARILY_LOST`, `PERSON_REACQUIRED`, `PERSON_EXITED` และ activity events `SIT_DETECTED`, `STAND_DETECTED`, `ACTIVITY_UNCERTAIN`.

ID เป็น anonymous runtime ID เท่านั้น. Generic mode ใช้ ID slot ที่จำกัดตาม `max_people`; เมื่อ track หมดอายุสามารถ reuse slot ได้ จึงไม่อ้างว่าเป็นคนเดิมในโลกจริง. Seat-aware mode ยังคงใช้ lifecycle แบบเดิมเพื่อรักษา seat-session semantics.

## ข้อมูลสำหรับ Sit/Stand AI ในอนาคต

ใช้ `scripts/collect_controlled_activity.py` แบบ attended collection:

- ผู้ควบคุมเลือก `SITTING` หรือ `STANDING` เอง
- `P001`, `P002`, ... เป็น anonymous participant IDs
- เก็บ session/track/visibility/world landmarks/canonical features/camera metadata
- gate และ V4 เป็น diagnostic เท่านั้น
- ไม่บันทึก raw video โดยค่าเริ่มต้น
- ยังไม่ train classifier จนกว่าจะมีหลาย participants และ grouped holdout

สถานะปัจจุบันยังเป็น `SIT_STAND_AI_NOT_TRAINED` และไม่ใช่การตรวจโรคหรือ Office Syndrome.
