# CCTV Multi-Person Architecture Audit

วันที่ตรวจสอบ: 9 กันยายน 2026  
เครื่อง: `DESKTOP-ALH6CUB`

## สรุปสถานะ

สถานะซอฟต์แวร์หลัก: `PERSON_CENTRIC_PIPELINE_READY`  
สถานะข้อมูล/โมเดล CCTV ใหม่: `PENDING_TRUSTED_MULTIPERSON_LABELS`  
สถานะกล้องจริง: short RTSP source run ล่าสุดผ่านการเชื่อมต่อ แต่ยังไม่ใช่การยืนยัน accuracy หรือห้องจริง

## เส้นทางหลักปัจจุบัน

```text
USB / VIDEO / RTSP
        ↓
VideoSource
        ↓
LatestFrameInput (RTSP one-slot; file/USB synchronous)
        ↓
MultiPersonMediaPipeBackend (num_poses=max_people)
        ↓
PoseDetection[]
        ↓
AnonymousPoseTracker → PERSON_01..PERSON_XX
        ↓
Per-person core quality gate
        ↓
Frozen RGB-native V4 (TUP/TLB) + Geometry indicators
        ↓
TemporalRiskEngine แยกต่อคน
        ↓
MultiPersonFrameResult / MultiPersonTelemetry
        ↓
Overlay / JSONL dashboard input
```

`SeatAwareRuntime` เป็นชั้นเสริมที่ใช้ `SEAT_XX` และต้องมี seat-layout จริงที่ผู้ใช้ระบุเอง ไม่ใช่ default ของ generic multi-person pipeline. Generic PERSON tracking เป็นโหมดแนะนำใน phase นี้.

## ผล audit รายส่วน

| ส่วน | ไฟล์หลัก | สถานะ | ข้อสรุป |
|---|---|---|---|
| Source | `src/posture_ai/inference/video_source.py` | KEEP | USB/file/RTSP, reconnect, stale, credential redaction |
| RTSP latest frame | `src/posture_ai/inference/latest_frame_runtime.py` | KEEP | one-slot latest-frame สำหรับ RTSP; ไม่สะสม backlog |
| RTSP profiles | `configs/rtsp_profiles.yaml` | KEEP | MAIN 101, SUB 102, ค่า secret มาจาก `.env` |
| Multi-pose MediaPipe | `src/posture_ai/multiperson/mediapipe_tasks.py` | KEEP | Tasks API ใช้ `num_poses=1..4` |
| Anonymous tracking | `multiperson/tracking.py` | IMPROVE | ใช้ center/bbox continuity; ยังไม่มี ground-truth ID metrics |
| Per-person runtime | `multiperson/runtime.py` | KEEP | V4/geometry/temporal แยก state ต่อ PERSON |
| Activity gate | `ergonomics/activity_gate.py` | EXPERIMENTAL | SITTING/STANDING/UNKNOWN จาก lower-body geometry; ไม่ใช่ trained AI |
| Overlay | `multiperson/overlay.py` | KEEP / PERSON-CENTRIC | card แยกต่อคน; standing/unknown แสดง monitor paused |
| Telemetry | `multiperson/telemetry.py` | KEEP | backward-compatible และแยก source health |
| Seat-aware | `multiperson/seating.py` | EXPERIMENTAL | ใช้เมื่อมี ROI/seat layout จริงเท่านั้น |
| Single-person AI Lab | `scripts/run_rtsp_ai_lab.py` | KEEP / DIAGNOSTIC | ใช้ตรวจ V4 และ latency ไม่ใช่ architecture หลัก |
| Single-person webcam path | `scripts/run_webcam_demo.py` | KEEP / DIAGNOSTIC | rollback และ hardware calibration เดิม |
| New CCTV candidate training | `outputs/models/cctv_v5/` | BLOCKED | ยังไม่มี trusted multi-person CCTV labels |
| ANYA pseudo-labels | `datasets/raw/anya/` | BLOCKED | ห้ามใช้เป็น supervised target |
| YOLO | ไม่มีการติดตั้ง | PENDING | ยังไม่มีหลักฐานว่า MediaPipe เป็น bottleneck ถาวร |

## การเปลี่ยนแปลงในรอบนี้

- launcher multi-person เปลี่ยน default เป็น `max_people=3` เพื่อให้สอดคล้องกับงาน CCTV แต่ยังรับ `1..4`
- RTSP multi-person ใช้ `LatestFrameInput` เดิม ไม่สร้าง capture path ที่สอง
- เพิ่ม preset `RTSP_CCTV`: process ทุกเฟรมที่อ่านได้, inference copy 960×540, latest-frame semantics
- default display copy เป็น 1280×720 เพื่อไม่วาดภาพ 2560×1440 โดยไม่จำเป็น
- เพิ่ม metrics แยก detection, core quality, rejected poses, frame age, inference latency และ drops
- เพิ่ม contract สำหรับ dataset registry และ canonical adapter ที่ reject label/representation ที่ยังพิสูจน์ไม่ได้
- เพิ่ม `scripts/collect_controlled_cctv.py` สำหรับเก็บข้อมูลแบบมีผู้ควบคุม โดยเลือก `PERSON_XX` และติดป้ายเอง 5 คลาส; ไม่ใช้ V4 prediction เป็น ground truth และไม่บันทึก raw video โดยค่าเริ่มต้น
- เพิ่ม activity gate แบบ `GEOMETRY_EXPERIMENTAL`, per-person pause semantics และ lifecycle/activity events
- เพิ่ม `scripts/collect_controlled_activity.py` สำหรับ future human-labelled SITTING/STANDING โดยไม่บันทึก raw video

## สิ่งที่ยังไม่ควรสรุป

การตั้ง `max_people=4` หมายถึงเพดานที่ส่งให้ MediaPipe ไม่ได้แปลว่ากล้องตรวจพบ 4 คนจริง รอบล่าสุด `max_people=3` นาน 5 วินาที ตรวจพบ 1 pose และ core-valid 1 pose, source 31.45 FPS, AI 6.37 FPS, median inference 113.3 ms, frame-age median 16.0 ms, stale 1 ครั้ง, decode/reconnect 0. ยังไม่มี manual expected-people ground truth จึงยังไม่มี person-count accuracy, missed-person rate, IDF1 หรือ cross-view validation

Activity gate ตอนนี้เป็น heuristic เพื่อหยุดการวิเคราะห์ตอนยืน/ข้อมูลไม่พอเท่านั้น ไม่ควรใช้เป็นรายงานความแม่นยำของ sit/stand จนกว่าจะมี human-labelled evaluation.

## คำสั่งหลัก

```powershell
& '.\.venv\Scripts\python.exe' scripts\run_multiperson_demo.py --source rtsp --profile HIKVISION_MAIN --max-people 3 --preset RTSP_CCTV
```

ระบบไม่ใช้ใบหน้า ไม่ใช้ชื่อครู และไม่ใช้ biometric identity
