# สถาปัตยกรรม CCTV posture intelligence ฉบับรวม

อัปเดต 10 กันยายน 2026 บน `DESKTOP-ALH6CUB`

## เส้นทางที่เตรียมไว้

```text
USB / video file / RTSP
        ↓
VideoSource + latest-frame buffer + source health
        ↓
MediaPipe multi-pose (max_people 1..4)
        ↓
anonymous PERSON_XX tracking
        ↓
per-person core quality
        ↓
frozen V4 expert (TUP/TLB)
        + Geometry indicators
        + temporal state
        ↓
hierarchical decision contract
  activity → sitting quality → subtype
        ↓
telemetry / dashboard / session summary
```

ส่วน `activity`, `sitting quality` และ `subtype` มี decision contract แล้ว แต่ยังไม่มี trusted RGB/CCTV training set จึงคืนค่า `UNCERTAIN` เมื่อไม่มีหลักฐาน ไม่เดาผลจาก V4.

## YOLO role

เตรียม optional candidate สองแบบ:

- YOLO person box → crop → MediaPipe ต่อคน
- YOLO Pose เป็น backend เปรียบเทียบแยกต่างหาก

ไม่มีการติดตั้งหรือดาวน์โหลด weight ในรอบนี้ เพราะ venv ยังไม่มี `torch`/`ultralytics` และ C: เหลือประมาณ 10.8 GiB. MediaPipe ยังเป็น backend หลัก.

## สถานะวิทยาศาสตร์

- V4 frozen และใช้เป็น expert TUP/TLB ต่อ track เท่านั้น
- Geometry เป็น auxiliary indicator ไม่ใช่ ground truth
- canonical CCTV schema `cctv_canonical_v1` มี hip-center และ optional torso scale แต่ยังไม่พิสูจน์ viewpoint invariance และไม่ compatible กับ V4 โดยอัตโนมัติ
- ยังไม่มี activity/sitting-quality/subtype model ที่ train ได้อย่างถูกต้อง

## สิ่งที่ไม่ทำ

ไม่รวม ANYA pseudo-label, ไม่ใช้ SitPose depth features เป็น RGB target, ไม่แก้ raw data/V1/V2/V4 และไม่อ้าง real-world accuracy ก่อนมี human ground truth.
