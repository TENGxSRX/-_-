# สถานะการ integrate Acer → Desktop

ตรวจบน `DESKTOP-ALH6CUB` วันที่ 8 กันยายน 2026

## ผลการรับไฟล์

- Taildrop ZIP อยู่ที่ `C:\Users\ADMIN\Downloads\acer_seat_aware_v2_transfer.zip`
- manifest อยู่ที่ `C:\Users\ADMIN\Downloads\acer_seat_aware_v2_transfer_manifest.json`
- ZIP SHA-256 ตรงกับ expected:
  `35980111ba65383bcb4cf405f45fcdc620d19e9be2fb0d30d960033cce12a76a`
- คัดลอกแบบไม่ย้ายไป `transfers/` และ extract แยกไว้ใน
  `outputs/integration_staging/acer_seat_aware_v2/`

## สิ่งที่ integrate แล้ว

```text
USB / VIDEO / RTSP
        ↓
    Desktop VideoSource
        ↓
 MediaPipe Multi-Pose (max_people 1–4)
        ↓
 anonymous PERSON_XX tracking
     ┌──┴─────────────┐
 generic mode       seat mode
     └──┬─────────────┘
 per-person / per-seat V4 + Geometry + Temporal
        ↓
 source-safe telemetry + dashboard
```

- `run_multiperson_demo.py` ใช้ `VideoSource` เดียวกับ RTSP runtime ไม่สร้าง
  `cv2.VideoCapture` ซ้ำใน AI layer
- generic รองรับ `--max-people 1` ถึง `4`
- seat mode ใช้ Hungarian assignment, dwell/hysteresis, occlusion และ
  passerby states จาก Acer
- seat mode ต้องระบุ `--seat-layout` เอง และไม่ใช้พิกัดห้องปลอม
- telemetry เก็บ source health แบบ allow-list และ redact URL credentials
- ใช้ Desktop V4 artifact เดิมเท่านั้น; ไม่ import V4 จาก Acer

## สิ่งที่ยังไม่ทำ

- ไม่เปิด webcam
- ไม่เชื่อม Hikvision RTSP
- ไม่ train/retrain V4, V5 หรือ ANYA
- ไม่ติดตั้ง YOLO
- ไม่สร้าง real seat coordinates
- ไม่ทำ face recognition หรือ biometric identity

## หลักฐาน Acer

`evidence/multiperson_test.jsonl` ถูก replay แบบ read-only ได้ 300 frames,
100 inference frames, 200 skipped frames, 21 anonymous IDs และสูงสุด 3 คนพร้อมกัน
หลักฐานนี้ไม่ถูกใช้เป็น training, tuning หรือ model evaluation data

## ผลทดสอบและ integrity

- Full test suite: **133 passed**
- ANYA raw, MultiPosture raw และ V1/V2/V4 model hashes ยังตรงกับ manifest เดิม
- V4 ยังคง `TUP` / `TLB`; Forward/Left/Right ยังคงเป็น Geometry Indicators
- `configs/seat_layout_hikvision.example.yaml` เป็น template ว่าง สถานะ
  `WAITING_FOR_REAL_ROOM_LAYOUT`
- code-only bundle v5: `outputs/laptop_dev_bundle_v5.zip`, ขนาด 27.61 MiB,
  SHA-256 `04d740daef4c8fb879f7ff875e5dd902caca659e0285e62b2e9cd0b9bf221a78`

สถานะรวม: **INTEGRATED_PIPELINE_READY_FOR_HARDWARE_TEST**

ข้อจำกัด: **HIKVISION_REAL_STREAM_NOT_YET_VALIDATED** และ
**REAL_ROOM_NOT_YET_VALIDATED**

## สถานะตามส่วน

| ส่วน | สถานะ |
|---|---|
| RTSP | `RTSP_SOFTWARE_READY` |
| Multi-person | `MULTI_PERSON_SOFTWARE_READY` |
| Seat-aware | `SEAT_AWARE_SOFTWARE_READY` |
| Integrated pipeline | `INTEGRATED_PIPELINE_READY_FOR_HARDWARE_TEST` |
| Hikvision stream | `HIKVISION_REAL_STREAM_NOT_YET_VALIDATED` |
| Room layout | `REAL_ROOM_NOT_YET_VALIDATED` |
