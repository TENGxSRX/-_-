# ผลทดสอบโหมดประสิทธิภาพสำหรับเครื่องสเปกต่ำ

ตรวจเมื่อ 2026-09-07

## สิ่งที่ทดสอบจริงคืนนี้

ใช้ไฟล์เล็ก `outputs/videos/engine_blank_analyzed.mp4` ซึ่งมี 2 frames ที่ 5 FPS และไม่มีคนอยู่ในภาพ ทดสอบแบบ prerecorded เท่านั้น ไม่เปิด webcam และไม่ใช้ dataset ใหญ่

| Preset | ตั้ง process ทุก | frames ที่อ่าน | MediaPipe calls | calls ต่อ frame ที่อ่าน | เวลาโดยประมาณ |
|---|---:|---:|---:|---:|---:|
| DESKTOP | ทุก 1 frame | 2 | 2 | 1.00 | 0.49 s |
| BALANCED | ทุก 2 frames | 2 | 1 | 0.50 | 0.32 s |
| LOW_SPEC | ทุก 3 frames | 2 | 1 | 0.50* | 0.28 s |

`*` LOW_SPEC ได้ 1 call เช่นเดียวกับ BALANCED เพราะไฟล์มีเพียง 2 frames เท่านั้น หากเป็น stream ยาว อัตราเป้าหมายคือประมาณ 1/3 ของ captured frames

## ความหมาย

- การลดจำนวน MediaPipe calls ช่วยลดภาระ CPU ที่ inference แต่ยังอ่านและแสดงทุก captured frame
- timestamp ใช้เวลาจริงของ frame; การข้าม frame ไม่ได้ลดระยะเวลา session และไม่ถูกนับเป็นหลักฐาน inference ใหม่
- ตัวเลขข้างต้นรวม overhead การเริ่ม MediaPipe และเล็กเกินไปสำหรับสรุป throughput ของระบบจริง
- เนื่องจากไฟล์ทดสอบว่าง จึงได้ pose detection 0/2 ซึ่งเป็นคุณสมบัติของ fixture ไม่ใช่ข้อสรุปว่าระบบตรวจคนไม่ได้
- ยังไม่มีผลเฉพาะ Acer จนกว่าจะรันบน Acer จริง จึงไม่อ้างว่าเป็น Acer FPS

## วิธีรันซ้ำ

```powershell
& '.\.venv\Scripts\python.exe' scripts\benchmark_performance_presets.py `
  --source outputs\videos\engine_blank_analyzed.mp4 `
  --max-frames 30
```

ผล JSON อยู่ที่ `outputs/metrics/performance_presets_offline.json` และใช้สำหรับตรวจ scheduling/MediaPipe call rate เท่านั้น ไม่ใช่ accuracy benchmark
