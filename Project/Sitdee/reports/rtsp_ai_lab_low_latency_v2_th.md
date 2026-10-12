# RTSP AI Lab Low-Latency V2

วันที่วัด: 2026-09-09  
เครื่อง: `DESKTOP-ALH6CUB`  
กล้อง: Hikvision profile ที่ตั้งค่าไว้ในเครื่อง (ไม่แสดง credential)

## สถานะ

`RTSP_AI_LAB_READY_FOR_LIVE_VIEW`

การวัดด้านล่างเป็น short engineering benchmark จริง ไม่มีการบันทึกวิดีโอ
ไม่มีการเก็บ human-labelled sample และไม่มีการ train โมเดล

## MAIN เทียบกับ SUB

| ค่า | HIKVISION_MAIN | HIKVISION_SUB |
|---|---:|---:|
| Source resolution | 2560×1440 | 640×360 |
| Source FPS ที่วัดได้ | 20.39 | 20.18 |
| AI inference copy | 960×540 | 640×360 (ไม่ต้องย่อ) |
| AI FPS ที่วัดได้ | 14.12 | 14.57 |
| Median frame age | 0.0 ms | 16.0 ms |
| Frame age p95 | 47.0 ms | 53.8 ms |
| Median inference time | 23.9 ms | 62.8 ms |
| Frames captured | 182 | 142 |
| Frames inferred | 92 | 92 |
| Old frames dropped | 181 | 141 |
| Decode failures | 0 | 0 |
| Reconnects | 0 | 0 |
| Stale events | 1 | 0 |
| Pose detected | 31.7% | 56.6% |
| Core accepted | 31.7% | 56.6% |

ตัวเลข pose/core ในรอบนี้ไม่ใช่ accuracy ของ V4 และอาจสะท้อนว่าช่วงที่วัดไม่
มีคนอยู่ในภาพหรือมุม/การบังร่างกายไม่เหมาะสม จึงต้องมีคนอยู่ในพื้นที่และทำซ้ำ
ก่อนใช้เป็น camera-placement conclusion

## วัดแบบเปิดหน้าจอจริง

วัด `HIKVISION_MAIN` อีกครั้ง 4 วินาทีโดยเปิดหน้าต่าง AI Lab และใช้ display copy
1280×720 ได้ผล: session `4.35 s`, source `19.94 FPS`, AI/display `8.73 FPS`,
median frame age `23.5 ms` (p95 `62.0 ms`), median inference `72.6 ms`,
median render `20.9 ms`, captured `139`, inferred/displayed `38`, และ old frames
dropped `138`. Decode failure และ reconnect เป็น `0` และ stale event เป็น `0`.
การย่อ display จากภาพต้นฉบับ 2560×1440 ช่วยลด render จากรอบก่อนประมาณ `80.2 ms`
เหลือ `20.9 ms` แต่รอบนี้ pose/core acceptance `42.4%` ยังเป็นเพียง engineering
telemetry ไม่ใช่ accuracy หรือ real-room validation. รอบนี้เก็บไว้ที่
`outputs/rtsp_ai_display_benchmark_v3/` และไม่มี recording

ตัวเลข headless benchmark กับ visible benchmark ไม่ควรนำมาเทียบตรง ๆ เพราะการ
แสดงผล GUI เพิ่มภาระ render และ scene/จังหวะเฟรมของกล้องอาจต่างกัน

## สิ่งที่แก้ใน pipeline

เดิม capture และ inference อยู่ใน loop เดียวกัน ทำให้การอ่าน source มีโอกาส
ตามหลังการประมวลผล ระบบใหม่สำหรับ RTSP ใช้:

```text
RTSP capture thread
    → latest-frame slot ขนาด 1
    → เฟรมเก่าถูกแทนที่/ทิ้ง
    → AI อ่านเฟรมใหม่สุด
    → resize copy → MediaPipe → frozen V4 → UI
```

ไม่สร้าง queue ที่โตขึ้นเรื่อย ๆ ค่า `old frames dropped` จึงเป็นพฤติกรรมที่
ตั้งใจ เพื่อรักษาความสดของภาพ เวลาปิดระบบจะหยุด reader และ release capture
อย่างปลอดภัย

## ความละเอียด inference

MAIN ยังรับภาพต้นฉบับ 2560×1440 แต่ส่งสำเนา 960×540 เข้า MediaPipe โดยรักษา
อัตราส่วนภาพและไม่เปลี่ยน world-landmark/V4 semantics ส่วน SUB ต้นฉบับ 640×360
อยู่ต่ำกว่าค่าจำกัดจึงใช้ขนาดเดิม

มีการวัด MAIN ที่ 1280×720 เพิ่ม: median inference `19.3 ms`, AI FPS `15.34`
แต่พบ pose `0/98` เฟรมใน session นั้น จึงยังเปรียบเทียบคุณภาพกับ 960 ไม่ได้
และไม่ใช้ผลดังกล่าวเพื่ออ้างว่า 1280 ดีกว่า

## Metrics ที่เพิ่ม

แต่ละ AI record/summary รองรับ:

- source FPS, display FPS, inference FPS
- frame/capture age (เวลาที่วัดได้จากการ publish ถึง AI read)
- inference time และ render time
- frames captured/inferred/dropped
- decode failures, reconnects และ stale events

OpenCV ไม่ให้ camera capture timestamp โดยตรง จึงไม่เรียก frame age นี้ว่า
network latency แบบสมบูรณ์

## Clean View

หน้าจอเริ่มต้นใหม่แสดงเฉพาะ:

- `AI POSTURE` พร้อม `TUP` หรือ `TLB` และ confidence ใหญ่
- `POSE: GOOD/PARTIAL/LOST` และ core validity
- FPS, AI FPS, frame age และ calibration status
- `GEOMETRY / EXPERIMENTAL` แยกจาก AI
- temporal state
- landmark หลัก 5 จุดและเส้นไหล่/ลำตัว/สะโพก

ปุ่ม:

- `D` เปิด/ปิด Debug
- `L` เปิด/ปิด landmark ทั้งชุด
- `Q` หรือ `ESC` ออก

ข้อความไม่ถูก mirror ค่าเริ่มต้นเป็น surveillance view ไม่ mirror; หากใช้
`--mirror` จะ mirror เฉพาะภาพและคำนวณตำแหน่ง overlay ให้ข้อความอ่านได้

## การเลือก default

กำหนด `RTSP_LOW_LATENCY` เป็น default ของ AI Lab ใช้ latest-frame และ inference
copy 960×540 สำหรับ MAIN การเลือกนี้ยึดความสดของเฟรมและการไม่สะสม backlog เป็น
หลัก แต่ pose/core acceptance รอบ benchmark ยังต่ำ จึงยังไม่ใช่ผลยืนยันคุณภาพ
สำหรับท่านั่งจริง

MAIN เป็นตัวเลือกที่เหมาะกว่าเมื่อจำเป็นต้องเก็บรายละเอียดร่างกายหลายคนหรือ
ต้องการภาพต้นฉบับสำหรับ UI/debug; SUB เบากว่าและเร็วขึ้นเล็กน้อย แต่ความละเอียด
640×360 อาจไม่พอเมื่อมีหลายคนหรืออยู่ไกล การตัดสินใจสุดท้ายควรทำซ้ำขณะมีคน
นั่งจริงในพื้นที่เป้าหมาย

สำหรับการทดลอง AI Lab รอบนี้ให้เริ่มที่ MAIN + inference 960×540 เพื่อรักษา
รายละเอียดและเผื่อการขยายไปหลายคน ส่วน SUB เหมาะเป็นโหมดประหยัดเมื่อทดสอบว่า
ความละเอียด 640×360 ยังให้ core landmarks ที่เสถียรหรือไม่

## Integrity และข้อจำกัด

- V4 ไม่ถูกแก้ ไม่ retrain และ feature schema ไม่เปลี่ยน
- ไม่เปิด webcam source 0
- ไม่แก้ camera settings/network/Tailscale
- ไม่ train V5 และไม่สร้าง controlled samples
- AI prediction ยังมีแค่ TUP/TLB; Forward/Left/Right เป็น geometry indicators
- benchmark นี้ไม่ใช่ accuracy validation หรือ medical assessment

รายงานตัวเลขเครื่องจักร: `reports/rtsp_ai_lab_main_sub_benchmark_th.md` และ
`outputs/metrics/rtsp_ai_lab_main_sub_benchmark.json`
