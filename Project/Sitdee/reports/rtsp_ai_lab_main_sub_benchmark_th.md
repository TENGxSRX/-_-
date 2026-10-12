# RTSP AI Lab MAIN/SUB latency benchmark

สถานะ: **PASS**
Preset: `RTSP_LOW_LATENCY` | ระยะเวลาประมาณ 6.0 วินาทีต่อ profile

การทดสอบนี้รับเฟรมแบบ latest-frame สำหรับ RTSP, ไม่บันทึกวิดีโอ, ไม่เก็บ human-labelled data และไม่ train model

| Profile | Result | Resolution | Source FPS | AI FPS | Median frame age | Median inference | Pose detected | Core accepted | Dropped old | Reconnects |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HIKVISION_MAIN | PASS | 2560x1440 | 20.39 | 14.12 | 0.0 ms | 23.9 ms | 31.7% | 31.7% | 181 | 0 |
| HIKVISION_SUB | PASS | 640x360 | 20.18 | 14.57 | 16.0 ms | 62.8 ms | 56.6% | 56.6% | 141 | 0 |

## การตีความ

ตัวเลข frame age เป็นเวลาที่วัดได้จากการ publish ในช่อง latest-frame ถึงเวลาที่ AI loop อ่าน ไม่ใช่ timestamp จากตัวกล้องโดยตรง เพราะ OpenCV ไม่เปิดเผย timestamp นั้น
การเลือก MAIN หรือ SUB ต้องพิจารณาความสดของเฟรมควบคู่กับ resolution และ core landmark acceptance ไม่เลือกจาก FPS อย่างเดียว

JSON: `outputs\metrics\rtsp_ai_lab_main_sub_benchmark.json`
