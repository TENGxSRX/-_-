# RTSP AI Lab MAIN/SUB latency benchmark

สถานะ: **PASS**
Preset: `RTSP_LOW_LATENCY` | ระยะเวลาประมาณ 6.0 วินาทีต่อ profile

การทดสอบนี้รับเฟรมแบบ latest-frame สำหรับ RTSP, ไม่บันทึกวิดีโอ, ไม่เก็บ human-labelled data และไม่ train model

| Profile | Result | Resolution | Source FPS | AI FPS | Median frame age | Median inference | Pose detected | Core accepted | Dropped old | Reconnects |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HIKVISION_MAIN | PASS | 2560x1440 | 14.50 | 15.34 | 0.0 ms | 19.3 ms | 0.0% | 0.0% | 183 | 0 |

## การตีความ

ตัวเลข frame age เป็นเวลาที่วัดได้จากการ publish ในช่อง latest-frame ถึงเวลาที่ AI loop อ่าน ไม่ใช่ timestamp จากตัวกล้องโดยตรง เพราะ OpenCV ไม่เปิดเผย timestamp นั้น
การเลือก MAIN หรือ SUB ต้องพิจารณาความสดของเฟรมควบคู่กับ resolution และ core landmark acceptance ไม่เลือกจาก FPS อย่างเดียว

JSON: `outputs\metrics\rtsp_ai_lab_main_1280_benchmark.json`
