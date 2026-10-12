# ผลการพัฒนา ERGONOMIC POSTURE ENGINE V1

ตรวจสอบวันที่ 2026-09-05

## สิ่งที่พัฒนาแล้ว

- โครงสร้าง landmark มาตรฐาน 33 จุด พร้อม x/y/z/visibility
- adapter จาก MediaPipe ไปยัง MultiPosture ที่ตรวจลำดับ feature 126 ค่าแบบ
  strict
- geometry engine แบบ deterministic
- indicator สำหรับ head-forward, shoulder tilt และ slouch ที่ระบุเป็น
  **EXPERIMENTAL**
- probability smoothing, majority voting, minimum persistence, recovery,
  cooldown และสถานะ pose หาย
- event log แบบ JSONL, session summary และ annotated video
- interface `PoseBackend` เพื่อรองรับ backend อื่นในอนาคต เช่น YOLO Pose

## ผลที่ตรวจสอบได้

- unit/integration tests ทั้งหมดผ่าน **23 tests**
- image input และ prerecorded video input ทำงานผ่าน test
- สร้าง annotated MP4, event log และ session summary จาก video smoke test ได้
- ประมวลผลภาพ SitStraightNet 756 ภาพได้ประมาณ 59.89 ภาพ/วินาที และ RSS process
  เพิ่มจากประมาณ 189.43 MB เป็น 213.00 MB
- MediaPipe ตรวจ pose จากภาพชุดนี้ได้ 0/756 ภาพ เนื่องจากภาพมืด/เป็น silhouette
- ไม่คำนวณ accuracy เพราะ SitStraightNet ไม่มี ground truth ที่ยืนยันได้

## สถานะ

| ส่วนประกอบ | สถานะ |
|---|---|
| MultiPosture direction model | READY |
| Feature adapter: schema/order | READY |
| Feature adapter: semantic real-image compatibility | NOT VALIDATED |
| Geometry engine | READY เป็น geometric indicators |
| Head-forward / shoulder-tilt / slouch | EXPERIMENTAL |
| Temporal risk engine | READY เป็น project logic ที่ตั้งค่าได้ |
| Offline image pipeline | READY ด้านการประมวลผลไฟล์ แต่ detection ขึ้นกับ input |
| Offline video pipeline | READY สำหรับ prerecorded media |
| ANYA ergonomic model | BLOCKED — ไม่มี verified ground truth |
| YOLO Pose | NOT YET BENCHMARKED |
| Live webcam | NOT TESTED — HARDWARE UNAVAILABLE |

ค่าการเตือน 15 วินาทีและ high-duration 60 วินาทีเป็น product settings ของ
โครงการ ไม่ใช่ medical threshold และระบบนี้ไม่ใช่เครื่องมือวินิจฉัย Office
Syndrome หรือโรคใด ๆ
