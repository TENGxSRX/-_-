# ขั้นตอน validation สำหรับ Live Webcam Prototype V1

สถานะปัจจุบัน: **REAL SESSION ANALYZED — NEEDS_CALIBRATION**

## คำสั่ง

1. เสียบ webcam
2. เปิด `check_camera.bat`
3. ใช้เลขกล้องที่ขึ้น `READY`
4. เปิด `run_webcam_demo.bat` ถ้าเป็นกล้อง 0 หรือรันคำสั่งเดียวกันโดยเปลี่ยน `--source` ถ้าเป็นกล้อง 1/2
5. รอหน้า CAMERA SETUP และขยับจน BODY VISIBILITY เป็นเครื่องหมายถูกครบต่อเนื่อง
6. กด `SPACE` เพื่อเริ่ม neutral calibration แล้วนั่งตรงตามธรรมชาติประมาณ 4 วินาที
7. หลัง calibration เสร็จ กด `SPACE` ก่อนเริ่มแต่ละขั้นตามคำสั่งบนจอ
8. สำหรับ step 8 ต้องกลับเข้าภาพและรอ core landmarks กลับมาก่อนกด `SPACE`
9. กด `Q`/`ESC` เพื่อจบก่อนเมื่อจำเป็น

## สิ่งที่จะวัดแยกกัน

- กล้องอ่านภาพได้หรือไม่
- MediaPipe ตรวจพบ pose หรือไม่
- landmark หลัก 5 จุดผ่าน visibility gate หรือไม่
- V4 แสดง `UPRIGHT`/`LEAN BACKWARD` อย่างไร
- geometry forward/left/right เปลี่ยนหรือไม่
- temporal warning ทำงานหรือไม่
- `POSE_LOST` และการกลับเข้าภาพทำงานหรือไม่
- active-step timer หยุดเมื่อขึ้น `INPUT LOST` หรือไม่
- FPS แบบ realtime

ไม่มีการสร้าง five-class accuracy จาก validation sequence และไม่มีการ
retrain จากเฟรมที่ผู้ใช้ทำ

ภาพ preview ถูก mirror เฉพาะตอนแสดงผล ไม่ได้เปลี่ยนภาพที่ใช้กับ MediaPipe,
V4 หรือ geometry

## PASS เบื้องต้น

ถือว่า software/hardware smoke test ผ่านเมื่อกล้องเปิดได้ ภาพต่อเนื่อง
MediaPipe ทำงาน มีเฟรมที่ core landmarks ผ่าน ระบบแสดง overlay และกด Q/ESC
แล้วปล่อยกล้องได้ พร้อมสร้าง JSONL และ session summary

ผลอาจเป็น `NEEDS_CALIBRATION` หาก neutral baseline เก็บไม่ครบ, geometry
ซ้าย/ขวาหรือ forward แสดงกลับด้าน, แกว่งมาก, core landmarks หลุดบ่อย หรือ
FPS ต่ำจนใช้งานไม่ได้ โดยห้ามแก้ threshold อัตโนมัติ ต้องบันทึกเหตุผลและ
version ของการเปลี่ยนแปลงก่อน
