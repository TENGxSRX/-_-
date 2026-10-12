# Live Webcam Validation V1

สถานะ: **WAITING_FOR_USER_VALIDATION_SESSION**

มีการทดสอบ smoke กับ Camera 0 จริงแบบจำกัดแล้ว โดยเปิดกล้อง อ่านภาพ และ
cleanup สำเร็จ แต่เอกสารนี้ยังเว้นว่างจนกว่าผู้ใช้จะทำ validation 9 ขั้นตอน
ด้วยตนเอง ห้ามเติม posture measurement หรือ accuracy ที่ไม่ได้วัดจริง

| รายการวัด | ผลลัพธ์ | หมายเหตุ |
|---|---|---|
| camera index |  |  |
| camera input |  |  |
| MediaPipe detection |  |  |
| five-core acceptance |  |  |
| V4 TUP/TLB behavior |  |  |
| forward geometry |  |  |
| left geometry |  |  |
| right geometry |  |  |
| temporal warning |  |  |
| POSE_LOST |  |  |
| recovery |  |  |
| realtime FPS |  |  |
| PASS / NEEDS_CALIBRATION |  |  |

V4 ยังเป็น pilot สองคลาสเท่านั้น ส่วน geometry indicators ไม่ใช่ AI classes
เพิ่มเติม
