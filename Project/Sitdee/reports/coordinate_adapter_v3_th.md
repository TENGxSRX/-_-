# รายงาน Coordinate Adapter V3

## ข้อสรุป

สถานะความเข้ากันได้ของ RGB จริง: **PARTIALLY_VALIDATED**

เอกสารต้นฉบับสนับสนุนว่า MultiPosture ใช้ MediaPipe World Landmarks หน่วยเมตร และมี origin ที่กึ่งกลางสะโพก การตรวจตัวเลขยืนยันว่า hip midpoint มี residual ใกล้ศูนย์อย่างเสถียรทั้งใน CSV และ world output ของ MediaPipe แต่ CSV ทางการที่มีจริงมี 99 coordinate columns ขัดกับคำอธิบายใน paper ที่ระบุ model input 33 ค่า และไม่พบ StandardScaler artifact จึงยังไม่ใช่การ validate แบบสมบูรณ์

## คำตอบหลัก

- V3 ที่มีหลักฐานรองรับดีที่สุด: `adapter_v3_a_world_hip_centered`
- ไม่พบหลักฐานให้ทำ geometric torso scaling, mirror หรือ z-flip
- image-normalized V2 เดิมไม่ตรงกับ source world-meter semantics
- SS → TLL ยังอธิบายได้ดีที่สุดด้วย combination ของ schema/model-input mismatch, scaler provenance, camera/domain gap และ/หรือ semantic label gap
- ยังไม่ retrain V1/V2 และยังไม่ฝึก ANYA
- ยังไม่ติดตั้ง YOLO เพราะ MediaPipe detection/core visibility ไม่ใช่ blocker
- ต้องเก็บวิดีโอ office-view ที่มี label จริงเพื่อ calibration ขั้นสุดท้าย

รายละเอียดภาษาอังกฤษและตัวเลขทั้งหมดอยู่ใน `reports/coordinate_adapter_v3.md` และรายงานย่อยในโฟลเดอร์เดียวกัน
