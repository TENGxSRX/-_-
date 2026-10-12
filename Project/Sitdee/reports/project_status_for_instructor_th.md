# สรุปโครงงานสำหรับอาจารย์

## เป้าหมาย

สร้างต้นแบบวิเคราะห์ท่านั่งจากภาพกล้อง โดยแยกผล AI posture ออกจากตัวบ่งชี้ geometry และออกแบบให้ตรวจสอบย้อนกลับได้

## สิ่งที่สร้างแล้ว

- pipeline กล้อง RGB → MediaPipe Pose World Landmarks → feature → model → temporal warning → UI
- โมเดล MultiPosture และ RGB-Native V4 สำหรับ pilot แบบ 2 คลาส
- offline image/video inference และ session summary
- calibration ท่านั่งกลาง, ตรวจคุณภาพ landmark และโหมด LOW_SPEC
- dashboard อ่าน session ที่บันทึกไว้โดยไม่ต้องเปิดกล้อง
- room camera planner สำหรับประเมินตำแหน่งกล้องเมื่อมี floor plan จริง

## AI ปัจจุบัน

V4 เป็น pilot ที่ใช้ `TUP = Upright` และ `TLB = Lean Backward` เท่านั้น ส่วน Forward/Left/Right เป็น geometry indicators เชิงทดลอง ไม่ใช่คลาส AI ของ V4

MultiPosture มีโมเดลหลายคลาสสำหรับการทดลอง แต่ความเข้ากันได้ของ coordinate กับภาพ RGB จริงยังไม่ยืนยันสมบูรณ์ จึงไม่ควรอ้างว่าใช้แทนระบบกล้องจริงได้แล้ว

## Dataset และการเลือก MediaPipe

ใช้ MultiPosture เป็น baseline และตรวจ ANYA แยกต่างหาก ปัจจุบัน ANYA ยังไม่ train เพราะ label ใน release ที่ตรวจพบเป็น pseudo-label ที่ยังไม่มี ground truth ที่ยืนยันได้

เลือก MediaPipe เพราะ pipeline ปัจจุบันตรวจ pose ได้และเหมาะกับ prototype บน CPU; YOLO ยังไม่จำเป็น เนื่องจากปัญหาหลักคือ coordinate/domain compatibility ไม่ใช่การตรวจไม่พบ pose

## สิ่งที่ validated และสิ่งที่ยังทดลอง

- webcam hardware และการทำงานพื้นฐาน: ผ่านการตรวจ session จริงก่อนหน้านี้
- V4: โมเดลและโครงสร้าง pipeline พร้อม แต่ live TUP/TLB และ geometry ต้องเก็บ session จริงเพิ่ม
- multi-person: ยังเป็น single-person runtime
- room coverage: เป็น geometric planning estimate ยังไม่ใช่ผลติดตั้งจริง
- ไม่มีการอ้าง medical diagnosis และยังไม่มีการยืนยัน real-world accuracy แบบทั่วไป

## ขั้นถัดไป

พรุ่งนี้ใส่ floor plan จริงและข้อมูลโต๊ะ/เก้าอี้ จากนั้นใช้ planner จัดลำดับ candidate camera แล้วทดสอบภาพจริงที่เน้น nose, shoulders และ hips ก่อนตัดสินจำนวนกล้องหรือการเก็บ local RGB dataset เพิ่ม
