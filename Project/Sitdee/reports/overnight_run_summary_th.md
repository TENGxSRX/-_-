# สรุปการรัน Overnight

วันที่ 2026-09-06 เครื่อง `DESKTOP-ALH6CUB`

## ผลหลัก

รอบนี้ทำงาน safe phases สำเร็จโดยไม่แก้โมเดลที่ freeze และไม่แก้ raw dataset ผลที่ชัดที่สุดคือ MediaPipe และ V4 core adapter ทำงานได้ดีเชิงการตรวจจับ: dense DriveNet ผ่าน pose/core 100% ส่วนข้อจำกัดที่เหลือเป็นเรื่องจำนวนคลาส ข้อมูลข้ามโดเมน และความหมายของ feature ไม่ใช่ pose detection

## Track A — ค้นหาชุดข้อมูลภายนอก

ผลการตัดสินใจคือ **NO_VALID_EXTERNAL_5CLASS_DATA** ยังไม่พบ archive RGB ภายนอกที่ยืนยัน label join และ participant grouping ได้ครบสำหรับ TLF/TLL/TLR จึงไม่ดาวน์โหลด archive ใหม่

MVSP/LGCSPNet ยังเป็น manual-access candidate ที่มีคุณค่าที่สุด เพราะหมวดหมู่ในบทความมี upright/back/left/right แต่ README ทางการชี้ไป Baidu และยังตรวจ archive, label join, participant IDs และ license ในเครื่องไม่ได้ คำว่า `down` เป็นท่า forward ร่วมกับก้มตัว จึงไม่แปลงเป็น TLF อัตโนมัติ MOPED25 เป็นเพียง research lead ยังไม่ใช่ชุดที่นำมาใช้แทนได้ทันที

## Track B — benchmark V4 แบบ frozen

- อ่าน source frames ครบ **12,395 เฟรม** จาก 6 คลิป
- dense MediaPipe/V4 ที่ 10 FPS: **4,134 เฟรม**
- pose detected: **4,134/4,134 = 100%**
- core landmark ผ่าน: **4,134/4,134 = 100%**
- POSE_LOST ที่ 1/2/5/10 FPS: **0**
- จำนวนแถว 1/2/5/10 FPS: **417 / 830 / 2,069 / 4,134**
- raw class switches ที่ 5 FPS: **3**, หลัง smoothing เหลือ **1**
- เวลา dense pass **500.21 วินาที**, RSS `183.3 → 222.9 MB`

เนื่องจาก DriveNet เป็น source เดียวกับที่ฝึก V4 จึงรายงาน prediction เป็น diagnostic ไม่เรียกว่า external accuracy ความผิดปกติที่ควรติดตามคือ SS_2 มี 2/355 เฟรมถูกทายเป็น TLB และ SS_3 มี 3 feature dimensions ที่อยู่นอก training range เกินครึ่งของแถว แม้ pose/core ผ่านทั้งหมด

## Track C — feature และ confound audit

เปรียบเทียบ participant 1/2 ซึ่งเป็น train domain กับ participant 3 ที่เป็น holdout พบ shift สูงสุดที่ `nose_y_world_hip_centered_scaled` SMD 1.212, hip-y สองข้าง 0.932 และ right-shoulder-y 0.925 ส่วนค่าสัมประสิทธิ์สูงสุดของ frozen LogisticRegression อยู่ใน shoulder x, nose x และ shoulder y

ผู้เข้าร่วมทุกคนมีทั้ง TUP และ TLB ดังนั้น participant ไม่ได้เท่ากับ label โดยตรง แต่ข้อมูลยังมีเพียง 6 วิดีโอและเฟรมติดกันมี temporal correlation จึงยังสรุป generalization กว้าง ๆ ไม่ได้ และรอบนี้ไม่ฝึกโมเดลใหม่

## Track D — architecture และการตัดสินใจ

- hybrid V4 + geometry demo: **เตรียมและ smoke-test แล้ว 60 เฟรมจาก SS_1** เป็น offline demo เท่านั้น geometry เป็นตัวชี้วัดเชิงกล้อง ไม่ใช่คะแนนทางการแพทย์
- V5: **ยังไม่ฝึก** เพราะยังขาดแถวที่มีหลักฐานสำหรับ TLF/TLL/TLR
- ANYA: **ยังไม่ฝึก** เพราะ release ปัจจุบันยังไม่มี target ที่ยืนยันได้
- coordinate adapter: **PARTIALLY_VALIDATED / ยังไม่ยืนยัน semantic compatibility กับ real RGB เต็มรูปแบบ**
- YOLO: **ยังไม่ติดตั้ง** เพราะหลักฐานปัจจุบันชี้ว่าปัญหาอยู่ที่ข้อมูล/coordinate domain ไม่ใช่ detection

## สถานะรวม

| ส่วนงาน | สถานะ |
|---|---|
| MediaPipe RGB pose detection | READY — 100% ของ 4,134 dense frames |
| V4 five-core extraction | READY — ผ่าน 100% |
| Geometry indicators | READY — experimental/camera-relative |
| Temporal smoothing | READY — 3 switches เหลือ 1 |
| MultiPosture V1/V2 | READY และเก็บไว้เดิม |
| MultiPosture V3 | เก็บไว้เดิม; semantics ยังไม่ resolved ทั้งหมด |
| RGB-native V4 | FROZEN PILOT — TUP/TLB เท่านั้น |
| RGB-native V5 | NOT TRAINED — ขาด TLF/TLL/TLR |
| ANYA | NOT TRAINED — target ยังไม่ยืนยัน |
| Hybrid V4 + geometry | EXPERIMENTAL OFFLINE DEMO READY |
| Offline image/video | READY และทดสอบแล้ว |
| Live webcam | NOT TESTED — HARDWARE NOT CURRENTLY AVAILABLE |
| YOLO | DEFERRED / ยังไม่ติดตั้ง |

## Integrity และ tests

- พื้นที่ว่าง C: หลังรัน **23.72 GB** และไม่มีการดาวน์โหลด archive ใหญ่ใหม่
- hash ของ ANYA 4 CSV ตรงกับ inventory เดิมทั้งหมด
- MultiPosture raw hash ยังเป็น `d21579200db9a57dd01c24fa6ece961b0be73aa0d5d402ae8e629761c03ebe38`
- DriveNet 6 คลิป hash ตรงกับ inventory ที่ยืนยันไว้
- test suite ผ่าน **65 tests ใน 9.68 วินาที**
- ไม่มีไฟล์ใต้ `datasets/raw/` ถูกแก้/เปลี่ยนชื่อ/ลบ และ V1–V4 ไม่ถูกเขียนทับ

## ขั้นตอนถัดไปที่แนะนำ

ทำ data action ที่ตรวจสอบได้หนึ่งทาง: ขอให้ผู้ใช้ตรวจ archive MVSP/LGCSPNet แบบ manual และยืนยัน label/participant/license หรือเก็บชุด RGB สำนักงานจากหลายคนให้ครบ 5 คลาส ก่อนหน้านั้นให้ใช้ V4 เป็น RGB pilot สองคลาส และใช้ MultiPosture เป็น research reference ห้าคลาสต่อไป ห้ามอ้างว่า real-camera five-class compatibility ผ่านแล้ว
