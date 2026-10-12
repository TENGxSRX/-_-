# ผลการค้นหาชุดข้อมูล RGB ภายนอกระหว่าง Overnight

## ข้อสรุป

รอบนี้ได้ผลเป็น **NO_VALID_EXTERNAL_5CLASS_DATA** คือยังไม่ได้ชุดข้อมูล RGB ภายนอกที่ยืนยันได้ครบสำหรับคลาสที่หายไปทั้ง 5 คลาส จึงไม่ดาวน์โหลด archive ใหม่ พื้นที่ว่าง C: ก่อนเริ่มประมาณ 19.97 GB และแหล่งที่พบส่วนใหญ่ยังขาด label join ที่ตรวจสอบได้ เป็น RGB-D/เซนเซอร์ หรือเป็นแหล่งที่ต้องเข้าถึงแบบ manual

## การคัดกรอง

| แหล่ง | สิ่งที่ตรวจพบ | การตัดสินใจ |
|---|---|---|
| MultiPosture / Zenodo | มี pose coordinates และคลาส TUP/TLF/TLB/TLL/TLR แต่ไม่มี RGB ดิบ | ใช้เป็น reference โมเดล 5 คลาสเดิม ไม่เพิ่ม RGB |
| MVSP / LGCSPNet | บทความระบุภาพ RGB หลายมุมและมี upright/back/left/right/down เป็นต้น | เป็น **manual-access candidate** เพราะ README ชี้ไป Baidu; ยังไม่ bypass login/CAPTCHA และยังไม่ยืนยัน archive/label/license |
| DriveNet v6 | วิดีโอ RGB ที่ mapping ได้ SS→TUP และ LB→TLB | ใช้ benchmark เดิม ยังไม่มีคลาส TLF/TLL/TLR |
| MMD-MSD | subset ภาพ/มุมที่มี mapping TUP ได้ แต่ไม่มี label sidecar รายไฟล์ที่ครบ | ใช้ audit/TUP เท่านั้น |
| SitStraightNet | งาน RGB-D มี straight/hunched/lying แต่ snapshot ไม่มี label join รายไฟล์ที่เชื่อถือได้ | smoke/audit เท่านั้น |
| CCTV Exam Monitor | มีคำว่า forward/backward/left/right ใน source vocabulary แต่ไฟล์ local เป็น hashed JPG และไม่มี sidecar ที่ยืนยัน label | audit เท่านั้น |
| SitPose และงาน pressure/sensor | ไม่ใช่ normal RGB ที่เข้ากับ MediaPipe pipeline | ไม่ใช้กับเป้าหมายนี้ |
| MOPED25 | มีวิดีโอและ kinematics ในงาน occupational แต่ไม่ใช่ drop-in five-class sitting RGB | เป็น lead สำหรับศึกษาต่อ ไม่ดาวน์โหลดรอบนี้ |

คำว่า `down` ใน MVSP เป็น “เอนไปข้างหน้าและก้มตัว” ซึ่งเป็นป้ายแบบผสม จึงไม่แปลงเป็น TLF โดยอัตโนมัติ และจะไม่สร้าง pseudo-label หรือเดาป้ายกำกับ

## แหล่งที่ผู้ใช้ต้องเข้าถึงด้วยตนเองในอนาคต

1. archive MVSP/LGCSPNet ทางการบน Baidu เพื่อยืนยันไฟล์สื่อ, label join, participant/view IDs และ license
2. MOPED25 เพื่อดูว่ามีส่วน seated RGB ที่มี mapping และ license ใช้ได้จริงหรือไม่

ไม่มีการดาวน์โหลด archive ใหม่ และ raw data เดิมรวมถึง backup V1–V4 ไม่ถูกแก้ไข
