# วิเคราะห์ Calibration และ Failure จากการทดสอบกล้องจริง V1

สถานะ: **มีหลักฐานจากกล้องจริง — ต้องปรับ Calibration**

Session: `20260907T155515Z`  
กล้อง: index `0`  
ไฟล์: `outputs/live/logs/webcam_20260907T155515Z.jsonl` และ
`outputs/live/session_reports/webcam_20260907T155515Z_summary.json`

รายงานนี้ใช้ไฟล์ session ที่สร้างจริงเท่านั้น ไม่ retrain V4, ไม่ train V5 หรือ
ANYA, ไม่ติดตั้ง YOLO และไม่แก้ raw data/model

## สรุปสั้น

| ส่วน | ผล | ความหมาย |
|---|---:|---|
| กล้อง | PASS | Camera 0 เปิดและจบ session พร้อม cleanup ได้ |
| MediaPipe | 88.86% | ใช้ได้ดี แต่ยังไม่ครบพอสำหรับ validation สมบูรณ์ |
| จุดหลัก 5 จุด | 66.30% | NEEDS_CALIBRATION; ปัญหาหลักคือสะโพก |
| V4 TUP/TLB | NEEDS_VALIDATION | Step 2 ได้ TUP อย่างเดียว และ log รุ่นเดิมไม่มี probability/features |
| Geometry | NEEDS_CALIBRATION | head-depth proxy แบบ absolute เกิน threshold เกือบทุกเฟรม |
| Warning | NEEDS_RETEST | ไม่มี warning และยังไม่มี supported poor class ต่อเนื่องพอ |
| FPS | วัดได้ | เฉลี่ย 10.95 FPS เป็นข้อมูลวินิจฉัย ไม่ใช่คำรับรองคุณภาพ |

session นี้เป็นหลักฐานพฤติกรรมจากกล้องจริง แต่เป็น session ก่อนเพิ่ม neutral
baseline, stable-ready gate, การ pause เมื่อ input หาย และการรอ reacquisition
ของ step 8 ซึ่งได้เพิ่มไว้สำหรับรอบถัดไปแล้ว

## 1. ข้อเท็จจริงของ session

- records `1,249`
- duration ตาม summary `103.617 วินาที`
- pose detection `88.86%`
- five-core acceptance `66.30%`
- pose lost `33.11%`
- V4 output ประมาณ `TUP 94.75%`, `TLB 5.25%`
- forward indicator `98.87%` ของ valid-pose time
- left indicator `8.04%`
- right indicator `7.23%`
- warning `0`
- longest continuous poor posture `3.55 วินาที`
- mean FPS `10.95`

ใน JSONL มี `1,045` records ที่ตรวจพบ pose และ `808` records ที่ผ่าน core และ
ได้ prediction จาก V4 แต่ไม่มี `ai_probabilities`, ไม่มี V4 feature vector และ
ไม่มีวิดีโอที่บันทึกไว้ ดังนั้นจะไม่สร้างย้อนหลังเป็น probability curve หรือ
feature means ของ step 2

## 2. จุดที่ทำให้ core ไม่ผ่าน

วิเคราะห์จาก records ที่ pose ถูกตรวจพบ `1,045` records:

| Landmark | Fail | อัตรา fail | Mean visibility | Median | P10 | P25 | P75 | P90 | >=0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nose | 0/1045 | 0.00% | 0.9874 | 0.9980 | 0.9790 | 0.9943 | 0.9991 | 0.9994 | 100.00% |
| left shoulder | 0/1045 | 0.00% | 0.9975 | 0.9986 | 0.9941 | 0.9970 | 0.9992 | 0.9996 | 100.00% |
| right shoulder | 0/1045 | 0.00% | 0.9951 | 0.9956 | 0.9898 | 0.9935 | 0.9974 | 0.9993 | 100.00% |
| left hip | 229/1045 | 21.91% | 0.6694 | 0.7783 | 0.2309 | 0.5557 | 0.8656 | 0.9225 | 78.09% |
| right hip | 233/1045 | 22.30% | 0.6677 | 0.7418 | 0.2427 | 0.5644 | 0.8547 | 0.9032 | 77.70% |

ดังนั้น failure ทั้งหมดของ core มาจากสะโพก ไม่ใช่ศีรษะหรือไหล่ โดย step 3
สะโพกซ้าย fail `37/66`, ขวา `35/66` เหลือ core acceptance `43.94%`; step 4
สะโพกซ้าย `22/68`, ขวา `20/68` เหลือ `67.65%` ทั้งสอง step ยังตรวจพบ pose
ได้ `100%` จึงเป็นปัญหา framing/visibility เมื่อเอนตัว ไม่ใช่ person detection

### คำแนะนำจัดกล้อง

1. ถอยกล้องจนเห็นสะโพกทั้งสองข้างและมีขอบภาพเหลือ
2. เห็นตั้งแต่ศีรษะถึงสะโพกเต็มภาพก่อนกด SPACE
3. ใช้มุมกว้างขึ้นถ้ามี โดยยังคงความละเอียดพอสำหรับศีรษะ/ไหล่
4. วางกล้องใกล้ระดับหน้าอกหรือไหล่ ไม่วางต่ำเงยมาก
5. ใช้มุมตรงหรือเฉียงเล็กน้อย และอย่าให้โต๊ะ/ที่วางแขนบังสะโพก
6. ให้แสงด้านหน้าและลำตัวสว่างสม่ำเสมอ

เป็นคำแนะนำด้าน framing ไม่ใช่ระยะวัดทางการแพทย์

## 3. ทำไม Forward indicator ขึ้นเกือบ 98.9%

โค้ดปัจจุบันใช้กฎ:

`abs(head_forward_proxy) >= 0.20`

โดย `head_forward_proxy` คือค่า z ระหว่างจมูกกับจุดกึ่งกลางไหล่ที่ normalize
แล้ว ไม่ใช่ trunk z โดยตรง ค่าที่วัดได้:

| ค่า | Step 1 | Step 2 | valid ทั้งหมด |
|---|---:|---:|---:|
| head_forward_proxy | -0.6414 | -0.5876 | mean -0.5733, median -0.5591 |
| trunk z / forward_back_component | -0.1639 | -0.1379 | mean -0.1257 |
| trunk inclination | -0.4476° | -0.2085° | mean -2.0418° |

แม้นั่งตรงใน step 1 ค่า head-depth proxy ก็มี absolute value ประมาณ `0.64`
ซึ่งเกิน `0.20` อยู่แล้ว การกลับเครื่องหมายอย่างเดียวจึงไม่ช่วย เพราะยังเกิน
เมื่อใช้ absolute value ปัญหาหลักคือใช้ค่ากล้อง-relative แบบ absolute โดยไม่มี
ค่า neutral ของ session และชื่อ `forward_back_component` ใน calibration อาจทำให้
เข้าใจผิด เพราะ flag จริงใช้ `head_forward_proxy`

จึงเพิ่ม neutral calibration เพื่อหาค่า baseline ตอนนั่งตรง แล้วลบเฉพาะใน
geometry indicators โดยไม่แตะ raw coordinate และไม่แตะ V4 model

## 4. ทำไม Step 2 เอนไปหลังแต่ V4 ยังเป็น TUP

step 2 มี pose/core ผ่าน `71/71` records และ V4 ได้ `TUP` ทั้งหมด confidence:

- mean `0.999904`
- median `0.999905`
- minimum `0.999712`

แต่ log รุ่นเดิมไม่มี probability แยก TUP/TLB, ไม่มี feature vector และไม่มี
วิดีโอ จึงแยกไม่ได้แน่ชัดว่าเป็นการเอนน้อยไปหรือเป็น domain gap จากกล้องจริง

ค่าที่เปลี่ยนจาก step 1 ไป step 2 มีไม่มาก:

- head-depth proxy `-0.6414` เป็น `-0.5876`
- trunk z `-0.1639` เป็น `-0.1379`
- trunk inclination `-0.4476°` เป็น `-0.2085°`

จึงจัดเป็น **INCONCLUSIVE** ไม่กล่าวหา V4 เกินหลักฐาน รอบถัดไป logger จะเก็บ
probability ทั้งแบบทันทีและแบบ smooth รวมถึง V4 feature names/values ตามลำดับ

## 5. ทำไม Step 8 กลับเข้าภาพไม่สำเร็จ

- step 8 เริ่ม `73.550 s`, frame `841` โดยยังไม่เห็น pose/core
- มี accepted เพียงหนึ่งเฟรมที่ `74.740 s`, frame `862`
- หายอีกครั้งที่ `74.857 s`, frame `863`
- จบ step ที่ `78.590 s` ขณะที่เฟรมที่เหลือไม่ผ่าน

flow เดิมเริ่มจับเวลา step ทันทีและไม่รอ stable reacquisition จึงไม่ควรตีความว่า
ผู้ใช้หรือ MediaPipe ล้มเหลวถาวร

flow ใหม่จะรอ `WAITING`, ต้องเห็น core ต่อเนื่องก่อน SPACE, block step 8 จนเห็น
core กลับมา และ pause timer เมื่อ input หาย

## 6. ทำไม Warning ไม่ขึ้น

ค่าที่ใช้คือ warning `15 วินาที`, high risk `60 วินาที`, recovery `3 วินาที`,
smoothing window `5` และ persistence `0.5 วินาที`

Temporal engine นับ poor direction เฉพาะ `TLF/TLB/TLL/TLR` ขณะที่ V4 มีแค่
`TUP/TLB` และ forward/left/right เป็น geometry ทดลอง ไม่ได้กลายเป็น poor model
direction อัตโนมัติ Step 9 มี TUP `245` records, TLB เพียง `1` และ pose lost
`56` records ทำให้ max poor duration ของ step 9 เป็น `0.0 วินาที` จึงไม่มี warning

ไม่ได้ลด threshold เพื่อบังคับให้ผ่าน รอบถัดไปต้องทดสอบ supported poor direction
ต่อเนื่องจริง หรือออกแบบเป็น geometry-warning experiment แยกต่างหาก

## 7. สิ่งที่เพิ่มแล้ว

- stable core gate `1.5 วินาที`
- neutral calibration `4.0 วินาที`
- เก็บ baseline แบบ runtime/session เท่านั้น ไม่ train และไม่สร้าง label
- geometry relative ต่อ neutral โดยไม่เปลี่ยน V4
- timer หยุดเมื่อ `INPUT LOST`
- step 8 รอ core reacquisition และ SPACE
- UI แสดง `INPUT GOOD / INPUT LOST`
- log รอบใหม่เก็บ TUP/TLB probabilities, smoothed probabilities และ V4 features

## 8. รอบทดสอบถัดไป

ให้ดับเบิลคลิก:

`run_webcam_calibration.bat`

flow คือ:

`CAMERA SETUP -> core stable -> SPACE -> NEUTRAL CALIBRATION -> กด SPACE แต่ละ step -> countdown 3 วินาที -> วัดผล`

เมื่อถึง step 8 ให้กลับเข้าภาพจน panel แสดง core ครบทุกจุดก่อนกด SPACE หาก input
หายระหว่าง step timer ควรหยุด ไม่ควรนับเป็นหลักฐาน posture

## สถานะสุดท้าย

WEBCAM HARDWARE: **PASS**  
MEDIAPIPE DETECTION: **PROMISING**  
CORE LANDMARK QUALITY: **NEEDS_CALIBRATION**  
V4 LIVE TUP/TLB: **NEEDS_VALIDATION**  
GEOMETRY: **NEEDS_CALIBRATION**  
TEMPORAL WARNING: **NEEDS_RETEST**  
LIVE FPS: **MEASURED — 10.95 FPS**

ระบบยัง **ไม่ถือว่า fully validated** ต้องทำ user-guided calibration/validation
อีกหนึ่งรอบด้วย flow ใหม่

หลังเพิ่ม safeguards ชุดทดสอบอัตโนมัติผ่าน **85/85** และไม่ได้แก้ V4, V5,
ANYA หรือ raw datasets
