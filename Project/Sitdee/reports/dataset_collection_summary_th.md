# สรุปการรวบรวมชุดข้อมูลท่าทางการนั่งและท่าทางการทำงาน

วันที่ตรวจสอบ: 5 กันยายน 2026 (เวลาเอเชีย/กรุงเทพฯ)

เอกสารนี้บันทึกขั้นตอนรวบรวมข้อมูล ส่วนสถานะการฝึกโมเดล การประเมิน และ offline inference อยู่ใน `reports/final_project_status_th.md` ไฟล์ข้อมูลดิบที่ดาวน์โหลดมาไม่ได้ถูกแก้ไข

## ตรวจสอบคอมพิวเตอร์เครื่องนี้

- ชื่อเครื่อง: `DESKTOP-ALH6CUB`
- ระบบปฏิบัติการ: Microsoft Windows 11 Home รุ่น `10.0.26200`
- โฟลเดอร์ทำงานเดิม: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel`
- โฟลเดอร์โครงการใหม่: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets`
- พื้นที่ว่างในไดรฟ์ C หลังดาวน์โหลดและ audit ANYA: ประมาณ `25.51 GB`
- Python `3.12.10`, Git `2.55.0.windows.3`, Node `v22.14.0`
- ติดตั้ง Kaggle CLI/package version `2.2.4` ใน `.venv` แล้ว
- ตรวจสอบ authentication ด้วย official file-list command สำเร็จ และไม่ได้พิมพ์หรือคัดลอก credential ลง repository

## ตารางสรุป

| ชุดข้อมูล | แหล่งที่มา | สถานะ | สิ่งที่อยู่ในไฟล์ | ตัวอย่าง/แถว | ผู้เข้าร่วม | ป้ายกำกับ | ขนาด | ใบอนุญาต | การใช้งานที่เหมาะสม |
|---|---|---|---|---:|---:|---|---:|---|---|
| ANYA | Kaggle | AUDITED_TARGET_BLOCKED | CSV 4 ไฟล์ที่มี core observations เดียวกันและ prediction ต่อ algorithm | 605,955 ต่อไฟล์ | ไม่มี field ผู้เข้าร่วมที่ยืนยันได้ | ไม่มี ground truth ที่ยืนยันได้; `Predicted_Label` เป็น output | 7,966,483,972 ไบต์ | Apache 2.0 | แหล่งข้อมูล ergonomic ที่ตรงที่สุด แต่ต้องได้ mapping ของ target ก่อนฝึก |
| MultiPosture | Zenodo | COMPLETE | พิกัดจุดร่างกายแบบ 3 มิติและป้าย upper/lower body | 4,794 | 13 คน | upper 5 แบบ และ lower 7 แบบ | 9,327,935 ไบต์ | CC BY 4.0 | ชุดเริ่มต้นที่สมบูรณ์และเล็กสำหรับจำแนกท่านั่ง |
| SitPose | GitHub | PARTIAL | ไฟล์คุณลักษณะมุมข้อต่อและโค้ด | 5,746 แถวในไฟล์ที่มี; งานวิจัยรายงาน 33,409 | ไม่มีใน CSV; งานวิจัยรายงาน 36 คน | 7 แบบ | CSV 601,411 ไบต์ | ไม่ได้ระบุใน repository | ใช้เป็นตัวอย่าง baseline และตรวจสอบความครบถ้วน |
| SitStraightNet | GitHub | PARTIAL | ภาพ JPEG และโค้ดโมเดล | 756 ภาพใน repository; งานวิจัยรายงาน 17,596 เฟรม | ไม่มีรหัสผู้เข้าร่วมในไฟล์ | ไม่มีไฟล์ป้ายกำกับแยกในชุดที่ตรวจ | ภาพ 14,931,004 ไบต์ | MIT | ตัวอย่างข้อมูลภาพ ไม่ใช่ชุดข้อมูลฉบับเต็ม |
| MVSP / LGCSPNet | GitHub + Baidu Pan | MANUAL_ACTION_REQUIRED | repository มีเพียงคำแนะนำดาวน์โหลด | ยังตรวจสอบไม่ได้ | ยังตรวจสอบไม่ได้ | ยังตรวจสอบไม่ได้ | README 146 ไบต์ | ไม่ได้ระบุใน repository | ใช้ภายหลังเมื่อดาวน์โหลดด้วยตนเองสำเร็จ |

## ไฟล์ที่มีอยู่ในเครื่องนี้

- MultiPosture: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\datasets\raw\multiposture\data.csv`
- SitPose: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\datasets\raw\sitpose\official-repository\Azure Kinect\data_all.csv`
- SitStraightNet: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\datasets\raw\sitstraightnet\official-repository\dataset`
- MVSP: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\datasets\raw\mvsp\official-repository\README.md`
- ไฟล์ ANYA ทางการ: `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\datasets\raw\anya\official-download`

## ผลการตรวจสอบสำคัญ

### 1. ANYA

เมทาดาทาปัจจุบันของ Kaggle ระบุขนาดรวม 7,966,483,972 ไบต์ ใบอนุญาต Apache 2.0 และเป็นเวอร์ชัน 1 ที่อัปเดตวันที่ 21 ธันวาคม 2025 ดาวน์โหลดไฟล์ `ANN.csv`, `Random_Forest.csv`, `lgbm.csv` และ `xgb.csv` จาก official client สำเร็จ ตรวจแฮชและ audit แบบ streaming แล้ว พบว่า core observations เหมือนกัน แต่ `Predicted_Label` ต่างกัน และยังไม่มี ground truth ที่ยืนยันได้

### 2. MultiPosture

ดาวน์โหลด `data.csv` จาก Zenodo สำเร็จและตรวจสอบแล้ว มี 4,794 แถว 102 คอลัมน์ ผู้เข้าร่วม 13 คน และคุณลักษณะพิกัด 99 คอลัมน์ ซึ่งเป็นพิกัด `_x`, `_y`, `_z` ของจุด 33 จุด มีป้ายกำกับ upper body 5 แบบและ lower body 7 แบบ ไม่พบค่าว่าง แถวผิดรูปแบบ หรือแถวซ้ำแบบเหมือนกันทุกคอลัมน์

SHA-256 คือ `d21579200db9a57dd01c24fa6ece961b0be73aa0d5d402ae8e629761c03ebe38` และ MD5 คือ `7efbc94d653acf32eefecec3ff26d6c6` ซึ่งตรงกับ checksum ที่ Zenodo เผยแพร่ คำอธิบายในหน้า Zenodo ระบุ 4,800 เฟรมและ 11 joints/33 dimensions แต่ไฟล์จริงในเครื่องมี 4,794 แถวและ 99 coordinate features จึงบันทึกค่าจากไฟล์จริงไว้ตามนั้น

### 3. SitPose

repository อย่างเป็นทางการมีไฟล์ `data_all.csv` จำนวน 5,746 แถว 14 คอลัมน์ และป้ายกำกับ 7 แบบ ได้แก่ `forward`, `healthy`, `lay`, `left`, `right`, `stand`, `wrong` ไม่พบค่าว่างหรือแถวซ้ำ และไม่มีคอลัมน์รหัสผู้เข้าร่วม เซสชัน วิดีโอ หรือเฟรม ไฟล์ใน repository มีน้อยกว่าจำนวน 33,409 จุดที่รายงานในบทความ จึงจัดเป็น PARTIAL

### 4. SitStraightNet

repository มีภาพ JPEG ที่อ่านได้ 756 ภาพ ทุกภาพมีขนาด 640×480 รวม 14,931,004 ไบต์ ไม่พบไฟล์ป้ายกำกับ/annotation แยกหรือไฟล์วิดีโอในโฟลเดอร์ dataset และชื่อไฟล์เป็นชื่อทั่วไป เช่น `image001.jpg` จึงไม่สามารถอ่านป้ายกำกับจากชื่อไฟล์ได้ บทความรายงาน 17,596 เฟรมจาก 66 วิดีโอ จึงจัดเป็น PARTIAL

### 5. MVSP / LGCSPNet

repository ทางการมีเพียง README ขนาด 146 ไบต์ พร้อมลิงก์ Baidu Pan ยังไม่ได้ดาวน์โหลดชุดข้อมูล เพราะต้องลงชื่อเข้าใช้หรือทำการยืนยันด้วยตนเอง

ลิงก์ทางการคือ:

https://pan.baidu.com/s/1577RDm5dmdXtQqNUjwJusw?pwd=qm7x

ให้เปิดลิงก์ด้วยตนเอง ทำการ login หรือ verification ตามที่ Baidu ขอ ใช้รหัสแตกไฟล์สาธารณะที่ระบุใน README คือ `qm7x` แล้วบันทึกไฟล์ไว้ใต้ `datasets/raw/mvsp/manual-download/` ไม่มีการข้าม CAPTCHA หรือข้อจำกัดการเข้าถึง

## คำตอบสำหรับการนำเสนออาจารย์

1. ชุดข้อมูลที่ตรงกับ ergonomic posture โดยตรงที่สุด: **ANYA**
2. ชุดข้อมูลที่เริ่มพัฒนา AI ได้ง่ายที่สุดในระยะถัดไป: **MultiPosture** เพราะเป็นไฟล์เล็ก มีป้ายกำกับชัด มี keypoints พร้อมใช้ และไม่มีวิดีโอขนาดใหญ่
3. สามชุดที่ควรนำเสนออาจารย์ก่อน: **ANYA**, **MultiPosture**, และ **SitPose** โดยต้องอธิบายว่า ANYA ยังรอการยืนยันตัวตน, MultiPosture ดาวน์โหลดและตรวจ checksum แล้ว, และ SitPose เป็นข้อมูลบางส่วนของ repository สาธารณะ

## พื้นที่และข้อจำกัด

- พื้นที่ของโครงการที่ใช้อยู่จริงประมาณ `28,379,260 ไบต์` หรือ `27.06 MiB` รวมไฟล์ raw ที่ดาวน์โหลด repository metadata รายงาน และสคริปต์
- ANYA ดาวน์โหลดและ audit แบบ streaming เสร็จแล้ว แต่ยังฝึกไม่ได้จนกว่าจะมี ground-truth mapping ที่ยืนยันได้
- MVSP ยังไม่ได้ดาวน์โหลดเพราะต้องดำเนินการบน Baidu ด้วยตนเอง
- สถานะ COMPLETE ของ MultiPosture หมายถึงดาวน์โหลดไฟล์ทางการครบและ checksum ตรง ไม่ได้หมายความว่าคำอธิบาย 4,800 เฟรมบนหน้าเว็บตรงกับจำนวนแถวจริงทุกประการ
- SitPose และ SitStraightNet เป็น snapshot ของ repository ไม่ใช่หลักฐานว่าข้อมูลเต็มตามบทความอยู่ใน repository
- ไม่ได้ดาวน์โหลดชุดข้อมูลขนาดใหญ่ที่ไม่จำเป็น และไม่ได้ฝึกโมเดล

## แหล่งข้อมูล

- ANYA: https://www.kaggle.com/datasets/chanankornjandaeng/anya-a-dataset-for-ergonomic-posture-analysis
- Kaggle API: https://www.kaggle.com/docs/api
- MultiPosture: https://zenodo.org/records/14230872
- SitPose: https://github.com/TaylorDurden1114/SitPose
- บทความ SitPose: https://arxiv.org/abs/2412.12216
- SitStraightNet: https://github.com/realratchet/SitStraightNet
- บทความ SitStraightNet: https://pmc.ncbi.nlm.nih.gov/articles/PMC8022631/
- MVSP / LGCSPNet: https://github.com/LKW123800/multi-view-sitting-posture-dataset
