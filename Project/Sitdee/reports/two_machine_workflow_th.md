# Workflow สองเครื่อง

## แบ่งหน้าที่

### Acer Aspire 4741G — งานเบา

- แก้ source code, geometry, temporal logic และ UI
- อ่านเอกสารและทำ presentation/demo
- รัน lightweight tests
- โหลด V4 และทดสอบโค้ดด้วย fixture หรือไฟล์เล็ก
- ทดสอบ MediaPipe แบบจำกัดเฟรมเมื่อจำเป็น

### Desktop — เครื่องหลัก

- เก็บ raw/processed datasets ทั้งหมด
- เก็บ ANYA, MultiPosture, MVSP และ DriveNet
- train/evaluate โมเดลและทำ benchmark เต็ม
- ประมวลผลวิดีโอยาวและทดสอบ webcam จริง
- เก็บผลทดลองและ telemetry ขนาดใหญ่

## กติกาข้อมูล

Desktop เป็น source of truth สำหรับข้อมูลและผลหนัก Acer ไม่ต้องมี
`datasets/` หรือ `validation_data/` เต็มชุด V4 ที่คัดลอกไป Acer เป็น runtime
artifact ขนาดเล็ก ไม่ใช่สิทธิ์ให้ฝึกหรือแก้โมเดลบน Acer

## สถานะ Git ตอนนี้

ตรวจแล้วโฟลเดอร์โครงการและโฟลเดอร์แม่ยังไม่มี `.git` และไม่มี remote ที่ยืนยันได้
จึงยังไม่ทำ `git init`, ไม่เพิ่ม remote และไม่ push เพื่อไม่สร้างประวัติหรือส่งข้อมูล
โดยไม่ได้รับการตรวจจากผู้ใช้

เมื่อพร้อมตั้ง Git repository ส่วนตัว ให้ทำบน Desktop ก่อน:

```powershell
git init
git add src scripts configs tests reports requirements-laptop.txt setup_laptop_dev.ps1 start_laptop_dev.bat fixtures README.md pytest.ini .gitignore outputs/models/rgb_native_v4 metadata/pose_landmarker_heavy.task
git status --short
```

ตรวจให้แน่ใจว่าไม่มี `datasets/`, `validation_data/`, secret หรือวิดีโอใน staged list
แล้วจึงเพิ่ม private remote และ push ตามนโยบายของผู้ใช้ การตั้ง remote/push เป็นขั้นตอน
ที่ต้องตรวจด้วยตนเอง เพราะยังไม่มี remote ปัจจุบันให้ยืนยัน

## หลังมี remote ที่ตรวจแล้ว

บน Acer:

```powershell
git pull --ff-only
git status --short
```

แก้เฉพาะ source/config/tests/docs แล้วตรวจ:

```powershell
git diff --check
git status --short
git add src scripts configs tests reports requirements-laptop.txt setup_laptop_dev.ps1 start_laptop_dev.bat fixtures README.md pytest.ini .gitignore
git diff --cached --stat
git commit -m "Improve lightweight development workflow"
git push
```

บน Desktop ให้ `git pull --ff-only`, รัน full test suite และตรวจ hash/model ก่อนงานหนัก
ถ้าไม่ต้องการตั้ง Git ตอนนี้ ให้ใช้ code-only ZIP เป็นช่องทางย้ายไฟล์ และห้ามส่ง dataset

## Remote Desktop

Acer สามารถ remote เข้า Desktop เพื่อสั่งงานหนักได้เมื่อจำเป็น ไม่ต้องเปิด port ใหม่
และห้ามเก็บรหัสผ่าน remote ไว้ใน repository
