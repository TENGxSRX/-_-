# ขั้นตอนใช้งาน Acer รายวัน

## วันแรก

1. แตก code-only bundle ลงโฟลเดอร์งาน
2. เปิด PowerShell ที่โฟลเดอร์นั้น
3. รันคำสั่งนี้เป็นคำสั่งแรก:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   .\setup_laptop_dev.ps1
   ```

4. เมื่อเสร็จแล้วเปิด `start_laptop_dev.bat`
5. ตรวจผล `LAPTOP_DEV_READY`
6. รัน `scripts\run_laptop_tests.py` หากต้องการรันซ้ำ
7. เปิด `README.md` และเริ่มงาน UI/เอกสาร/logic

## วันต่อ ๆ ไป ถ้ามี Git แล้ว

เปิด PowerShell แล้วใช้:

```powershell
cd C:\Users\<ชื่อผู้ใช้>\Documents\office-posture-datasets
git pull --ff-only
.\.venv\Scripts\python.exe scripts\laptop_health_check.py
.\.venv\Scripts\python.exe scripts\run_laptop_tests.py
```

หลังแก้ไฟล์:

```powershell
git diff --check
git status --short
git add src scripts configs tests reports fixtures requirements-laptop.txt setup_laptop_dev.ps1 start_laptop_dev.bat README.md pytest.ini .gitignore
git diff --cached --stat
git commit -m "Describe the change"
git push
```

ถ้ายังไม่มี Git ให้แก้ไฟล์ใน bundle แล้วส่งเฉพาะ code-only bundle กลับ Desktop
ห้าม copy โฟลเดอร์ทั้งโครงการ เพราะอาจนำ raw data ติดไปด้วย

## งานที่ Acer ทำได้สบาย

- แก้ Python/UI และเอกสาร
- รัน geometry/temporal logic
- โหลด V4
- รัน unit tests และ fixture tests
- ทดลอง MediaPipe แบบ LOW_SPEC และจำกัดเฟรม

## งานที่ควรส่งกลับไป Desktop

- ฝึกหรือ tune โมเดล
- อ่าน ANYA หรือ dataset ใด ๆ ขนาดใหญ่
- ประมวลผล DriveNet เต็มชุด
- benchmark ยาว
- export/retrain V1, V2, V4, V5 หรือ ANYA
- webcam validation จริงและการเก็บข้อมูลผู้ใช้

อย่ารัน training เป็นงานแรกบน Acer
