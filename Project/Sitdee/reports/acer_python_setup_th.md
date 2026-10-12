# การตั้งค่า Python สำหรับ Acer

## เวอร์ชันที่แนะนำเพียงหนึ่งเดียว

ใช้ **Python 3.12.x แบบ 64-bit** โดย Desktop ที่ตรวจแล้วใช้ Python `3.12.10`
64-bit รุ่นเดียวกันนี้เข้ากันได้กับ MediaPipe, OpenCV และไฟล์โมเดล V4 ที่มีอยู่

ไม่ต้องติดตั้ง CUDA และไม่ต้องติดตั้งไลบรารีฝึกโมเดลหนัก เช่น XGBoost,
LightGBM หรือ PyTorch สำหรับงานบน Acer ชุดนี้

## ขั้นตอนครั้งแรก

1. แตก code-only bundle ไว้ในโฟลเดอร์งาน เช่น
   `C:\Users\<ชื่อผู้ใช้>\Documents\office-posture-datasets`
2. เปิด PowerShell ที่โฟลเดอร์โครงการ
3. ตรวจว่า Python ถูกต้อง:

   ```powershell
   py -3.12 --version
   ```

4. รันไฟล์ตั้งค่า:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   .\setup_laptop_dev.ps1
   ```

สคริปต์จะสร้าง `.venv`, ติดตั้งเฉพาะ requirements แบบเบา, ตรวจ import,
โหลด V4 และรัน lightweight tests เท่านั้น จะไม่ดาวน์โหลด dataset และไม่เปิดกล้อง
เอง

## ครั้งต่อ ๆ ไป

ดับเบิลคลิก `start_laptop_dev.bat` หรือใช้:

```powershell
.\.venv\Scripts\python.exe scripts\laptop_health_check.py
.\.venv\Scripts\python.exe scripts\run_laptop_tests.py
```

## เงื่อนไขพื้นที่

สคริปต์จะแจ้งเตือนเมื่อพื้นที่ว่างต่ำกว่า 15 GiB และจะไม่ลบไฟล์ให้อัตโนมัติ
อย่าดาวน์โหลด ANYA, DriveNet หรือ archive ขนาดใหญ่ลง Acer
