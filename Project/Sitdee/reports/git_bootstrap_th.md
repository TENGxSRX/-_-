# การเตรียม Git อย่างปลอดภัย

สถานะปัจจุบัน: **ยังไม่มี Git repository และยังไม่มี remote**

เตรียม `.gitignore` ไว้แล้ว แต่คืนนี้ยังไม่สั่ง `git init`, ยังไม่สร้าง commit และยังไม่ push ออก Internet

## เมื่อพร้อมเริ่ม repository ในเครื่อง

เปิด PowerShell ที่ project root แล้วตรวจสอบก่อน:

```powershell
Set-Location "C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets"
git init
git status --short
```

จากนั้นตรวจว่าไม่มี dataset, video, token, `.venv`, log ส่วนตัว หรือ model artifact ที่ไม่ตั้งใจจะบันทึก ก่อนเลือกไฟล์ที่จะ checkpoint:

```powershell
git add .gitignore README.md src scripts tests configs fixtures reports
git status
git commit -m "Initial auditable office posture prototype"
```

ไม่ควรใช้ `git add .` โดยไม่อ่าน `git status` ก่อน โดยเฉพาะเมื่อมีข้อมูล raw/participant recording ในเครื่อง

## ก่อนใช้ remote

- ตรวจชื่อ repository และสิทธิ์ให้ชัดเจน
- ตรวจ `.gitignore` อีกครั้งด้วย `git status --ignored`
- ห้ามใส่ Kaggle token, `.env`, `kaggle.json`, participant video หรือ raw dataset
- ค่อยเพิ่ม remote และ push เมื่อผู้ใช้สั่งโดยตรง

การมี Git จะช่วยย้อนดู code/report ได้ แต่ไม่ควรใช้แทนการเก็บ raw data และ hash manifest ที่จัดการแยกกัน
