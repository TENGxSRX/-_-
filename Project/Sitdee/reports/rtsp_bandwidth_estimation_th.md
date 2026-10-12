# การประเมิน bandwidth ของกล้อง

เครื่องมือ `scripts/estimate_network_bandwidth.py` ให้ผลเป็น **ESTIMATE ONLY** ต้องใส่ bitrate ที่วัดหรือค่าที่ผู้ดูแลกล้องยืนยันเอง ไม่ได้เดา bitrate ของ Hikvision

สูตรพื้นฐาน:

```text
aggregate Mbps = bitrate ต่อกล้อง × จำนวนกล้อง
storage ต่อชั่วโมง = Mbps × 3600 ÷ 8
```

ตัวอย่างการรันเมื่อมีค่าจริง:

```powershell
& '.\.venv\Scripts\python.exe' scripts\estimate_network_bandwidth.py `
  --bitrate-mbps 4 `
  --cameras 2 `
  --hours 8 `
  --overhead-percent 15
```

ค่า `4 Mbps` และ `15%` ในตัวอย่างเป็นตัวเลขสาธิต ไม่ใช่ค่าของกล้องจริง ต้องแทนด้วยค่าที่ตรวจแล้ว
