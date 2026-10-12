# Sitdee + Roboflow Pose V1

Roboflow ใช้เป็นระบบจัดการภาพ ติดป้าย keypoints และเทรน pose detector แบบ 2D โดยยังไม่แทน MediaPipe ในระบบหลักจนกว่าจะผ่านการทดสอบกับผู้เข้าร่วมที่ไม่อยู่ในชุดฝึก

## เก็บภาพในเครื่อง

```powershell
.\.venv\Scripts\python.exe scripts\collect_roboflow_posture_images.py --participant P001 --session S01 --camera-index 0 --consent
```

กด `1..5` เลือก TUP/TLF/TLB/TLL/TLR และกด `SPACE` เพื่อบันทึกภาพหนึ่งภาพ ควรเว้นช่วงและเปลี่ยนท่าจริง ไม่บันทึกเฟรมติดกันจำนวนมาก

## Roboflow

1. สร้างโปรเจกต์ Keypoint Detection แบบ private
2. ใช้ skeleton ตาม `configs/roboflow_pose.example.yaml`
3. อัปโหลดเฉพาะข้อมูลที่มีความยินยอม
4. แบ่งคน: train=P001-P003, valid=P004, test=P005 (ตัวอย่าง)
5. ห้ามสุ่มเฟรมจากคน/session เดียวกันข้าม split
6. annotate จุดที่เห็นจริง; จุดที่โต๊ะบังห้ามเดา
7. สร้าง Dataset Version และเทรน pose model

หลัง export dataset ให้ตรวจ leakage:

```powershell
.\.venv\Scripts\python.exe scripts\audit_roboflow_split.py C:\path\to\roboflow-export
```

## การเชื่อมระบบ

`src/posture_ai/pose/roboflow_adapter.py` แปลง JSON keypoints จาก Roboflow เป็น canonical landmark names สำหรับ shadow evaluation เท่านั้น ค่า z ถูกตั้งเป็น 0 จึงห้ามส่งเข้าโมเดล V4/V2 เดิมซึ่งฝึกจาก MediaPipe world coordinates ต้องสร้าง classifier 2D รุ่นใหม่และวัดผลเทียบ MediaPipe ก่อน deploy

API key ต้องเก็บใน `.env` และห้าม commit ภาพบุคคล, API key หรือ Hosted API response ที่มีข้อมูลส่วนตัว
