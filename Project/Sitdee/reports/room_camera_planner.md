# Room Camera Planner V1

สถานะ: **พร้อมรับข้อมูลห้องจริง แต่ยังไม่มีข้อสรุปจำนวนกล้อง**

เครื่องมือ `scripts/room_camera_planner.py` รับ JSON หรือ YAML ที่มี `room`, `cameras`, `targets` และ `obstacles` แล้วคำนวณผลประเมินเชิงเรขาคณิตสำหรับทุกคู่ camera-target

## สิ่งที่คำนวณ

- ระยะ camera → target และทิศทางมุมมอง
- ความกว้างแนวนอนและความสูงแนวตั้งโดยประมาณจาก FOV
- อยู่ใน horizontal FOV หรือไม่
- อยู่ในระยะใช้งานที่กำหนดหรือไม่
- เส้นมุมมองตัด obstacle หรือไม่
- ประมาณจำนวน pixel ตามความกว้างตัวบุคคล ถ้ารู้ resolution และ `person_width_m`
- คะแนน candidate เพื่อจัดลำดับ ไม่ใช่คำตัดสินอัตโนมัติ
- target ที่ยังไม่มี camera แบบไม่บังและอยู่ใน FOV

## วิธีใช้เมื่อมีแผนผังจริง

```powershell
& '.\.venv\Scripts\python.exe' scripts\room_camera_planner.py `
  --spec path\to\real_room.json `
  --output-json outputs\room_plan\real_room_plan.json `
  --output-svg outputs\room_plan\real_room_plan.svg
```

ยังไม่ใส่ขนาดห้องจริงใน repository เครื่องมือจะ reject ข้อมูลที่ไม่ครบแทนการเติมค่าที่เดาเอง

## ความหมายของผลลัพธ์

ทุกผลลัพธ์ติดป้าย `GEOMETRIC PLANNING ESTIMATE` เพราะยังไม่รวมเลนส์จริง ความสูงของศีรษะ การบังโดยจอ/คน แสง มุมกล้องจริง และความสามารถของ MediaPipe จึงต้องตรวจด้วยภาพทดลองภายหลัง

คะแนนให้ความสำคัญกับการอยู่ใน FOV, ระยะ, ไม่มี obstacle, มุมมองด้านหน้า และ resolution โดยประมาณ กล้องที่คะแนนสูงไม่ได้แปลว่าผ่านการติดตั้งจริง

## จำนวนกล้อง

เครื่องมือแสดง `greedy_camera_set_estimate` เพื่อช่วยดู candidate ขั้นต้น แต่ `camera_count_decision` จะเป็น `PENDING_REAL_FLOOR_PLAN_REVIEW` จนกว่าจะเห็นห้องจริงและทดสอบ visibility ของ nose/shoulders/hips
