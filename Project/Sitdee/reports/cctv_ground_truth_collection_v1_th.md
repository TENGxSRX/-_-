# Controlled CCTV Ground Truth Collection V1

สถานะ: **READY_FOR_ATTENDED_COLLECTION** — เครื่องมือพร้อม แต่ยังไม่มี
ข้อมูล human ground truth ใน `data/controlled_cctv_v2/`

## กติกา

- ใช้กล้อง CCTV จริงและมี operator อยู่หน้าจอตลอดเวลา
- ใช้ participant code แบบไม่ระบุตัวตน เช่น `P001`, `P002`, `P003`
- ใช้ runtime track ID `PERSON_XX` เพื่อเลือกเป้าหมายชั่วคราวเท่านั้น
- ห้ามใช้ชื่อ บุคคลจริง ใบหน้า หรือ biometric identity
- label ต้องมาจากคน; V4 prediction, Geometry และ Temporal เป็น diagnostic เท่านั้น
- ค่าเริ่มต้นไม่บันทึกวิดีโอดิบ

## ขั้นตอนต่อระยะ

### Stage A — คนเดียว

ใช้ P001, P002, P003 สลับตำแหน่ง LEFT/CENTER/RIGHT และ NEAR/MID/FAR เมื่อทำได้
แต่ละคนเก็บ `TUP` และ `TLB` หลายคลิปสั้น โดยกำหนด `EXPECTED PEOPLE=1`

### Stage B — สองคน

กำหนด `EXPECTED PEOPLE=2` และเก็บคู่ TUP/TLB ให้ครบเท่าที่ทำได้ เช่น
TUP+TUP, TUP+TLB, TLB+TUP, TLB+TLB

### Stage C — สามคน

กำหนด `EXPECTED PEOPLE=3` ซึ่งเป็น product target หลัก ทดสอบตำแหน่งแนวนอน
และระยะต่างกัน เช่น TUP/TUP/TUP, TUP/TLB/TUP, TLB/TUP/TLB

### Stage D — สี่คน

กำหนด `EXPECTED PEOPLE=4` เป็น stress test วัดการตรวจพบและ core quality ก่อน
ไม่จำเป็นต้องสร้าง posture sample หากการติดป้ายไม่ชัด

## วิธีใช้หน้าจอ

- `[` / `]`: ลด/เพิ่ม EXPECTED PEOPLE 1–4
- `TAB` หรือ `N`: เลือก target `PERSON_XX`
- `1..5`: เลือก human label `TUP/TLB/TLF/TLL/TLR`
- `SPACE`: เริ่ม/หยุดเก็บ accepted samples
- `0`: `IGNORE` และหยุดการเก็บ target
- `U`: `UNCERTAIN` และหยุดการเก็บ target
- `R`: reset label
- `Q/ESC`: จบ session

ผู้ที่มองเห็นแต่ไม่ใช่ target จะยังอยู่ใน telemetry เพื่อนับ detection/tracking
แต่ถูกบันทึกสถานะ `IGNORE_FOR_POSTURE_TRAINING` และไม่สร้าง posture sample

## คำสั่งตัวอย่าง

```powershell
# Stage A
& '.\.venv\Scripts\python.exe' scripts\collect_controlled_cctv.py --source rtsp --profile HIKVISION_MAIN --max-people 1 --expected-people-count 1 --preset RTSP_CCTV --subject-id P001 --clip-id CCTV_STAGEA_P001_LEFT_NEAR --target-position-id LEFT_NEAR --distance-band NEAR

# Stage B
& '.\.venv\Scripts\python.exe' scripts\collect_controlled_cctv.py --source rtsp --profile HIKVISION_MAIN --max-people 2 --expected-people-count 2 --preset RTSP_CCTV --subject-id P001 --clip-id CCTV_STAGEB_P001_P002_01

# Stage C
& '.\.venv\Scripts\python.exe' scripts\collect_controlled_cctv.py --source rtsp --profile HIKVISION_MAIN --max-people 3 --expected-people-count 3 --preset RTSP_CCTV --subject-id P001 --clip-id CCTV_STAGEC_P001_P002_P003_01

# Stage D
& '.\.venv\Scripts\python.exe' scripts\collect_controlled_cctv.py --source rtsp --profile HIKVISION_MAIN --max-people 4 --expected-people-count 4 --preset RTSP_CCTV --subject-id P001 --clip-id CCTV_STAGED_01
```

เก็บ metadata, sample CSV, rejection reasons, collection events และ telemetry
ไว้ใต้ `data/controlled_cctv_v2/`; ไม่ train V5 ในขั้นตอนนี้
