# รายการไฟล์สำหรับ Acer แบบเบา

ตรวจเมื่อ 2026-09-07 จาก Desktop ซึ่งเป็น source of truth:
`C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets`

## ขนาดที่ตรวจได้

| ส่วน | ขนาด |
|---|---:|
| โฟลเดอร์โครงการทั้งหมด รวม `.venv` | 14.43 GiB |
| `datasets/` | 13.35 GiB |
| `validation_data/` | 0.36 GiB |
| `outputs/` | 0.12 GiB |
| `.venv/` | 0.59 GiB |
| `src/` | 0.50 MiB |
| `scripts/` | 0.89 MiB |
| `tests/` | 0.28 MiB |
| `reports/` | 0.42 MiB |
| `metadata/` | 29.26 MiB |
| ชุดไฟล์ที่จะใส่ Acer ก่อนบีบอัด ZIP | 29.93 MiB |

ZIP code-only ที่สร้างจริงมีขนาด 27.44 MiB

ชุด Acer ต่ำกว่าเป้าหมาย 5 GiB มาก ไฟล์ runtime ที่ใหญ่ที่สุดคือ pose model
ขนาดประมาณ 29.24 MiB

## REQUIRED_ON_LAPTOP — ต้องมีบน Acer

- `src/`
- `scripts/`
- `configs/posture_engine.yaml`
- `configs/laptop_low_spec.yaml`
- `tests/`
- `README.md`, `pytest.ini`, `.gitignore`
- `requirements-laptop.txt`
- `setup_laptop_dev.ps1`, `start_laptop_dev.bat`
- `metadata/pose_landmarker_heavy.task`
- `outputs/models/rgb_native_v4/` ทั้ง 4 ไฟล์
- `fixtures/` สำหรับพัฒนา UI และทดสอบ summary
- รายงาน workflow สำหรับ Acer

V4 โหลดได้โดยไม่ต้องมี raw video

## OPTIONAL — มีได้แต่ไม่จำเป็น

- รายงาน Markdown ที่ต้องการอ่านประกอบ
- ไฟล์ provenance เล็ก ๆ ใน `metadata/`
- saved session summary หากกำลังทำหน้าจอ UI
- media ตัวอย่างขนาดเล็กในอนาคต เมื่อยืนยัน license และสิทธิ์แจกจ่ายแล้ว

รอบนี้ไม่ได้คัดลอกวิดีโอใด ๆ ไปที่ Acer

## KEEP_DESKTOP_ONLY — เก็บเฉพาะ Desktop

- `datasets/raw/` ทุกชุด รวม ANYA, MultiPosture CSV, MVSP, SitPose,
  SitStraightNet, local recordings และ DriveNet
- `datasets/processed/` และ `datasets/incoming/`
- `validation_data/`
- `outputs/validation/`, log/video/session report ของ live,
  `outputs/metrics/` และผล benchmark/training ขนาดใหญ่
- โมเดลฝึก `multiposture_v1`, `multiposture_upperbody_v2` และโมเดลอนาคต
- `.venv/`, cache, `__pycache__/` และโฟลเดอร์ชั่วคราว

นี่เป็นกติกาแยกเครื่อง ไม่ใช่การลบไฟล์ Desktop

## กติกาการ sync

ให้ sync เฉพาะ source code, config, tests, เอกสาร, V4 runtime artifact และ
pose model ห้ามใช้ `git add .` จนกว่าจะตรวจรายการ staged แล้ว ปัจจุบันโฟลเดอร์นี้
ยังไม่มี Git repository หรือ remote จึงยังไม่มีการ push ใด ๆ
