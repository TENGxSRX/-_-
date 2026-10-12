# CCTV Detection Evaluation

สถานะปัจจุบัน: **PENDING_REAL_GROUND_TRUTH**

ผล RTSP short benchmark เดิมเป็นเพียง engineering telemetry และไม่มีค่า
`EXPECTED PEOPLE` จาก operator จึงยังคำนวณ recall ของ 1/2/3/4 คนไม่ได้

## Metric ที่จะคำนวณ

ต่อหนึ่ง frame จะเก็บ:

- `expected_people_count`
- `detected_people_count`
- `core_valid_people_count`

จากนั้นรายงานแยกตาม expected count:

- `EXACT_COUNT_RATE`: detected เท่ากับ expected
- `UNDERCOUNT_RATE`: detected น้อยกว่า expected
- `OVERCOUNT_RATE`: detected มากกว่า expected
- `MEAN_DETECTED_PEOPLE`
- `MAX_DETECTED_PEOPLE`
- `ALL_PEOPLE_CORE_VALID_RATE`: จำนวน detected และ core-valid ครบตาม expected

## ข้อห้ามตีความ

ถ้า expected=3 แต่ detected=2 จะเป็น detection undercount ไม่ใช่ V4 error
ถ้า detected=3 แต่ core-valid=2 จะเป็น core-quality failure ไม่ใช่ classifier error

เครื่องมือ `scripts/audit_cctv_training_readiness.py` จะอ่าน collection events
และสร้าง metric เมื่อมีข้อมูลจริง หากไม่มีข้อมูลจะรายงาน `NO_GROUND_TRUTH_FRAMES`
โดยไม่เติมค่าแทน
