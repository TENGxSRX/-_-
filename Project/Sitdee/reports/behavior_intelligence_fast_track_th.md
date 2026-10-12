# Behavior Intelligence Fast Track

## สถานะ

**READY_FOR_BEHAVIOR_DATA_COLLECTION**

ข้อมูลจริงล่าสุดจาก CAM02 พบว่า pose/core ส่วนบนยังใช้งานได้ แต่เข่าเห็นครบประมาณ 1.6% และข้อเท้า 0% ใน smoke ล่าสุด จึงทำให้ geometry Sit/Stand เดิมเป็น `UNKNOWN` เกือบทั้งหมด นี่คือ bottleneck หลักของความเข้าใจพฤติกรรม ไม่ใช่จำนวนคนสูงสุดที่ตรวจพบได้

## Contract ใหม่: `activity_partial_pose_v1`

- เป้าหมาย Activity AI ในอนาคต: `SITTING`, `STANDING`; confidence ต่ำจะเป็น `UNCERTAIN` ที่ runtime และหยุดสะสมความเสี่ยง
- ใช้ nose, ไหล่, hip ที่มีเมื่อมองเห็น, geometry torso ที่ normalize ด้วย shoulder width, bbox ขนาด/สัดส่วน, visibility และ availability mask
- ไม่ใช้ face, PERSON ID, track ID หรือพิกัด x/y ตำแหน่งคงที่ในภาพเป็น feature
- hip/knee/ankle ไม่พร้อมจะถูกเก็บเป็น mask + visibility อย่างชัดเจน ไม่ discard frame เพียงเพราะมองไม่เห็นขา
- เตรียม comparison ใน trainer: `PARTIAL_CCTV_POSE` เทียบกับ `FULLER_AVAILABLE_POSE` ด้วย participant-grouped validation หลังมีข้อมูลจริง

## One-pass pilot collection

ใช้ `scripts/collect_behavior_pilot.py` โดย operator เป็นผู้ให้ ground truth เท่านั้น:

- `D` = `STANDING` และ `posture_label=NONE`
- `1` = `SITTING` และ `posture_label=TUP`
- `2` = `SITTING` และ `posture_label=TLB`
- `SPACE` เริ่ม/หยุดเก็บ, `TAB` เปลี่ยน PERSON เป้าหมาย, `0` ยกเลิก label, `Q` จบ

ไม่มีการบันทึกวิดีโอดิบ และ V4/geometry/tracking/temporal เก็บได้เพียง diagnostic ไม่ใช่ label.

ข้อมูลจะไปที่ `data/behavior_partial_pose_v1/samples.csv` พร้อม session telemetry และ visibility/quality metadata แบบไม่ระบุตัวตน

## Pilot P001–P003

ใช้เวลาคนละไม่เกิน 10–15 นาที; เก็บสามเงื่อนไข `STANDING`, `SITTING+TUP`, `SITTING+TLB` อย่างละช่วงสั้นในตำแหน่ง near/mid/far และตำแหน่งหลังกโต๊ะ/monitor ที่เห็นขาไม่ครบตามจริง หลีกเลี่ยงการเก็บเฟรมซ้ำจำนวนมากจากท่าเดียวกัน

## การ train และการประเมิน

- Activity AI V1 จะไม่ train จนกว่าจะมี human label ที่ตรวจ schema ผ่าน มีทั้งสอง activity class และอย่างน้อย P001–P003 สำหรับ participant-independent grouped validation
- เมื่อพร้อม trainer จะ benchmark Logistic Regression, Random Forest และ HistGradientBoosting โดยห้าม random split ของ frame จาก participant เดียวกัน
- V4 ยังคง frozen TUP/TLB; จะประเมินกับ CCTV ground truth เฉพาะแถว `SITTING+TUP/TLB` หลังมีข้อมูลจริง ก่อนคิด V5
- V5 ถูก block จนกว่าจะมีข้อมูล TUP/TLB จริงและผล V4 baseline บน participant ที่ไม่เคยเห็น

## Integration และ Shadow Mode

เป้าหมาย integration ในอนาคตคือ `PERSON_XX → Activity AI → SITTING → V4/V5 → Temporal`; `STANDING`/`UNCERTAIN` จะ pause temporal risk ของคนนั้นเอง

Shadow Mode ยังคงใช้ค้นหา `ACTIVITY_UNCERTAIN`, landmark หาย, prediction flip, camera distance และ desk-occlusion suspected เพื่อสร้างรายการ `NEEDS_HUMAN_LABEL` เท่านั้น ห้าม pseudo-label.
