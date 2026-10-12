# CCTV Frozen V4 Real-World Evaluation

สถานะปัจจุบัน: **PENDING_HUMAN_LABELLED_CCTV_SAMPLES**

ยังไม่มีการรายงาน accuracy ของ V4 บน CCTV เพราะยังไม่มี sample ที่มี human
posture label และไม่มีการเปิดใช้ prediction เป็น ground truth

## เงื่อนไขการประเมิน

นับเฉพาะ sample ที่:

1. operator ให้ `human_posture_label` อย่างชัดเจน
2. target `PERSON_XX` ถูกเลือกโดย operator
3. required core landmarks และ V4 features ผ่าน gate
4. label อยู่ใน `TUP/TLB` สำหรับ frozen V4

จะรายงาน accuracy, balanced accuracy, macro-F1, precision/recall/F1 ต่อคลาส
และ confusion matrix โดยแยก participant, position, distance และจำนวนคนในเฟรม

คนที่ตรวจไม่พบหรือ core ไม่ผ่านจะถูกนับใน detection/core report เท่านั้น
ไม่ถูกนำไปนับเป็น classifier error

## การป้องกัน leakage

ใช้ participant-independent grouping และห้ามสุ่ม frame จาก clip เดียวกันข้าม
train/test การประเมินจริงยังไม่เริ่มจนกว่าจะมีข้อมูลหลาย participant และหลาย
CCTV position
