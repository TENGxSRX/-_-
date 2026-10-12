# แทร็กฝึก RGB-native V4

## สถานะ

**RGB-NATIVE V4: ฝึกเป็น PILOT แล้ว — รองรับเพียง 2 คลาสที่ mapping ได้**

V4 ใช้เส้นทางเดียวกันทั้งตอนฝึกและตอนอนุมานออฟไลน์: `วิดีโอ RGB → MediaPipe Pose World Landmarks → ย้ายจุดอ้างอิงไปกึ่งกลางสะโพก → feature V4 → classifier` และแยกจากโมเดล MultiPosture V1/V2/V3

## ข้อมูลและป้ายกำกับ

- แหล่งทางการ: DriveNet v6 (https://data.mendeley.com/datasets/v37nxdn2gh/6) ใบอนุญาต CC BY 4.0
- วิดีโอที่ใช้และ mapping ได้: 6 คลิป; สุ่มแบบกำหนดแน่นอน 300 เฟรมต่อคลิป
- แถวที่ผ่าน core landmark และใช้ฝึก: 1800
- `SS → TUP` และ `LB → TLB` เป็น COMPATIBLE
- `MB` และ `RP` ยัง AMBIGUOUS จึงไม่ใช้ฝึก และไม่มีการเดาป้ายกำกับ
- แบ่งตามผู้เข้าร่วมล่วงหน้า: คนที่ 1/2 เป็น train และคนที่ 3 เป็น holdout; ไม่มีผู้เข้าร่วมซ้ำข้ามชุด

## ผลโมเดล

- โมเดลที่เลือก: `LogisticRegression` กับ feature `normalized_world` จำนวน 15
- Grouped-CV macro F1: 1.0000 ± 0.0000
- Holdout ผู้เข้าร่วมคนที่ 3 macro F1: 1.0000; balanced accuracy 1.0000
- ผลนี้เป็น pilot สองคลาสในโดเมนวิดีโอขับรถ ไม่ใช่ผล generalization หกคลาสในสำนักงาน

## ข้อจำกัดสำคัญ

- MultiPosture ยังคงเป็นโมเดลทิศทาง 5 คลาสและไม่ได้ถูกแก้ไข
- V4 ยังไม่ครอบคลุม TLF, TLL, TLR หรือ SLOUCH
- frozen re-test เป็นเพียง overlap diagnostic จากแหล่ง DriveNet เดียวกัน ไม่ใช่ external test อิสระ
- ไม่เปิด webcam, ไม่ฝึก ANYA, ไม่ติดตั้ง YOLO และไม่แก้ข้อมูลดิบ
