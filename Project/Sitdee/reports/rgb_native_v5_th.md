# สถานะ RGB-native V5

## ผลสรุป

**ยังไม่ฝึกโมเดล V5 แบบหลายคลาส** เพราะข้อมูลที่ตรวจสอบได้มีเพียง TUP และ
TLB; ยังขาด TLF, TLL และ TLR จึงไม่สร้างป้ายกำกับสมมติและไม่ฝึกโมเดลที่อาจ
ทำให้เข้าใจผิด

## ข้อเท็จจริงจากแหล่งข้อมูล

ตรวจสอบ DriveNet v6 และชุดสาธารณะ MMD-MSD ตามลิงก์ทางการที่ระบุในรายงาน
การจับคู่ป้ายกำกับใช้เฉพาะชื่อ/ความหมายที่แหล่งข้อมูลรองรับ ไม่ใช้ pseudo-label
และไม่นำ MVSP/LGCSPNet มาใช้ เพราะใบอนุญาตและไฟล์ดาวน์โหลดยังตรวจสอบไม่ครบ

## ค่าที่ตรวจได้จากไฟล์ในเครื่อง

- Manifest ทั้งหมด 452 แถว: DriveNet 417 แถว และ MMD-MSD 35 แถว
- ป้ายกำกับที่ใช้ได้: TUP 244 แถว, TLB 183 แถว; ไม่มีป้ายกำกับ 25 แถวเก็บไว้
  สำหรับ audit
- ทุกแถวตรวจพบ core pose และสร้างฟีเจอร์ได้ครบ โดยใช้ 5 landmarks:
  nose, ไหล่ซ้าย/ขวา, สะโพกซ้าย/ขวา
- ไฟล์ต้นฉบับที่เก็บ hash 44 ไฟล์ รวม 1,146,521,497 bytes
- ใช้การสุ่มเฟรมวิดีโอแบบคงที่ 1 fps; ไม่สุ่มแยกเฟรมติดกันเพื่อหลีกเลี่ยง leakage

## ผลการทดลองและการตัดสินใจ

สร้าง manifest, ฟีเจอร์ 15/15/30 คอลัมน์, sampling audit และ source hash แล้ว
แต่ยังไม่ทำ grouped CV, holdout, leave-one-dataset-out หรือ error analysis ของ
classifier เพราะคลาสที่จำเป็นยังไม่ครบ นี่คือ data-readiness gate ไม่ใช่ผลว่า
โมเดลล้มเหลว

คลาสที่ต้องเพิ่มต่อไป: TLF (เอนไปข้างหน้า), TLL (เอนไปซ้าย), TLR (เอนไปขวา)
โดยต้องมีป้ายกำกับจากแหล่งทางการ และแบ่งตาม participant/session/video ห้ามสุ่ม
แบ่งรายเฟรม

## สถานะโครงการ

- **V4:** คงเดิมและใช้เป็น baseline
- **V5:** ยังไม่ฝึก — ข้อมูลไม่พอสำหรับ 5 คลาส
- **Offline RGB/MediaPipe extraction:** พร้อมในระดับ manifest/feature extraction
- **Webcam:** ไม่ได้เปิดใช้งาน
- **ANYA:** ไม่ได้ฝึก
- **YOLO:** ไม่ได้ติดตั้ง

ไฟล์สำคัญ: `datasets/processed/rgb_native_v5/manifest.csv`,
`datasets/processed/rgb_native_v5/features.npz`,
`reports/v5/rgb_v5_label_mapping.md`,
`reports/v5/dataset_class_confounds.md`,
`outputs/models/rgb_native_v5/NOT_TRAINED.md`
