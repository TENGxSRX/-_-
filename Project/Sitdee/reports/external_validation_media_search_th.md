# รายงานค้นหาและตรวจสอบสื่อ RGB ภายนอก

## ผลสรุป

เลือก DriveNet รุ่น 6 จาก Mendeley Data เพราะเป็นวิดีโอ RGB จริง มีชื่อคลาสจากแหล่งข้อมูล มี metadata ของผู้เข้าร่วม/วิดีโอ และระบุใบอนุญาต CC BY 4.0 ชัดเจน ดาวน์โหลดไว้เพียง 3 คลิป รวม 369.80 MiB โดยไม่แก้ไขไฟล์ต้นฉบับ

- `SS` → `TUP` (Sit Straight → Upright): map ได้ตรงความหมาย
- `LB` → `TLB` (Lean Backward → Lean Backward): map ได้ตรงความหมาย
- `MB` (Move Backward): เก็บเป็น `UNLABELED` เพราะไม่ควรเดาความหมายเป็นคลาสของโครงการ
- ไม่ดาวน์โหลด `RP` เพราะไม่จำเป็นต่อ smoke test และความหมายของ “Right Position” ยังไม่ชัดว่าเท่ากับ Lean Right

## ตารางไฟล์

| ไฟล์ | label โครงการ | class ต้นทาง | participant | frames | duration (s) | bytes | SHA-256 |
|---|---|---|---:|---:|---:|---:|---|
| SS_1.mp4 | TUP | SS | 1 | 2131 | 71.04991164605075 | 153589862 | f4161e3bb9faf61831422e0d88c253f9746423a86be1deac8f997212a4426f79 |
| LB_2.mp4 | TLB | LB | 2 | 1058 | 35.27842614204735 | 77562475 | baa2dbb0c1d4dec761389ab4941114e27c61928ac8132c6eae0abcd84ce677ac |
| MB_1.mp4 | UNLABELED | MB | 1 | 2074 | 69.14024735806915 | 156608940 | 51c8b6790f0094672779fed1fa7e3cadeea1911b04e5256ceff74fe6ef40c34f |

ตรวจ hash ตรงกับ metadata ของแหล่งข้อมูล: **True**

## ผลทดสอบ 10 ประเด็น

1. แหล่งข้อมูล: DriveNet v6 official Mendeley Data, `https://data.mendeley.com/datasets/v37nxdn2gh/6`
2. สื่อเป็น RGB และมีบุคคลนั่งจริงในรถ แต่ไม่ใช่ภาพโต๊ะทำงานสำนักงานโดยตรง
3. ฉลาก `SS`, `LB`, `MB` มาจาก class folder และ `meta_data.csv`; ไม่มีการเดาฉลากจากภาพ
4. ใบอนุญาต: CC BY 4.0 ต้องให้เครดิตและลิงก์ใบอนุญาต
5. ไฟล์ต้นฉบับถูกเก็บไว้โดยไม่แก้ไข และตรวจ SHA-256 แล้ว
6. ประมวลผลเพียง 300 เฟรมแรกต่อคลิป (~10 วินาที) เพื่อจำกัดขนาดการทดลอง
7. MediaPipe ตรวจพบ pose 900/900 เฟรม
8. adapter ที่ปลอดภัยรับ landmark ครบและ visibility ≥ 0.5 ได้ 0 เฟรม เนื่องจากขา/เท้าบางส่วนไม่ชัดจากมุมด้านข้าง
9. จึงยังไม่มี accuracy หรือ confusion matrix ของ external RGB ชุดนี้
10. ชุดนี้เพียงพอสำหรับ offline pose smoke test และยืนยัน failure mode แต่ยังไม่พอพิสูจน์ว่า MediaPipe RGB ใช้กับ MultiPosture model ได้จริง; ยังต้องใช้สื่อมุมกล้องที่เหมาะสมหรือวิดีโอที่ผู้ใช้บันทึกเองภายหลัง

## ที่เก็บผลลัพธ์

- manifest: `validation_data/external/external_manifest.csv`
- annotated videos: `outputs/validation/external/videos/`
- telemetry: `outputs/validation/external/metrics/`, `summaries/`, `logs/`
- audit JSON: `outputs/validation/external/external_media_audit.json`

สถานะ webcam: **NOT TESTED — HARDWARE NOT CURRENTLY AVAILABLE**
