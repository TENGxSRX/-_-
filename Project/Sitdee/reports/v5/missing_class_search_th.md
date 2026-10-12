# ผลการค้นหาข้อมูล RGB สำหรับคลาสที่ขาดของ V5

วันที่: 2026-09-05

## ผลตัดสิน

**V5_PUBLIC_DATA_INSUFFICIENT** — ระยะนี้ยังไม่พบ public RGB data ที่มีหลักฐาน
เพียงพอสำหรับ TLF, TLL และ TLR จึงไม่เพิ่มข้อมูลเข้า supervised training,
ไม่สร้าง pseudo-label, ไม่เดาป้ายกำกับจากภาพ และไม่ฝึก V5

ข้อมูลที่ยืนยันได้ในปัจจุบัน:

| คลาส | ตัวอย่าง | กลุ่มอิสระ/ผู้เข้าร่วม | datasets | video/session |
|---|---:|---:|---:|---:|
| TUP | 244 | 8 คน (MMD 5 ID เป็น ID ที่อนุมานจาก protocol) | 2 | 3 video + 5 image sessions |
| TLB | 183 | 3 คน | 1 | 3 video |
| TLF | 0 | 0 | 0 | 0 |
| TLL | 0 | 0 | 0 | 0 |
| TLR | 0 | 0 | 0 | 0 |

## MVSP/LGCSPNet

paper รายงานชุดข้อมูลภาพ RGB หลายมุมมอง 7 posture categories, 7 fixed views
และ 1 dynamic view รวม 49,798 ภาพ แต่ repository ทางการมีเพียง README และลิงก์
ดาวน์โหลด Baidu ไม่มี license file, annotation manifest, participant table หรือ
archive listing ที่ตรวจได้จากเครื่อง

ความหมายที่ตรวจจากแหล่งต้นฉบับ:

- `upright` → COMPATIBLE กับ TUP
- `back` → COMPATIBLE กับ TLB
- `down` = เอนไปหน้าและก้มตัว → **AMBIGUOUS**, ไม่บังคับให้เป็น TLF
- `left` → COMPATIBLE กับ TLL
- `right` → COMPATIBLE กับ TLR
- `cheek` และ `leg` → ไม่ใช่ trunk-direction label

MVSP **ยังไม่ได้รับมา** เพราะเข้าถึง archive และ license ยังยืนยันไม่ได้ หากมี
สิทธิ์เข้าถึงโดยชอบธรรม ให้ดาวน์โหลดด้วยตนเองจาก README ทางการ เก็บ archive เดิม
ไว้ที่ `datasets/raw/rgb_native_v5/mvsp/`, คำนวณ SHA-256 และตรวจ label/participant
ก่อนส่งให้ทำ audit รอบใหม่ ห้าม bypass CAPTCHA หรือ access control

## แหล่งอื่น

CCTV Exam Monitor มีชื่อคลาส Forward/Left/Right แต่ local archive ไม่มี sidecar
ที่เชื่อม label กับภาพ จึงเป็น audit-only ส่วน SitStraightNet มีคำว่า hunched over
แต่ไม่มี per-file label join จึงเป็น smoke-only MMD-MSD และ DriveNet มีประโยชน์ต่อ
TUP/TLB เท่านั้น ข้อมูล pressure sensor ไม่ใช่ RGB จึงใช้แทนไม่ได้

## ผลในเครื่อง

ไม่มีการดาวน์โหลดข้อมูลใหม่ในระยะนี้ manifest ยังคง 452 แถว และ MediaPipe ตรวจ
pose/core feature ได้ 452/452 แถว พื้นที่ว่าง C: ประมาณ 19.97 GB และไม่ได้เริ่ม
ดาวน์โหลด archive ขนาดเกิน 2 GB เพราะ MVSP ยังไม่ผ่าน access/license gate

ถ้าเพิ่ม MVSP เฉพาะ TLF/TLL/TLR จะทำให้ dataset identity ทำนาย posture label ได้
โดยตรง จึงต้องเพิ่ม TUP/TLB จาก source เดียวกัน หรือมี missing classes จากหลาย
แหล่ง ก่อนเริ่ม V5 training

รายละเอียดภาษาอังกฤษอยู่ที่ `reports/v5/missing_class_search.md`
