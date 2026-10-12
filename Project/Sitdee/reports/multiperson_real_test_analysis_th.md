# วิเคราะห์ Real Multi-Person Log

วันที่วิเคราะห์: 8 กันยายน 2026  
ไฟล์ต้นฉบับ: `outputs/multiperson_test.jsonl`  
ขนาดไฟล์จริง: `2,757,902 bytes`  
สถานะ: หลักฐานจาก telemetry จริงของ runtime เดิม ไม่ใช่ ground-truth identity

## สรุปผลที่ตรวจได้

| รายการ | ผลจริง |
|---|---:|
| จำนวน telemetry frames/lines | 300 |
| JSON lines ที่อ่านไม่ได้ | 0 |
| inference frames | 100 |
| skipped frames | 200 |
| ช่วง timestamp ใน log | 1.4305–83.3531 วินาที |
| unique `PERSON_XX` IDs | 21 (`PERSON_01` ถึง `PERSON_21`) |
| จำนวน detection พร้อมกันสูงสุด | 3 |
| การกระจาย subject count | 1 คน: 57 frames, 2 คน: 240 frames, 3 คน: 3 frames |
| รวม subject records ใน telemetry | 546 |
| first-appearance/new-ID events | 21 |
| additional IDs หลัง ID แรก | 20 |
| proxy ID-creation rate | 21 / 300 = 7.0% ต่อ telemetry line |
| IDs ที่มี `TEMPORARILY_LOST` | 7 IDs |
| telemetry gap ที่ subject หายจากรายการแล้วกลับมา | 0 ตามนิยามนี้ |

ID-creation rate และอัตรา `21 unique IDs / 3 max simultaneous detections = 7.0` เป็นเพียง fragmentation proxies ไม่ใช่ IDF1, identity accuracy หรือการยืนยันว่าคนจริงเปลี่ยนคน เพราะ log ไม่มี ground-truth identity

## สถานะ tracking ที่พบ

จาก subject records ทั้งหมด 546 รายการ:

| `tracking_state` | จำนวน |
|---|---:|
| `NEW` | 63 |
| `ACTIVE` | 462 |
| `TEMPORARILY_LOST` | 21 |

มี 7 IDs ที่เคยเข้า `TEMPORARILY_LOST`: `PERSON_01`, `PERSON_08`, `PERSON_09`, `PERSON_10`, `PERSON_13`, `PERSON_14`, `PERSON_15`

`TEMPORARILY_LOST` ไม่เท่ากับ seat ว่าง เพราะ runtime เดิมยังคง track ไว้ในผลลัพธ์ช่วงสั้น ๆ ตาม lifecycle rule

## Track lifetime ที่สังเกตจาก telemetry

นิยามในตารางนี้คือ `last telemetry frame - first telemetry frame` ของ ID เดียวกัน จึงเป็น telemetry span ไม่ใช่เวลาที่พิสูจน์ได้ว่าบุคคลจริงอยู่ในที่นั่ง

| ID | first frame | last frame | records | span (s) | states ที่พบ |
|---|---:|---:|---:|---:|---|
| PERSON_01 | 0 | 14 | 15 | 2.318 | NEW, ACTIVE, TEMPORARILY_LOST |
| PERSON_02 | 12 | 56 | 45 | 12.925 | NEW, ACTIVE |
| PERSON_03 | 12 | 56 | 45 | 12.925 | NEW, ACTIVE |
| PERSON_04 | 57 | 71 | 15 | 3.614 | NEW, ACTIVE |
| PERSON_05 | 57 | 71 | 15 | 3.614 | NEW, ACTIVE |
| PERSON_06 | 72 | 149 | 78 | 22.686 | NEW, ACTIVE |
| PERSON_07 | 72 | 149 | 78 | 22.686 | NEW, ACTIVE |
| PERSON_08 | 150 | 158 | 9 | 1.799 | NEW, ACTIVE, TEMPORARILY_LOST |
| PERSON_09 | 150 | 191 | 42 | 11.296 | NEW, ACTIVE, TEMPORARILY_LOST |
| PERSON_10 | 159 | 200 | 42 | 11.325 | NEW, ACTIVE, TEMPORARILY_LOST |
| PERSON_11 | 192 | 209 | 18 | 3.374 | NEW, ACTIVE |
| PERSON_12 | 207 | 209 | 3 | 0.000 | NEW |
| PERSON_13 | 210 | 227 | 18 | 4.608 | NEW, ACTIVE, TEMPORARILY_LOST |
| PERSON_14 | 210 | 233 | 24 | 6.016 | NEW, ACTIVE, TEMPORARILY_LOST |
| PERSON_15 | 228 | 248 | 21 | 3.660 | NEW, ACTIVE, TEMPORARILY_LOST |
| PERSON_16 | 246 | 278 | 33 | 5.333 | NEW, ACTIVE |
| PERSON_17 | 276 | 278 | 3 | 0.000 | NEW |
| PERSON_18 | 279 | 296 | 18 | 4.572 | NEW, ACTIVE |
| PERSON_19 | 279 | 296 | 18 | 4.572 | NEW, ACTIVE |
| PERSON_20 | 297 | 299 | 3 | 0.000 | NEW |
| PERSON_21 | 297 | 299 | 3 | 0.000 | NEW |

## Detector-order changes

ไม่สามารถวัด detector output order ดิบจาก log นี้ได้ เพราะ schema ปัจจุบันบันทึก `subjects` หลังจากเรียงตาม `PERSON_XX` และไม่ได้เก็บ `detector_index` หรือ raw pose order

ตรวจพบว่า sequence ของ subject IDs ที่มองเห็นใน telemetry เปลี่ยน 17 ครั้ง แต่ตัวเลขนี้เป็นการเปลี่ยน roster/สมาชิกที่มองเห็น ไม่ใช่หลักฐานว่า detector คืน `[A,B]` แล้วสลับเป็น `[B,A]`

## Temporary disappearance และ ID creation

- ไม่พบช่วงที่ subject ถูกถอดออกจากรายการแล้วกลับมา โดยใช้การเปรียบเทียบ telemetry lines ติดกันเป็นนิยามของ gap
- พบ 21 first-appearance events และ ID ต่อเนื่องสูงถึง `PERSON_21`
- ใน log เดียวกันมีช่วง 3 detections พร้อมกันเพียง 3 frames แต่เกิด IDs ใหม่หลายชุด แสดงหลักฐานเชิง runtime ว่า generic association มี fragmentation เมื่อ scene เปลี่ยน/คนเข้าออก
- ไม่สามารถสรุปได้ว่าแต่ละ ID ใหม่คือคนใหม่จริงหรือ track เดิมแตก เพราะไม่มี identity ground truth และไม่มี raw detector order

## เปรียบเทียบกับสาเหตุที่เลือก Seat-aware V2

หลักฐานจริงสนับสนุนการเปลี่ยน persistent context จาก `PERSON_XX` เป็น `SEAT_XX` ในห้องทำงานที่ที่นั่งคงที่:

1. `PERSON_XX` เพิ่มจาก 1 ถึง 21 ใน run เดียว
2. tracker เดิมมี temporary-lost records แต่ไม่สามารถพิสูจน์ continuity ของคนจริงได้
3. มีมากกว่า 1 คนพร้อมกัน และสูงสุด 3 detections จึงไม่ควรใช้ detector order เป็น identity
4. telemetry เดิมไม่มี seat/ROI/anchor ทำให้ไม่สามารถแยก seated occupant จาก passerby ได้

## ข้อจำกัดของหลักฐาน

- log นี้มี telemetry ที่ถูกเขียนจาก runtime ก่อน seat-aware mode
- ไม่มีภาพต้นฉบับในรายงานนี้ จึงตรวจตำแหน่ง ROI หรือ passerby ด้วยสายตาไม่ได้
- ไม่มี raw detector index/order, face data หรือ ground-truth identity
- ค่า `v4_prediction` และ geometry เป็น output ของระบบเดิม ไม่ใช่ clinical validation
- จึงใช้ผลนี้เพื่อออกแบบ state/assignment และรายงาน fragmentation เท่านั้น ไม่ใช่ accuracy claim
