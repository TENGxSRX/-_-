# สถานะชุดข้อมูล Local RGB V1

สถานะ: **WAITING_FOR_LOCAL_RGB_DATA**

- ผู้เข้าร่วมที่มีวิดีโอ: **0 / 10**
- sessions ที่มีวิดีโอ: **0 / 20**
- วิดีโอ: **0 / 100**
- usable frames หลัง pre-flight: **0**

| ท่า | วิดีโอที่มี | ผู้เข้าร่วมอิสระ |
|---|---:|---:|
| TUP | 0 / 20 | 0 |
| TLF | 0 / 20 | 0 |
| TLB | 0 / 20 | 0 |
| TLL | 0 / 20 | 0 |
| TLR | 0 / 20 | 0 |

ยังไม่มีการฝึกโมเดล การตัดสินใจนี้เป็น data-readiness gate เท่านั้น ต้องให้ครบ
ทั้ง 5 ท่าและมีหลาย participant ก่อนจึงจะเปิด training ได้ การทดลองแรกในอนาคต
ต้องเป็น LOCAL ONLY และแบ่ง train/validation/test ตาม participant
