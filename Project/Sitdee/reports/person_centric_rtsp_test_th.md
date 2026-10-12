# Person-centric RTSP short test

วันที่ 10 กันยายน 2026, profile `HIKVISION_MAIN`, preset `RTSP_CCTV`, duration 8 วินาที, `max_people=4`

## ผลที่วัดได้

| รายการ | ผล |
|---|---:|
| source resolution | 2688 × 1520 |
| source FPS | 43.15 |
| AI inference FPS | 3.39 |
| frame age median / p95 | 16.0 / 193.2 ms |
| inference median / p95 | 243.4 / 370.8 ms |
| frames captured / inferred | 245 / 29 |
| old frames overwritten | 244 |
| decode failures / reconnects | 0 / 0 |
| MediaPipe detection rate | 100% ของเฟรมที่ infer |
| core valid rate | 100% ของ pose ที่ตรวจพบ |
| mean detected people | 2.45 |
| maximum detected people | 3 |
| maximum active tracks | 4 |

## Tracking diagnostics

- anonymous IDs ที่พบ: `PERSON_01..PERSON_04`
- `PERSON_APPEARED`: 6
- `PERSON_ACTIVE`: 62
- `PERSON_TEMPORARILY_LOST`: 7
- `PERSON_REACQUIRED`: 3
- `PERSON_EXITED`: 3
- fragmentation proxy: มี 6 appeared events ในช่วงสั้น ๆ; ยังไม่ใช่ ID-switch accuracy เพราะไม่มี manual identity ground truth

การ reuse ID slot จำกัดการโตของ ID และไม่อ้างว่าเป็นบุคคลเดิมในโลกจริง. การทดสอบนี้จึงสรุปได้เพียง software/runtime behavior ไม่ใช่ tracking accuracy.

## Activity gate

activity state ทั้งหมดที่สังเกตได้เป็น `UNKNOWN` จำนวน 81 subject observations. ไม่มี `SIT_DETECTED` หรือ `STAND_DETECTED` ในช่วงทดสอบนี้ ซึ่งแปลว่า gate ปฏิเสธการตัดสินใจเมื่อหลักฐาน lower-body ไม่พอ/ไม่สอดคล้อง ไม่ใช่หลักฐานว่าไม่มีคนนั่งหรือยืน.

ดังนั้นระหว่าง run นี้ V4 sitting-posture evaluation ถูก pause อย่างถูกต้อง และไม่เกิด sitting warning จากการเดา activity.

## ข้อจำกัด

ยังไม่มี expected-people ground truth ต่อเฟรม, manual track IDs, activity labels หรือ posture labels จึงยังไม่คำนวณ person-count accuracy, IDF1, sit/stand accuracy หรือ posture accuracy.

JSON ดิบของ diagnostic อยู่ที่ `outputs/metrics/cctv_multiperson_benchmark.json` และไม่บันทึก raw video.
