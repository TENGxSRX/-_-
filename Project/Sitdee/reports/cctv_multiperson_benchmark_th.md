# CCTV multi-person benchmark

สถานะ: **REAL_MULTIPERSON_SOURCE_RUN_COMPLETED_BUT_NOT_GROUND_TRUTH_VALIDATED**
Profile: `HIKVISION_MAIN` | Preset: `RTSP_CCTV` | ประมาณ 8.0 วินาทีต่อค่า max_people

การทดสอบนี้ใช้ RTSP latest-frame แบบ read-only ไม่บันทึกวิดีโอ ไม่เก็บ label และไม่ train โมเดล

| max_people | Source FPS | AI FPS | Frame age median | Inference median | Captured | Inferred | Dropped old | Max detected | Max core valid | Core OK | Reconnects |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 43.15 | 3.39 | 16.0 ms | 243.4 ms | 245 | 29 | 244 | 3 | 3 | 100.0% | 0 |

## Tracking และ activity diagnostics

### max_people=4
- active tracks: `{'mean': 2.793103448275862, 'max': 4, 'min': 1}`
- anonymous IDs seen: `['PERSON_01', 'PERSON_02', 'PERSON_03', 'PERSON_04']`
- lifecycle events: `{'PERSON_APPEARED': 6, 'PERSON_ACTIVE': 62, 'PERSON_TEMPORARILY_LOST': 7, 'PERSON_REACQUIRED': 3, 'PERSON_EXITED': 3}`
- activity events: `{}`
- activity state distribution: `{'UNKNOWN': 81}`
- tracking state distribution: `{'NEW': 6, 'ACTIVE': 65, 'TEMPORARILY_LOST': 10}`
- fragmentation proxy: `{'appeared_events': 6, 'reacquired_events': 3, 'temporary_lost_events': 7, 'exited_events': 3, 'id_switch_accuracy': 'NOT_MEASURED_NO_MANUAL_PERSON_GROUND_TRUTH'}`


## ข้อจำกัด

ยังไม่มี manual ground truth สำหรับ expected_people_count จึงไม่คำนวณ person-count accuracy, missed-person rate หรือ IDF1. หากไม่มีคนนั่งในภาพ สถานะจะเป็น `REAL_MULTIPERSON_DATA_NOT_AVAILABLE`.
การเพิ่ม `max_people` เป็นการตั้งเพดานของ MediaPipe ไม่ใช่หลักฐานว่ากล้องตรวจพบคนจำนวนดังกล่าวจริง.

JSON: `outputs\metrics\cctv_multiperson_benchmark.json`
