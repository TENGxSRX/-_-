# CCTV Anonymous Tracking Real-World Evaluation

สถานะปัจจุบัน: **PENDING_MANUAL_TRACK_GROUND_TRUTH**

โค้ด anonymous tracker พร้อมแล้ว ใช้ center/bbox continuity และไม่มี face,
ชื่อ หรือ biometric identity แต่ยังไม่มีคลิปที่ operator ทำเครื่องหมาย ID
ต่อเนื่องเพื่อคำนวณ real-world ID stability

## Metric ที่จะรายงาน

- new track count
- average continuous track duration
- temporarily lost observations
- expired observations
- track fragmentation
- manually observable ID-switch evidence
- การกลับเข้าสู่เฟรมหลังหายชั่วคราว

ต้องทดสอบฉากเดินเข้า/ออก, คนเดินตัดหน้า, คนสองคนไขว้กัน และ partial occlusion
โดยเก็บ expected count และ target metadata แยกจาก model prediction

ค่า IDF1 หรือ switch accuracy จะไม่รายงานจนกว่าจะมี manual track ground truth
รอบ RTSP ก่อนหน้าจึงใช้ได้เพียงวัด lifecycle/throughput ไม่ใช่ tracking accuracy
