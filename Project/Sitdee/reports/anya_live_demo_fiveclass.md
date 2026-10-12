# ANYA Live Webcam Five-Class Demo

สถานะ: **พร้อมสำหรับการทดสอบแบบมีผู้ใช้หน้ากล้อง แต่ยังไม่ถือว่า live-validated**

## การเชื่อมต่อ

คำสั่งเดิมยังใช้โมเดล Upper-body V2 ตามค่าเริ่มต้น ส่วนโมเดล P001 five-class ต้องเลือกด้วย `--posture-model webcam-p001-fiveclass-exp-v1` เท่านั้น

เส้นทางทำงานของโหมดทดลอง:

`Webcam/File/RTSP -> YOLO -> ByteTrack -> selected track -> CandidateSelector -> padded-crop safety -> MediaPipe Pose V2 -> six NO_HIP features -> webcam_p001_fiveclass_exp_v1 -> GUI`

เฉพาะ track ที่ผู้ใช้คลิกเลือกเท่านั้นที่ส่งเข้า pose/model ในโหมดทดลอง ไม่มีการสลับ target อัตโนมัติ และ crop ที่มีบุคคลอื่นปนยังถูกปฏิเสธก่อน MediaPipe

## หน้าจอ

- กล่องเหลือง: พบคนแต่ยังไม่ได้เลือก
- กล่องเขียว: target ที่เลือก พร้อม `SELECTED: TRACK <id>`
- แสดงสถานะ pose แยกจากผล posture
- วาด upper-body skeleton เฉพาะ selected target เมื่อ pose สดและ valid
- แสดงชื่อโมเดลและ `EXPERIMENTAL`
- แสดง TUP/TLF/TLB/TLL/TLR พร้อมคำไทย หรือ `UNKNOWN — ยังไม่มั่นใจ`
- แสดง FPS และ processing time โดยประมาณ
- เมื่อ target หาย แสดง `TARGET LOST` และไม่ย้าย prediction ไปยัง track ใหม่

## ข้อจำกัด

โมเดลนี้ฝึกจาก P001 เพียงคนเดียว สอง sessions และยังไม่ผ่าน live webcam evaluation หรือ independent-person evaluation ผล offline เดิมห้ามใช้เป็น live accuracy ไม่มีการบันทึก raw image/video และไม่มี prediction ใดถูกใช้เป็น Human GT

## คำสั่งทดสอบ

```powershell
.\.venv_yolo\Scripts\python.exe .\scripts\run_upper_body_posture_v2.py --source webcam --camera-index 0 --camera-width 1280 --camera-height 720 --camera-fps 30 --max-seconds 120 --posture-model webcam-p001-fiveclass-exp-v1
```

คลิกในกล่องของผู้สาธิต รอ `SELECTED: TRACK <id>` และ skeleton จากนั้นทำ TUP, TLF, TLB, TLL, TLR ทีละท่า ผล `UNKNOWN` ถือเป็นการปฏิเสธอย่างถูกต้อง ไม่ควรฝืนเปลี่ยน threshold
