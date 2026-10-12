# สรุปการเตรียม Desktop สำหรับ RTSP AI Server

## 1. สถานะ RTSP readiness

**RTSP SOFTWARE PREPARED — PHYSICAL CAMERA NOT TESTED**

Desktop มี `VideoSource` กลางสำหรับ USB, file และ RTSP พร้อม source health, reconnect แบบจำกัด และ stale-source protection แล้ว แต่ยังไม่มีผลยืนยันจาก Hikvision จริง

## 2. สิ่งที่ทำแล้ว

- source abstraction เดียวสำหรับ USB/file/RTSP
- Hikvision main/sub profile template: `101` และ `102`
- redacted URL diagnostics และ `.env.example`
- `SOURCE_CONNECTING`, `SOURCE_READY`, `SOURCE_STALE`, `SOURCE_RECONNECTING`, `SOURCE_FAILED`, `SOURCE_CLOSED`
- capped reconnect/backoff และ latest-frame buffer แบบหนึ่งเฟรม
- RTSP_DESKTOP / RTSP_BALANCED และ dual-resolution inference copy
- dashboard แยก SOURCE HEALTH จาก AI POSTURE/geometry
- diagnostic แบบไม่บันทึกภาพและ bandwidth estimator

## 3. สิ่งที่ยังไม่ได้ทดสอบ

- ไม่ได้เชื่อม physical Hikvision
- ไม่ได้เปิด RTSP network connection
- ไม่ได้ยืนยัน IP, port, username, password, codec, bitrate หรือ resolution จริง
- ไม่ได้วัด FPS/latency/reconnect จาก network จริง
- ไม่ได้ทดสอบ multi-person จากกล้องจริง

## 4. การเชื่อมต่อภายหลัง

```text
Hikvision → PoE switch → camera network → Desktop RTSP/OpenCV
                                      หรือ
Hikvision → Acer gateway/relay → private path → Desktop
```

ยังไม่ตัดสินว่าจะใช้ direct route หรือ Acer gateway จนเห็น topology และ policy จริง

## 5. Direct route กับ Acer gateway

- Direct route: latency และจุดเสียหายน้อยกว่า แต่ Desktop ต้อง route ถึง camera network
- Acer gateway/relay: แยก network ได้ แต่เพิ่ม latency, bandwidth dependency และงานดูแล
- VPN/private link: ใช้ได้เมื่อ network แยกกัน แต่ยังไม่เลือก vendor และไม่ติดตั้งคืนนี้

## 6. Secrets/config ที่ต้องมีในเครื่องจริง

ใช้ `.env` local หรือ environment variables:

- `POSTURE_CAMERA_HOST`
- `POSTURE_CAMERA_PORT`
- `POSTURE_CAMERA_USER`
- `POSTURE_CAMERA_PASSWORD`
- `POSTURE_HIKVISION_PROFILE`
- หรือ `POSTURE_RTSP_URL`

ไม่ใส่ค่าเหล่านี้ใน source, batch, telemetry, report หรือแชต

## 7. Reconnect behavior

เมื่อ read ล้มเหลว ระบบเพิ่ม failure counter, เข้า `SOURCE_RECONNECTING`, รอ backoff ที่มีเพดาน และลองจำนวนครั้งจำกัด หากไม่สำเร็จจะเป็น `SOURCE_FAILED`/`SOURCE_STALE` ไม่ทำ busy-loop และไม่เปลี่ยนเป็น posture warning

## 8. Stream health

รายงาน connected state, frames received, decode failures, consecutive failures, reported/effective FPS, resolution, last-frame age, reconnect count, late-frame count และระบุชัดว่า packet loss ไม่สามารถวัดจาก OpenCV ได้โดยตรง

## 9. Tests

มี mock tests สำหรับ source parsing, credential redaction, open/read failure, reconnect, stale buffer, shutdown, bandwidth, resizing และ backward-compatible dashboard fields ชุดทดสอบเต็มผ่าน `109 passed`

## 10. Integrity

V4 ยังคง frozen สองคลาส, ไม่ train V5/ANYA, ไม่ติดตั้ง YOLO และไม่แก้ raw datasets หรือ model artifacts โมเดลเดิมยังแยกจาก source runtime ใหม่

ตรวจ hash ซ้ำแล้ว: ANYA raw 4 ไฟล์, MultiPosture raw และ model V1/V2/V4 ตรงกับ
manifest เดิมทั้งหมด C: เหลือประมาณ 21.28 GB หลังงานนี้ และ bundle code-only รุ่นใหม่
`outputs/laptop_dev_bundle_v4.zip` มี 153 entries, SHA-256
`266bab5fd2aac64631cd93b8c6f3912e422a93cf40a99b36cfcb72d63c97b543` โดยไม่มี dataset,
วิดีโอ, `.venv`, cache หรือ `.env` จริง

## 11. Checklist เมื่อกล้องมาถึง

ใช้ `reports/hikvision_camera_test_checklist_th.md` ตรวจรุ่น, PoE, IP/subnet, reachability, web interface, resolution/FPS/codec/bitrate, RTSP port, paths `101/102`, VLC/OpenCV และการไม่เปิดเผย password

## 12. คำสั่งถัดไปเพียงคำสั่งเดียว

หลังกล้องต่อและตั้งค่า `.env` แล้ว ให้รัน diagnostic นี้ก่อน:

```powershell
& '.\.venv\Scripts\python.exe' scripts\check_rtsp_source.py --source rtsp --profile HIKVISION_MAIN --frames 30 --seconds 10
```

ถ้า health ผ่าน ค่อยเริ่ม `run_rtsp_posture.bat` ตามสิทธิ์และ topology ที่ได้รับอนุญาต
