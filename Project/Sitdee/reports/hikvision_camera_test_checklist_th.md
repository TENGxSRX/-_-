# เช็กลิสต์ทดสอบ Hikvision เมื่อกล้องมาถึง

ทำตามลำดับและหยุดทันทีถ้าการเปลี่ยนแปลงอาจกระทบระบบกล้องของสถาบัน ไม่ต้อง factory reset และไม่แก้ configuration เดิมโดยไม่ได้รับอนุญาต

## ข้อมูลและการเชื่อมต่อ

- [ ] รุ่นกล้องและ firmware
- [ ] วิธีจ่ายไฟ: PoE switch หรือวิธีอื่น
- [ ] Ethernet link ขึ้นที่กล้องและ PoE switch
- [ ] จด camera IP จากผู้ดูแล ไม่เดา IP
- [ ] จด IP/subnet ของ Acer และ Desktop
- [ ] ตรวจ reachability/ping ตาม policy ของ network
- [ ] web interface ของกล้องเข้าถึงได้จากจุดที่ทดสอบ

## ค่าของ stream

- [ ] main stream resolution
- [ ] sub stream resolution
- [ ] FPS ที่ตั้งจริง
- [ ] codec เช่น H.264/H.265
- [ ] bitrate ที่ตั้งจริง เพื่อนำไปคำนวณ bandwidth
- [ ] RTSP เปิดใช้งานอยู่หรือไม่
- [ ] RTSP port ที่ผู้ดูแลยืนยัน
- [ ] ทดสอบ path channel 1 main: `Streaming/Channels/101`
- [ ] ทดสอบ path channel 1 sub: `Streaming/Channels/102`

## ตรวจจากเครื่องที่จะประมวลผล

- [ ] สร้าง `.env` จาก `.env.example` ด้วยค่าจริงบนเครื่องที่ปลอดภัย
- [ ] ไม่ส่ง password ในแชต, screenshot, log หรือ report
- [ ] รัน `scripts/check_rtsp_source.py` แบบจำกัด frame/duration
- [ ] ทดลอง VLC หรือ OpenCV ตามสิทธิ์ที่ได้รับ
- [ ] ตรวจ resolution, reported/effective FPS, reconnect และ last-frame age
- [ ] ตรวจว่า diagnostic ไม่สร้าง recording หรือ screenshot
- [ ] ค่อยรัน `run_rtsp_posture.bat` หลัง source health ผ่าน

## หลักฐานที่ควรเก็บ

เก็บเฉพาะรุ่น, stream profile, resolution, FPS, codec, health summary และเวลาทดสอบ ไม่เก็บ URL ที่มี credentials และไม่จำเป็นต้องเก็บวิดีโอจริง
