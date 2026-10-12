# ANYA / SitDee — Security & Privacy

## ความลับและ credential
- **ห้าม commit `.env`** (ถูก ignore แล้ว) ให้ใช้ `.env.example` ซึ่งมีแต่ค่าตัวอย่างปลอม แล้วคัดลอกเป็น `.env` ในเครื่องเท่านั้น
- ห้ามใส่ RTSP URL ที่มี username/password ใน README, config ที่ commit, log, report หรือข้อความแชท
  - ให้ใช้ profile (`--profile HIKVISION_MAIN`) และค่าจาก `.env`
  - โค้ดมีการซ่อน credential ในข้อความ diagnostic
- ห้าม commit `kaggle.json`, API key, token, private key หรือรหัส Wi-Fi
- ไฟล์ access code ของ dashboard / review app (`outputs/.../access_code.txt`) อยู่ใน `outputs/` ซึ่ง ignore อยู่แล้ว และห้ามพิมพ์ลง log (supervisor ซ่อนบรรทัดนี้ให้)

## ถ้า credential หลุด (rotation)
1. เปลี่ยนรหัสผ่านกล้อง, token หรือ key นั้นทันทีที่ต้นทาง (กล้อง / NVR / บริการ)
2. อัปเดต `.env` ในเครื่องที่เกี่ยวข้อง
3. ถ้าหลุดเข้า Git ให้ถือว่า credential นั้นใช้ไม่ได้แล้ว การลบออกจาก history ไม่พอ ต้องเปลี่ยนรหัสด้วย
4. ตรวจ history ด้วย `git log --all -S "<ค่า>"` **ในเครื่องเท่านั้น** อย่าพิมพ์ค่าลงในที่สาธารณะ

## วิดีโอ CCTV และข้อมูลส่วนบุคคล
- วิดีโอ / ภาพผู้เข้าร่วมเป็นข้อมูลส่วนบุคคล **ห้าม commit และห้ามอัปโหลดไปบริการภายนอก** (รวมถึงบริการ AI)
- ปิดการบันทึกวิดีโอดิบเป็นค่าเริ่มต้น ส่วน dashboard บันทึกได้เฉพาะเมื่อผู้ควบคุมยืนยันความยินยอม (video / review / evaluation / training แยกกัน)
- ใช้รหัสนามแฝง (เช่น P001, R002) ห้ามใช้ชื่อจริงในไฟล์ที่ commit
- ข้อมูลที่ใช้ฝึก / ประเมินต้องผ่านด่านที่ตรวจความยินยอม, ตัวตน และ provenance ห้ามใช้ข้อมูลที่ไม่ได้รับอนุญาต
- **ไม่ใช้ face recognition** และ track ID ไม่ใช่ตัวตนของบุคคล

## พฤติกรรมเครือข่ายที่แนะนำ
- ให้ dashboard และ review app bind กับ `127.0.0.1` หรือ IP Tailscale เท่านั้น **ห้าม bind `0.0.0.0` และห้ามเปิดสู่ internet**
- ใช้ allow-list ของ IP, access code, cookie แบบ HttpOnly / SameSite=Strict และ header กัน CSRF (มีอยู่แล้วใน branch dashboard / ML)
- พัฒนาและทดสอบแบบ offline ก่อน test ห้ามเปิดกล้องหรือ RTSP จริง

## การรายงานปัญหาความปลอดภัย
repository นี้เป็นงานวิจัยส่วนตัว ถ้าพบช่องโหว่หรือข้อมูลหลุด **อย่าเปิด issue สาธารณะ** ให้แจ้งเจ้าของ repository โดยตรง และถ้าเป็น credential ให้ rotate ทันที
