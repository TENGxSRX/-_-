# คู่มือเก็บข้อมูลทดสอบความทนทานของ AI แบบควบคุม

คู่มือนี้ใช้กับข้อมูลใหม่ที่มนุษย์เป็นคนเลือกป้ายกำกับ เพื่อดูว่า V4 ทำงานกับ
ภาพ RGB จริงได้สม่ำเสมอเพียงใด และเพื่อเตรียมข้อมูลสำหรับพิจารณา V5 ในภายหลัง
ไม่ใช่การวินิจฉัยสุขภาพหรือการรับรองท่านั่งทางการแพทย์

## กติกาหลัก

1. ใช้ผู้เข้าร่วมที่ยินยอมและใช้รหัสแทนชื่อ เช่น `P001`, `P002`, `P003`
2. เก็บเฉพาะ 2 ป้ายที่ V4 รองรับ: `TUP` และ `TLB`
3. ผู้ควบคุมการทดลองเป็นคนเลือก label; ห้ามใช้ผล AI มาแทน label
4. `TUP` คือท่านั่งตัวตรงตามคำอธิบายของการทดลองนี้
5. `TLB` คือท่าเอนหลังตามคำอธิบายของการทดลองนี้
6. ไม่ต้องทำท่าที่ฝืน เจ็บ หรือเสี่ยง ให้หยุดทันทีเมื่อไม่สบาย
7. เก็บวิดีโอ raw เป็นค่าเริ่มต้น **ปิด** เครื่องมือเก็บเฉพาะ feature/telemetry
8. ข้อมูลที่ผ่าน gate เท่านั้นจะเข้า `samples.csv`; ข้อมูลตก gate จะอยู่ใน
   `rejections.jsonl` พร้อมเหตุผล

## แนะนำแผน pilot

ใช้ผู้เข้าร่วมประมาณ 3–5 คน เพื่อให้มีความแตกต่างของรูปร่าง มุมกล้อง เสื้อผ้า
และตำแหน่งนั่ง โดยแต่ละคนควรมีหลายช่วงสั้น ๆ ของทั้งสองป้าย เช่น

- `P001_TUP_01`, `P001_TUP_02`
- `P001_TLB_01`, `P001_TLB_02`
- ทำเช่นเดียวกันกับ `P002` ถึง `P005` ตามเวลาที่ปลอดภัย

เป้าหมายคือความหลากหลายและการแยกผู้เข้าร่วม ไม่ใช่การถ่ายเฟรมซ้ำจำนวนมาก
จากคลิปเดียว ก่อนใช้ฝึก V5 ต้องตรวจ sample count, class balance, duplicate/near-
duplicate และ subject-grouped split ก่อนเสมอ

## วิธีเริ่ม

ตั้งค่า credential ไว้ใน local `.env` หรือ process environment เท่านั้น ตาม
`.env.example` ห้ามใส่ password ใน command history, source code, log หรือภาพ
หน้าจอ จาก project root รัน:

```powershell
& '.\.venv\Scripts\python.exe' scripts\collect_controlled_posture.py --source rtsp --profile HIKVISION_MAIN --subject-id P001 --clip-id P001_TUP_01
```

ถ้าต้องใช้ sub stream:

```powershell
& '.\.venv\Scripts\python.exe' scripts\collect_controlled_posture.py --source rtsp --profile HIKVISION_SUB --subject-id P001 --clip-id P001_TLB_01
```

เครื่องมือต้องเปิด preview เพื่อให้กดปุ่มได้ หากไม่มีหน้าต่าง preview จะไม่มี
sample ที่ติด label ถูกสร้างขึ้น

## การควบคุมบนหน้าต่าง

| ปุ่ม | ความหมาย |
|---|---|
| `1` | เลือก human label `TUP` |
| `2` | เลือก human label `TLB` |
| `SPACE` | เริ่ม/หยุดการบันทึก accepted frames หลังเลือก label แล้ว |
| `R` | หยุด capture และล้าง label ที่เลือก |
| `Q` หรือ `ESC` | จบ session และเขียน metadata |

ควรกดเลือก label ให้ตรงกับท่าที่ผู้เข้าร่วมกำลังทำ แล้วกด `SPACE` เฉพาะช่วง
ที่ต้องการเก็บข้อมูล หากเปลี่ยนจาก TUP เป็น TLB ให้หยุดก่อน เลือก `2` แล้ว
กด `SPACE` ต่อ หรือแนะนำให้จบ process และเปิด clip id ใหม่เพื่อให้ audit ง่าย

## Quality gate

accepted frame ต้องมี:

- pose detected
- nose, ไหล่ซ้าย/ขวา และสะโพกซ้าย/ขวา visibility ถึง threshold ใน config
- V4 feature ครบ 15 ค่าและเป็น finite numbers
- frame dimensions ใช้ได้

ค่า V4 confidence และ geometry ถูกเก็บเป็นข้อมูลประกอบ ไม่ถูกใช้แทน human
label และไม่ทำให้ label กลายเป็น pseudo-label

## ป้องกัน data leakage

- `subject_id` ต้องใช้แยก train/validation/test
- ห้ามสุ่มเฟรมจากคนเดียวกันไปอยู่ทั้ง train และ test
- ไม่รวม session/clip เดิมข้าม split หากมาจากผู้เข้าร่วมและช่วงเดียวกัน
- หลังเก็บต้องตรวจ duplicate และ temporal adjacency
- V4 baseline ประเมินบนข้อมูลใหม่ได้ แต่ห้ามเลือก label ด้วย V4 prediction

## ไฟล์ผลลัพธ์

```text
data/controlled_rgb_v5/
├─ samples.csv          accepted feature rows + human_label
├─ rejections.jsonl     rejected frames + transparent reasons
└─ sessions/            metadata ของแต่ละ session
```

ใน `samples.csv`:

- `human_label` = label จากคน
- `v4_prediction` = ผลจาก frozen V4 เพื่อเปรียบเทียบ
- `v4_confidence` และ `v4_probabilities_json` = ความมั่นใจของ V4
- feature columns = ลำดับเดียวกับ V4 artifact

## ก่อนอนุญาตให้ train V5

ต้องมีข้อมูลที่ตรวจสอบแล้วและมีอย่างน้อยหลาย participant groups ต่อ class,
class balance ที่อธิบายได้, schema ตรง, ไม่มีกลุ่มรั่วข้าม split และมีรายงาน
V4 baseline ก่อน เครื่องมือ train V5 ปัจจุบันจะไม่ถูกเรียกในช่วงที่ยังไม่มี
ข้อมูลนี้
