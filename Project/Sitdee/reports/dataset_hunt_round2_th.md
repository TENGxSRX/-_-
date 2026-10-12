# รอบค้นหา Dataset ภายนอกครั้งที่ 2

วันที่ตรวจ: 10 กันยายน 2026  
ขอบเขต: ค้นหาและตรวจความเหมาะสมของข้อมูลภายนอกเท่านั้น **ไม่ได้ฝึกโมเดลใหม่ ไม่ได้แก้ V4 ไม่ได้แก้ raw dataset เดิม**

## สรุปสั้น

ตรวจแหล่งข้อมูลทั้งหมด **14 แหล่ง** แล้ว ยังไม่พบชุด RGB สาธารณะชุดใหม่ที่ผ่านพร้อมกันทั้ง 4 เงื่อนไข: ป้ายกำกับรายไฟล์ที่ตรวจได้, group ID ที่ใช้กัน leakage ได้, สิทธิ์การใช้ที่ชัด, และความหมายตรงกับระบบท่านั่งของโครงการ

จึงไม่มีการเริ่ม V5 หรือการนำข้อมูลภายนอกเข้าฝึกในรอบนี้ ผลที่สำคัญกว่าคือพบว่า "ชื่อ dataset/ไฟล์" ไม่ใช่หลักฐานของ modality หรือ label เสมอไป โดยเฉพาะไฟล์เสริม MSR ที่ดาวน์โหลดมาแล้วมีแต่ skeleton ไม่ใช่ RGB ตามชื่อที่อ้าง

รายละเอียดเชิงโครงสร้างอยู่ใน [dataset_registry_v2.json](../data/dataset_registry_v2.json)

## 5 แหล่งที่มีค่าที่สุดในรอบนี้

| อันดับ | แหล่ง | สิ่งที่ยืนยันจากต้นทาง | ประโยชน์ที่แท้จริง | สถานะใช้ฝึกตอนนี้ |
|---:|---|---|---|---|
| 1 | MCD-MSD | งานหน้าเครื่องคอมพิวเตอร์ 10 คน; มีภาพ/วิดีโอ และ protocol SK/EK/EKC | ใกล้โจทย์ “correct vs usual sitting” มากที่สุด | หยุด: ใบอนุญาตและ label↔media join ยังไม่ตรวจ |
| 2 | MMD-MSD | 100 คนเต็มชุด, subset 5 คน, video/camera และ sensor ระหว่างทำงานหน้าคอมพิวเตอร์ | มีความหลากหลายด้านการใช้งานจริง | หยุด: full ชุดต้องขอ, สิทธิ์/label รายไฟล์ยังไม่ชัด |
| 3 | MOPED25 | 11 คน, 25 งานอาชีพ, RGB monocular ที่ sync กับ 37-marker kinematics | reference ดีด้าน ergonomic pose และ calibration | local release มีเพียง sample marker/โค้ด ไม่ใช่วิดีโอคนจริง |
| 4 | AnDyDataset | 13 คน, annotation EAWS จาก 3 annotators | reference ด้าน ergonomic taxonomy และ subject grouping | Zenodo ไม่แสดงสิทธิ์, 92.5 GB และวิดีโอที่ปล่อยเป็น avatar replay |
| 5 | MSR DailyActivity3D | 10 คน, 16 activity, RGB/depth/skeleton ตามหน้า creator | ใช้ศึกษา sit/stand และ action protocol ได้ | artifact ที่ดาวน์โหลดไม่ตรง metadata และไม่มี RGB |

## ผลตรวจแหล่งสำคัญ

### MCD-MSD — ใกล้ปัญหามากที่สุด แต่ยังไม่อนุญาตให้ฝึก

หน้า ErgoResearch ระบุชัดว่า MCD-MSD ประกอบด้วย motion capture, images, videos และ models จากผู้เข้าร่วม 10 คน งานทดลองให้พิมพ์หน้าคอมพิวเตอร์ใน 3 protocol:

- `SK`: ท่าตามธรรมชาติกับคีย์บอร์ดมาตรฐาน
- `EK`: ท่าตามธรรมชาติกับคีย์บอร์ด ergonomic
- `EKC`: ท่าที่ได้รับการสอนให้ถูกต้อง พร้อม pillow และคีย์บอร์ด ergonomic

นี่ไม่ใช่ label `good/bad` ที่ใช้ได้ทันที: `SK` และ `EK` คือ *usual posture* ไม่ควรถูกแปลงเป็น “incorrect” โดยอัตโนมัติ ส่วน `EKC` เรียกได้เพียง “instructed correct posture” จนกว่าจะเห็นไฟล์ label และการ join จริง หน้า source เปิดลิงก์ข้อมูล แต่ไม่ได้ระบุ license จึงยังไม่ดาวน์โหลด/ฝึก. [ErgoResearch](http://isr.tu-varna.bg/ergo/index.php/resursi), [paper](https://www.mdpi.com/2073-431X/11/7/116)

### MMD-MSD — ข้อมูลจริงใหญ่ แต่ยังไม่ใช่ corpus ฝึก

หน้า source ระบุ full corpus 100 คน มี video recorder, camera, PPG, EDA, skin temperature และ accelerometer ระหว่างทำงานกับคอมพิวเตอร์; มี public subset 5 คน และ full corpus “available upon request” ข้อมูลสาธารณะในเครื่องก่อนหน้านี้เป็นเพียง subset ภาพที่ audit แล้ว ไม่ใช่ release ที่ตรวจ join ได้. ไม่มี license ที่ source page จึงบล็อกการใช้ supervised ไว้ก่อน. [ErgoResearch](http://isr.tu-varna.bg/ergo/index.php/resursi), [paper](https://www.mdpi.com/1999-4893/17/5/187)

### MOPED25 — ergonomic video จริงตามบทความ แต่ local package ไม่ใช่ full corpus

บทความระบุ RGB monocular ที่บันทึกพร้อม kinematics ของ 11 คน ทำ 25 งาน รวม 790 motions. GitHub ทางการใช้ Apache-2.0 และระบุว่าข้อมูล marker/video อยู่ใน `.mat`/`.avi`; แต่ zip ที่ตรวจในเครื่องมีเพียงโค้ด, 4 `.mat` sample ของ `sub07`, และภาพเอกสาร 5 ภาพ ไม่มี `.avi` หรือภาพ RGB ของคน. จึงเก็บไว้เป็น reference ไม่ใช่ corpus RGB. [official repository](https://github.com/human-systems-ise-ncsu/MOPED25), [paper](https://doi.org/10.1016/j.jbiomech.2020.110086)

### AnDyDataset — annotation ดี แต่ modality ไม่ใช่ RGB คนจริงที่ต้องการ

Zenodo record บอกว่า 13 คนทำงานอุตสาหกรรม และมี EAWS action/posture annotations จากผู้ทำ annotation 3 คน. Record ระบุด้วยว่าวิดีโอที่ปล่อยเป็น human-avatar replay ของ motion ไม่ใช่กล้อง RGB คนจริง และรายการไฟล์รวม 92.5 GB จึงเกิน download budget ของรอบนี้ อีกทั้งช่อง license ไม่ได้ระบุค่า. [official AnDy page](https://andydataset.loria.fr/), [Zenodo record](https://zenodo.org/records/3254403)

### MSR DailyActivity3D — แยก “สิ่งที่ source บอก” ออกจาก “สิ่งที่ได้จริง”

หน้า creator ระบุ 10 คน × 16 activities × 2 repetitions และกล่าวว่ามี RGB AVI, depth BIN, skeleton TXT โดยทำ activity หนึ่งรอบในท่านั่งและอีกรอบในท่ายืน. อย่างไรก็ตาม archive เสริม PLOS ที่ดาวน์โหลดและ audit แล้วมี 321 entries: directory เดียว `MSRAction3DSkeletonReal3D` และ 320 ไฟล์ `.txt`; ไม่มี RGB/AVI, depth, หรือ class-name manifest. แม้จำนวน 320 สอดคล้องกับ protocol แต่ห้ามเดาว่า `a01…a16` map กับ class ใด จึงเป็น **metadata/content mismatch**. [creator page](https://sites.google.com/view/wanqingli/data-sets/msr-dailyactivity3d), [PLOS supporting record](https://doi.org/10.1371/journal.pone.0114147.s003)

## แหล่งอื่นที่ตรวจแล้ว

| แหล่ง | ข้อเท็จจริงที่พบ | การตัดสินใจ |
|---|---|---|
| UTD-MHAD | มี RGB/depth/skeleton/inertial และ action `sit-to-stand`, `stand-to-sit`; เป็น action transition ไม่ใช่ static desk posture | ไม่ใช่ label เป้าหมาย V5 |
| USSP / MSBN-SPose | paper รายงาน Upright, HeadDown, HandOnChin, Standing หลายมุมกล้อง | ไม่พบ public release/license ที่ตรวจได้ |
| Sitting Behaviors Healthy Population | 38 คน, 5 วันในสำนักงาน; public release เป็น pressure mat/bioradar CSV, วิดีโอ Kinect ใช้ทำ label แต่ไม่ได้ปล่อยเป็น RGB | no RGB adapter/training |
| SitPose | local เป็น partial Kinect/depth feature CSV ไม่มี group ID | ไม่ fusion กับ MediaPipe RGB |
| SitStraightNet | local snapshot มี JPEG 756 ภาพแต่ไม่มี sidecar label join | smoke/preprocessing เท่านั้น |
| MVSP/LGCSPNet | archive ยังไม่มี; Baidu Pan ต้องทำด้วยตนเอง | `WAITING_FOR_MANUAL_MVSP_ARCHIVE` |
| Toyota Smarthome | RGB/depth/skeleton และ participant metadata แต่ access gated | ไม่ดำเนินการแทนผู้ใช้ |
| Roboflow sitting keypoints | 655 ภาพสำหรับ keypoint detection แต่ไม่มี posture class/grouping ที่พิสูจน์ได้ | ไม่ใช้ supervise posture |

## คำตอบต่อความต้องการของโครงการ

### SIT / STAND

- มีหลักฐานระดับ protocol หรือ action ใน MSR และ UTD-MHAD
- แต่ยังไม่มี local RGB corpus ที่ label รายไฟล์และ participant split ตรวจได้
- ดังนั้น **ยังไม่มี SIT/STAND model ใหม่**

### CORRECT / INCORRECT

- MCD-MSD มี protocol ที่ใกล้ที่สุด (`EKC` instructed-correct) แต่ `SK/EK` ไม่ใช่ incorrect โดยนิยาม
- MMD-MSD อาจเกี่ยวกับ improper posture แต่ label ต่อไฟล์ยังไม่ตรวจ
- ดังนั้น **ยังไม่มี correct/incorrect ground truth ที่อนุญาตให้ฝึก**

### TUP / TLF / TLB / TLL / TLR

- มี ground truth ที่ตรวจได้ใน MultiPosture เท่านั้น
- แหล่งใหม่ส่วนใหญ่เป็น activity, EAWS, sensor หรือ label ที่ความหมายไม่เท่ากัน
- ดังนั้น **V4 และ MultiPosture model เดิมไม่ถูกแก้**

## Engineering decision

1. เก็บ external files ใหม่ไว้ใต้ `data/external/` แยกจาก `datasets/raw/` เดิม
2. ห้ามแปลง label หรือสร้าง feature adapter ใหม่จากข้อมูลภายนอกในรอบนี้
3. หากได้ permission/file manifest ของ MCD-MSD หรือ MMD-MSD ในอนาคต ให้ทำ file-level audit ก่อนดาวน์โหลด video จำนวนมาก
4. ทางลัดที่ถูกต้องที่สุดสำหรับ V5 ยังเป็นการเก็บ RGB จริงของกล้อง/มุมห้องเป้าหมาย พร้อม human label และ participant/session grouping
