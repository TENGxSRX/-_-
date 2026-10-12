# Audit การดาวน์โหลดข้อมูลภายนอก — รอบ 2

วันที่ตรวจ: 10 กันยายน 2026  
นโยบาย: เก็บไฟล์ต้นฉบับแบบ immutable, ตรวจ hash, ตรวจรายชื่อใน archive ก่อน extraction และไม่เขียนทับ `datasets/raw/`

## ผลดาวน์โหลดจริง

ดาวน์โหลดเพิ่ม **2 archive ขนาดเล็กเพื่อ audit เท่านั้น** รวม `15,581,203` bytes (`14.86 MiB`). ไม่มีการดาวน์โหลด raw RGB/video ใหม่ในรอบนี้

| Dataset / source | Local file | bytes | SHA-256 | ผล inspect โดยไม่ extract | สถานะ |
|---|---|---:|---|---|---|
| MOPED25 official GitHub release | `data/external/moped25/official_github_release/MOPED25-master.zip` | 3,141,361 | `c3a712e0305dd3077020110867ebe905ee03f4f4c4ccba76f76254f6884ee1bf` | 20 files: 4 `.mat` sample, MATLAB code, ภาพเอกสาร; raw human RGB video = 0 | `AUDIT_ONLY_SAMPLE` |
| PLOS supplementary file named MSR DailyActivity3D | `data/external/msr_daily_activity3d/official_plos_supporting_data/msr_daily_activity3d_dataset_s3.rar` | 12,439,842 | `5394a12f7ab4a5cc15d091ba32ed5317ae91b5705451db986d3d23ff9bdba260` | 321 entries: directory `MSRAction3DSkeletonReal3D` + 320 `.txt`; RGB/depth/video = 0 | `DO_NOT_TRAIN_ARTIFACT_MISMATCH` |

## MOPED25 package inspection

- source: [official GitHub repository](https://github.com/human-systems-ise-ncsu/MOPED25)
- repository license: Apache-2.0
- entries with data: `sub07_09_01_trc.mat`, corresponding box/virtual-marker `.mat` files
- `.avi`, `.mp4`, `.mov`, `.mkv`: **0 entries**
- `.png` exists only as documentation/interface figures, not human-camera data

การตีความที่อนุญาต: local package เป็น sample marker/utility release จากต้นทาง  
การตีความที่ห้าม: ห้ามเรียกว่า "MOPED25 full RGB dataset" หรือใช้ task name เป็น posture target

## MSR archive inspection

- source record: [PLOS supporting file S3](https://doi.org/10.1371/journal.pone.0114147.s003)
- archive MIME reported by server: `application/x-rar-compressed`
- `tar -tf` อ่าน archive ได้สำเร็จโดยไม่ต้อง extract
- 320 filenames อยู่ในรูป `aXX_sXX_eXX_skeleton3D.txt`, 16 action slots × 10 subject slots × 2 execution slots
- ชื่อ directory ใน archive คือ `MSRAction3DSkeletonReal3D`, ไม่ตรงกับชื่อ supporting record ที่อ้าง MSR DailyActivity3D
- ไม่มี class-name manifest; จึงยังพิสูจน์ไม่ได้ว่า `a01...a16` คือ activity ใด
- ไม่มี RGB/depth file แม้หน้า creator ของ MSR DailyActivity3D ระบุ raw RGB AVI/depth/skeleton สำหรับ source corpus

ผล engineering: archive นี้ยังคงเก็บเป็น raw source artifact แต่ถูกปิดสิทธิ์ training และไม่มีการ extract/transform

## แหล่งที่ “ไม่ดาวน์โหลด” และเหตุผล

| แหล่ง | เหตุผลที่ไม่ดาวน์โหลด |
|---|---|
| MMD-MSD full | source ระบุ full corpus 100 คนให้ขอ access; public subset มี 5 คน แต่ license ไม่ระบุ |
| MCD-MSD | public link มี แต่ source ไม่ระบุ license และยังไม่เห็น label/media manifest |
| AnDyDataset | 92.5 GB เกิน budget 30 GB และ Zenodo record ไม่มี license value |
| MOPED25 full media | บทความกล่าวว่า public แต่ current automated audit พบเพียง GitHub sample; full download endpoint ยังยืนยันไม่ได้ |
| MSR original RGB folder | หน้า creator มี public shared folder แต่ local downloadable PLOS artifact ไม่ใช่ RGB; ห้ามแทนกันโดยเดา |
| Toyota Smarthome / MVSP | gated/request/manual Baidu access |

## Integrity

- ไม่มีไฟล์ใน `datasets/raw/` ถูกแก้โดยงานนี้
- ไม่ extract archive ทับ source หรือ project data
- V4 model hash ก่อนงาน: `af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd` (ตรง expected)
- V4 model hash หลังงาน: `af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd` (ตรง expected และตรงกับก่อนงาน)
