# Dataset Fusion Registry

วันที่ตรวจสอบ: 9 กันยายน 2026  
หลักการ: ใช้ข้อมูลจากไฟล์/รายงานที่ตรวจจริงเท่านั้น และไม่รวม dataset แบบเดาสุ่ม

| Dataset | ตัวอย่างที่วัดได้ | ผู้เข้าร่วม | RGB ดิบ | รูปแบบ | label/ความหมาย | grouping | trust | supervised ได้หรือไม่ |
|---|---:|---:|---|---|---|---|---|---|
| MultiPosture | 4,794 แถว | 13 | ไม่ใช่ RGB | 33 indexed XYZ + labels | TUP, TLF, TLB, TLL, TLR upper-body | subject มี | VERIFIED_GROUND_TRUTH | ได้ |
| DriveNet v6 | 1,800 sampled frames | ไม่ยืนยันเป็น participant field | มี RGB video | MediaPipe features จาก video | TUP/TLB mapping จำกัด | clip/source มี | PARTIAL_OR_UNCERTAIN_LABELS | ได้เฉพาะ pilot เดิม |
| ANYA | 605,955 ต่อไฟล์ | ไม่มี participant ที่ยืนยัน | ไม่ใช่ RGB ที่ join ได้ | 33 raw slots + 13 normalized slots | `Predicted_Label`/cluster ไม่ใช่ verified target | filename/cluster ยังไม่พอ | PSEUDO_LABEL_ONLY | ไม่ได้ |
| SitPose | 5,746 แถวใน local CSV | ไม่มีใน CSV | ไม่ใช่ RGB-native | joint/angle features จาก Kinect/depth | 7 source labels | ไม่มี session/frame grouping | PARTIAL_OR_UNCERTAIN_LABELS | ไม่ได้ |
| SitStraightNet | 756 JPEG | ไม่มี label join | มีภาพ | JPEG | ไม่มี sidecar label ที่ยืนยัน | ไม่มี | UNLABELED | ไม่ได้ |
| MMD-MSD public subset | 35 ภาพที่ใช้ใน audit | 5 ตาม subset protocol แต่ mapping จำกัด | มีภาพบางส่วน | project-extracted pose | ใช้ TUP diversity แบบ protocol-limited | mapping ไม่สมบูรณ์ | PARTIAL_OR_UNCERTAIN_LABELS | ไม่ได้สำหรับ production |
| MVSP/LGCSPNet | ยังไม่มี local archive | ยังไม่ทราบ | ยังไม่ทราบ | ยังไม่ทราบ | repository names อย่างเดียว | ยังไม่ทราบ | UNUSABLE_FOR_CURRENT_TASK | ไม่ได้ |

## การตัดสินใจ fusion

1. MultiPosture เป็น supervised baseline หลักของ direction model
2. DriveNet ใช้เป็น RGB-native two-class pilot/diagnostic เท่านั้น
3. ANYA ใช้ distribution/representation audit ได้ แต่ห้ามใช้ pseudo-label เป็น ground truth
4. SitPose ไม่สามารถต่อเข้ากับ RGB MediaPipe schema ได้โดยตรง
5. SitStraightNet ใช้ smoke/preprocessing เท่านั้น
6. MMD-MSD subset ยังไม่พอเป็นแหล่ง label production
7. MVSP ยังอยู่สถานะ `WAITING_FOR_MANUAL_MVSP_ARCHIVE` และห้ามเดาความเหมาะสมก่อนตรวจ archive จริง

## กติกาการรวมข้อมูล

ทุก sample ที่จะเข้า experiment ต้องมี `dataset_id`, original label, canonical label ที่มีหลักฐาน, coordinate type, feature schema version และ group key เท่าที่มี หาก mapping หรือ coordinate semantics พิสูจน์ไม่ได้ ให้คง sample ไว้ใน audit แต่ไม่ให้เข้า supervised fusion

ยังไม่มีการ train CCTV_V5 หรือ hierarchical production model ในรอบนี้ เพราะ trusted RGB multi-person CCTV labels ยังไม่พร้อม
