# ตารางความเข้ากันได้ของ Dataset ภายนอก

คำว่า “เข้ากันได้” ในรายงานนี้ไม่ใช่แค่มีคนอยู่ในภาพ แต่หมายถึง label, modality, coordinate semantics, grouping และสิทธิ์ ต้องผ่านพร้อมกันก่อนใช้สร้าง supervised model

| Dataset | RGB คนจริงที่ local | label source ที่ตรวจได้ | subject/session key | `TUP/TLF/TLB/TLL/TLR` | sit/stand | ergonomic correct/risk | MediaPipe feature fusion | ใช้ train รอบนี้ |
|---|---|---|---|---|---|---|---|---|
| MultiPosture | ไม่ | expert validated, 5 classes | มี subject | ตรง | ไม่ใช่โจทย์หลัก | ไม่มี binary label | ต้อง validate real-RGB adapter แยก | **เป็น source เดิมเท่านั้น** |
| DriveNet v6 | มี 3 clips | `SS→TUP`, `LB→TLB` เท่านั้น | clip/source | บางส่วน 2 class | ไม่ | ไม่ | domain shift เคยพบ | ไม่เพิ่ม train |
| MCD-MSD | ยังไม่มี local | SK/EK/EKC ระดับ protocol | source บอก 10 คน แต่ยังไม่เห็น files | ไม่มี | นั่ง | `EKC` เป็น instructed-correct เท่านั้น | ต้อง extract MediaPipe ใหม่ | ไม่ได้ |
| MMD-MSD | ไม่มี corpus ใหม่ | label ต่อไฟล์ไม่ตรวจ | 100/5 คนตาม source | ไม่พิสูจน์ | ไม่พิสูจน์ | ยังไม่พิสูจน์ | ต้อง audit | ไม่ได้ |
| MOPED25 sample | ไม่มี | task/motion, ไม่ใช่ posture class | sample `sub07` | ไม่มี | task dependent | ไม่มี direct | marker ≠ MediaPipe โดยอัตโนมัติ | ไม่ได้ |
| AnDyDataset | ไม่มี real-human RGB local | EAWS posture/action | reported participant structure | ไม่ตรง | ไม่ใช่ primary | EAWS ≠ binary label | avatar/video semantics แยก | ไม่ได้ |
| MSR artifact | ไม่มี | `aXX` ไม่มี manifest | `sXX` filename | ไม่มี | protocol-level only | ไม่มี | skeleton type unknown | ไม่ได้ |
| UTD-MHAD | ไม่มี | sit-to-stand / stand-to-sit action | source protocol | ไม่มี | transition only | ไม่มี | ต้อง audit | ไม่ได้ |
| SitPose | ไม่ | 7 source labels แต่ local partial | ไม่มี | approximate บาง labelเท่านั้น | มี stand label | `wrong` ยังไม่ map | Kinect features ≠ MediaPipe | ไม่ได้ |
| SitStraightNet | มี JPEG | ไม่มี join รายภาพ | ไม่มี | source paper-level only | ไม่ | ไม่ | smoke test ได้ | ไม่ได้ |
| Sitting Behaviors | ไม่ | sitting labels ของ sensor data | participant/day reported | ไม่ตรวจ | sitting only | ไม่ | ไม่มี RGB | ไม่ได้ |
| USSP | ไม่มี | paper-level four classes | ไม่ทราบ | Upright approximate เท่านั้น | standing direct | ไม่ | public release ยังไม่พบ | ไม่ได้ |
| MVSP | ยังไม่มี archive | ยังไม่ตรวจ | ยังไม่ตรวจ | ยังไม่ตรวจ | ยังไม่ตรวจ | ยังไม่ตรวจ | ยังไม่ตรวจ | ไม่ได้ |
| Toyota Smarthome | ไม่มี | daily activity | access gated | ไม่ | activity dependent | ไม่ | future research only | ไม่ได้ |

## ความหมายของช่องสำคัญ

- **label source ที่ตรวจได้**: ต้องเป็น manifest/annotation ที่ join กับ media ได้ ไม่ใช่เพียงข้อความใน paper
- **subject/session key**: ต้องใช้ตรวจว่า train และ test ไม่มีคน/วิดีโอเดียวกันข้ามกัน
- **MediaPipe feature fusion**: ต้องตรวจ landmark identity, image/world coordinate, normalization และ feature order ก่อนเสมอ
- **approximate**: ใช้เป็น hypothesis สำหรับ audit เท่านั้น ไม่ใช่การ map เพื่อเทรน

## แผนที่ label ที่อนุญาตในปัจจุบัน

| Project label | แหล่งที่อนุญาตให้ถือเป็น ground truth ในโครงการ | แหล่งที่ห้าม map แบบอัตโนมัติ |
|---|---|---|
| `TUP` | MultiPosture; DriveNet `SS` เฉพาะ pilot ที่มีข้อจำกัด | healthy, upright จาก paper อื่น, `EKC` |
| `TLB` | MultiPosture; DriveNet `LB` เฉพาะ pilot | lie/lay, reclined ที่ไม่มี definition เดียวกัน |
| `TLF` | MultiPosture | head-down, hunched, forward activity label |
| `TLL/TLR` | MultiPosture | left/right ที่ไม่บอกว่า trunk lean |
| acceptable / ergonomic risk | **ยังไม่มี source ใหม่ที่ใช้ได้** | usual posture, any `Predicted_Label`, cluster, filename |
| sit / stand | **ยังไม่มี grouped RGB source local** | action transition, protocol description หากไม่มี per-file join |

## ข้อสรุป

ตอนนี้ external dataset ที่ผ่าน “RGB + label join + grouping + license + semantic match” สำหรับ V5 มีจำนวน **0**. นี่ไม่ใช่ความล้มเหลวของ model; เป็นผล audit ที่ป้องกันการฝึกบน label ผิดความหมายและ leakage.
