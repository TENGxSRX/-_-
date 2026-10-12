# Sit/stand และ sitting-quality dataset audit

สถานะ standing: **STANDING_DATA_INSUFFICIENT**
สถานะ sitting quality: **SITTING_POSTURE_DATA_INSUFFICIENT**

## ข้อสรุป

ยังไม่มีชุดข้อมูล RGB/MediaPipe ที่มี label sit/stand และ participant/session grouping ที่ตรวจสอบได้เพียงพอสำหรับเทรน production model.
จึงยังไม่ train activity classifier, sitting-quality classifier หรือ subtype classifier.

## หลักฐานจาก registry

| Dataset | RGB | label ที่พบ | ใช้ train ตอนนี้ | ข้อจำกัด |
|---|---:|---|---:|---|
| MultiPosture | False | TUP, TLF, TLB, TLL, TLR | True | Primary supervised baseline. Local row count is 4,794, not the prose count of 4,800. |
| DriveNet v6 | True | TUP, TLB | True | Used for the frozen two-class RGB-native pilot and smoke/diagnostic validation, not a five-class CCTV corpus. |
| ANYA | False | Predicted_Label, clusterid, normclusterid | False | Use for distribution/representation audits only until a source-backed label mapping is joinable. |
| SitPose | False | forward, healthy, lay, left, right, stand, wrong | False | Useful as an audit/reference source, not eligible for blind feature fusion. |
| SitStraightNet | True | NONE_VERIFIED | False | Smoke/preprocessing only; never use image filename as ground truth. |
| MMD-MSD public subset | True | TUP | False | Audit/diversity reference only; not sufficient to train the missing directions. |
| MVSP/LGCSPNet | None | NONE_VERIFIED | False | Manual archive and license verification remain pending; do not infer usefulness. |

## Engineering decision

- MultiPosture ใช้เป็น direction expert เดิมเท่านั้น ไม่ใช่ sit/stand ground truth.
- SitPose มี label `stand` แต่เป็นข้อมูล Kinect/depth และไม่มี grouping ที่พอสำหรับ CCTV RGB จึงเป็น reference เท่านั้น.
- SitStraightNet เป็นภาพที่ยังไม่มี verified per-file labels.
- ANYA เป็น pseudo-label/distribution audit เท่านั้น.
- ต้องเก็บ CCTV ground truth แบบ human-labelled และใช้ participant/session grouped split ก่อนเริ่ม V5.
