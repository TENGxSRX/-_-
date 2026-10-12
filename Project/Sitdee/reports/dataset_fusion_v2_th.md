# Dataset fusion V2 decision

## อนุญาตให้ใช้เป็นอะไร

| แหล่งข้อมูล | การใช้ที่ปลอดภัยตอนนี้ |
|---|---|
| MultiPosture | frozen direction expert / grouped baseline |
| DriveNet v6 | RGB smoke/diagnostic และ V4 pilot ที่ mapping จำกัด |
| Controlled CCTV V2 | future human-labelled evaluation/training เมื่อ readiness gate ผ่าน |
| ANYA | schema/distribution audit เท่านั้น; pseudo-label ไม่ใช่ target |
| SitPose | reference audit; depth/Kinect ไม่ใช่ RGB feature fusion |
| SitStraightNet | image preprocessing smoke test เมื่อไม่มี verified labels |
| MMD-MSD subset | diversity/audit reference; mapping จำกัด |
| MVSP/LGCSPNet | รอ archive/license/media join |

## กติกา fusion

ห้าม concatenate features ข้าม coordinate convention โดยเดา. ทุก sample ต้องมี source, coordinate type, feature schema version, participant/session/video group และ label provenance. การแบ่งข้อมูลต้อง group ก่อนทำ preprocessing/resampling; ห้ามสุ่ม frame ที่ติดกันข้าม train/test.

## สถานะ

ยังไม่มี cross-dataset training result ที่ scientific valid สำหรับ CCTV hierarchy และยังไม่ train V5.
