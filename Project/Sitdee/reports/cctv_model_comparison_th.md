# CCTV Model Comparison

## ผลรอบนี้

ยังไม่มีการ train โมเดล candidate `CCTV_V5_BINARY` หรือ `CCTV_V5_MULTICLASS` เนื่องจากยังไม่มี trusted multi-person CCTV dataset ที่มี human labels และ grouping พร้อมใช้ การไม่รายงาน metric เป็น data-readiness decision ไม่ใช่ code failure

| โมเดล/ตระกูล | ใช้ในระบบ | สถานะ | ข้อจำกัด |
|---|---|---|---|
| RGB-native V4 LogisticRegression | baseline live | FROZEN / READY | TUP/TLB เท่านั้น; real CCTV generalization ยังไม่ validate |
| MultiPosture V1 HistGradientBoosting | direction baseline | READY | 126 features; source coordinate caveat |
| MultiPosture upper-body V2 HistGradientBoosting | seated robustness experiment | EXPERIMENTAL | 5 core landmarks; external semantics ยังไม่ proven |
| LogisticRegression / RandomForest / HistGradientBoosting CCTV_V5 | future candidate | NOT TRAINED | รอ controlled labels |

## Promotion gate

candidate ใหม่จะเลื่อนสถานะได้ต่อเมื่อผ่าน subject-independent grouped validation, cross-view/source test, temporal stability, core acceptance และ throughput ที่หลายคน โดยไม่ใช้ final test tuning

## Current decision

ใช้ V4 เป็น baseline ต่อไป ใช้ multi-person runtime กับ V4 ต่อคนได้ในเชิง software architecture แต่ไม่เรียกผลเป็น production accuracy และไม่ overwrite V4/V1/V2
