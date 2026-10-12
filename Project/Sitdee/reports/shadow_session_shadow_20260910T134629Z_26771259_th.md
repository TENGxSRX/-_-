# รายงาน Shadow Observation

Session: `shadow_20260910T134629Z_26771259`
ระยะเวลา: `28.06200000003446` วินาที | observations: `186` | inference frames: `221`

## ข้อจำกัดสำคัญ

ข้อมูลนี้เป็น telemetry วิศวกรรมแบบไม่ระบุตัวตน ไม่บันทึกวิดีโอดิบ และไม่ใช่ ground truth หรือฉลากสำหรับ train โมเดล.

## ผลที่วัดได้

- pose detected: `52.68817204301075`% | core valid: `52.68817204301075`%
- activity: `{'UNKNOWN': {'count': 186, 'percentage': 100.0}}`
- V4 executed: `0` | distribution: `{}`
- source frame age: `{'median': 16.00000006146729, 'p95': 62.000000034458935}` ms | inference FPS: `{'median': 7.947929624222498, 'p95': 8.684002875369268}`

## Landmark availability

- nose: `100.0`% (186)
- shoulders_bilateral: `100.0`% (186)
- hips_bilateral: `100.0`% (186)
- knees_bilateral: `1.6129032258064515`% (3)
- ankles_bilateral: `0.0`% (0)

## UNKNOWN reasons

- `LOW_LOWER_BODY_VISIBILITY`: 98 (52.68817204301075% ของ UNKNOWN observations)
- `DESK_OCCLUSION_SUSPECTED`: 98 (52.68817204301075% ของ UNKNOWN observations)
- `POSE_LOST`: 88 (47.31182795698925% ของ UNKNOWN observations)

## Bottleneck ranking

1. **LOWER_BODY_VISIBILITY** — activity UNKNOWN 100.0%; knees bilateral 1.6%; ankles bilateral 0.0%
2. **POSE** — pose detected 52.7%; core valid 52.7%
3. **TRACKING** — temporary lost + reacquired 76 across 221 inference frames
4. **INSUFFICIENT_GROUND_TRUTH** — Shadow telemetry contains no human labels by design; it cannot measure activity or posture accuracy.

## ชุดเก็บข้อมูลที่เล็กที่สุดที่แนะนำ

ตัวขับหลัก: **LOWER_BODY_VISIBILITY**

เก็บ human-labelled ชุดเล็ก 3 คนแบบไม่ระบุตัวตน: ยืน/นั่งอย่างละ 20–30 วินาที ในตำแหน่งที่เห็นขาชัด 1 จุด และหลังโต๊ะ 1 จุดต่อคน เพื่อแยกปัญหามุมกล้อง/โต๊ะจาก activity gate.
