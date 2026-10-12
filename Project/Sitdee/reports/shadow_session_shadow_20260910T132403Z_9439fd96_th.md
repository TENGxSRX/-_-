# รายงาน Shadow Observation

Session: `shadow_20260910T132403Z_9439fd96`
ระยะเวลา: `6.734000000054948` วินาที | observations: `68` | inference frames: `34`

## ข้อจำกัดสำคัญ

ข้อมูลนี้เป็น telemetry วิศวกรรมแบบไม่ระบุตัวตน ไม่บันทึกวิดีโอดิบ และไม่ใช่ ground truth หรือฉลากสำหรับ train โมเดล.

## ผลที่วัดได้

- pose detected: `100.0`% | core valid: `97.05882352941177`%
- activity: `{'UNKNOWN': {'count': 68, 'percentage': 100.0}}`
- V4 executed: `0` | distribution: `{}`
- source frame age: `{'median': 15.999999945051968, 'p95': 52.25000002537851}` ms | inference FPS: `{'median': 4.954076873106239, 'p95': 5.101499008138145}`

## Landmark availability

- nose: `97.05882352941177`% (66)
- shoulders_bilateral: `100.0`% (68)
- hips_bilateral: `100.0`% (68)
- knees_bilateral: `2.9411764705882355`% (2)
- ankles_bilateral: `0.0`% (0)

## UNKNOWN reasons

- `LOW_LOWER_BODY_VISIBILITY`: 68 (100.0% ของ UNKNOWN observations)
- `DESK_OCCLUSION_SUSPECTED`: 66 (97.05882352941177% ของ UNKNOWN observations)
- `CORE_INVALID`: 2 (2.9411764705882355% ของ UNKNOWN observations)

## Bottleneck ranking

1. **LOWER_BODY_VISIBILITY** — activity UNKNOWN 100.0%; knees bilateral 2.9%; ankles bilateral 0.0%
2. **INSUFFICIENT_GROUND_TRUTH** — Shadow telemetry contains no human labels by design; it cannot measure activity or posture accuracy.

## ชุดเก็บข้อมูลที่เล็กที่สุดที่แนะนำ

ตัวขับหลัก: **LOWER_BODY_VISIBILITY**

เก็บ human-labelled ชุดเล็ก 3 คนแบบไม่ระบุตัวตน: ยืน/นั่งอย่างละ 20–30 วินาที ในตำแหน่งที่เห็นขาชัด 1 จุด และหลังโต๊ะ 1 จุดต่อคน เพื่อแยกปัญหามุมกล้อง/โต๊ะจาก activity gate.
