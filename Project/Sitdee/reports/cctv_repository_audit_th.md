# Complete CCTV repository audit

ตรวจบน `DESKTOP-ALH6CUB` วันที่ 10 กันยายน 2026 โดยไม่เปิดกล้องและไม่ train model

| ชั้น | จุดตรวจหลัก | ผล |
|---|---|---|
| source | `src/posture_ai/inference/video_source.py`, `latest_frame_runtime.py` | USB/file/RTSP, latest-frame, reconnect/stale พร้อม |
| pose | `src/posture_ai/multiperson/mediapipe_tasks.py` | `max_people=1..4` พร้อม; MediaPipe เป็นหลัก |
| tracking | `multiperson/tracking.py` | anonymous IDs, continuity; ยังไม่มี manual ID ground truth |
| per-person inference | `multiperson/runtime.py` | V4/Geometry/Temporal state แยกต่อคน |
| seat layer | `multiperson/seating.py` | มี contract; ต้องใช้ seat layout จริง |
| hierarchy | `ergonomics/hierarchical.py` | evidence-gated contract; models not trained |
| optional YOLO | `detection/yolo_optional.py` | lazy interface; no install/weights |
| canonical schema | `features/canonical_cctv.py` | versioned; not V4-compatible by assumption |
| collection | `scripts/collect_controlled_cctv.py` | manual labels, no raw video by default |
| readiness gate | `scripts/audit_cctv_training_readiness.py` | current status NOT_READY |
| tests | `tests/` | 165 passed |

## Integrity

- V4 SHA-256: `af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd`
- V1 SHA-256: `1db2dc98ddbfd8b27ed0827dfc6df320f9c9f714fc71e388c4f0cc09910f0ec9`
- V2 SHA-256: `8c4b77327d91f136ebfe706385668f0d0c61d07cc1eab0ed82a24c2b1aef4070`
- MultiPosture raw SHA-256: `d21579200db9a57dd01c24fa6ece961b0be73aa0d5d402ae8e629761c03ebe38`
- ANYA raw CSV hashes rechecked against the existing manifest; unchanged.

## Explicit blockers

1. No human-labelled `data/controlled_cctv_v2` rows; only the empty V2 header exists.
2. No trusted RGB/MediaPipe sit-stand corpus with participant/session grouping.
3. No sitting correct/incorrect corpus with defensible labels.
4. No local YOLO dependency/weights; disk headroom is too low for a safe install.
5. No real-room camera placement or multi-person ground-truth evaluation yet.
