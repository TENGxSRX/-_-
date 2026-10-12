# ANYA / SitDee — Current Project Status

## Snapshot
| รายการ | ค่า |
|---|---|
| วันที่ | 2026-09-30 |
| Repository | `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets` |
| Branch | `master` (ดู HEAD ด้วย `git log -1`; ก่อน commit handoff อยู่ที่ `ce03f90f`) |
| สถานะ test | focused subset 6 ไฟล์ **52 passed** (2026-09-30) ยังไม่ได้รันทั้ง suite บน master (ต้องใช้ dataset ในเครื่อง) |
| Environment หลัก | Windows 11, i3-10105, 16 GB, GTX 1660 SUPER; `.venv` (Python 3.12.10) + `.venv_yolo` (PyTorch CUDA, Ultralytics) |
| หลักฐานอ้างอิง | `reports/ANYA_CURRENT_STATE_AUDIT.md` (audit 2026-09-23, อยู่ในเครื่อง) และรายงานใน `reports/` |

> **สำคัญ:** งานล่าสุดหลายส่วน**ยังไม่อยู่บน master** แต่อยู่ใน branch/worktree แยก (ดูตารางท้ายไฟล์และ [CONTINUATION.md](CONTINUATION.md) §5)

## Project Goal
- ใช้แกน AI แบบ modular ชุดเดียว รองรับ 2 โหมด
  - **Personal:** webcam, ผู้ใช้ 1 คน
  - **Multi-person:** CCTV/RTSP, ติดตามท่าทางรายคน
- ใช้เพื่อ ergonomic feedback ไม่ใช่การวินิจฉัยทางการแพทย์

## What Works Now
| Component | Status | หลักฐาน / ข้อจำกัด |
|---|---|---|
| Input: file / USB webcam / RTSP | VERIFIED | `video_source.py` มี test; เคยรันกับ USB และ RTSP จริงแล้ว |
| YOLO26n person detection | VERIFIED (PC, CUDA) | ~13–16 FPS ตอนรัน RTSP จริง, ~45 FPS แบบ offline |
| ByteTrack | IMPLEMENTED_NOT_FULLY_VALIDATED | ID เป็นแบบชั่วคราว ยังไม่ได้วัด ID-switch / IDF1 |
| CandidateSelector / target selection | IMPLEMENTED_NOT_FULLY_VALIDATED | เป็นนโยบายลดข้อมูล ไม่ใช่การล็อกตัวตน |
| Overlap / padded-crop safety | IMPLEMENTED_NOT_FULLY_VALIDATED | มี test เชิงเรขาคณิต ตรวจจับคนที่ detector มองไม่เห็นไม่ได้ |
| MediaPipe Pose V2 stage | IMPLEMENTED (telemetry/evidence) | POSE_OK ≠ ท่าทางถูกต้อง, ไม่เรียก V4 |
| Posture models | EXPERIMENTAL | ดู Model Status |
| Temporal / duration logic | IMPLEMENTED_NOT_FULLY_VALIDATED | ยังไม่มี false-alert rate เทียบกับ GT |
| Dashboard (ติดตามสด + บันทึก + review + export) | IMPLEMENTED บน branch `claude/anya-dashboard-v2-integrity-r2` | ใช้งานจริงบนเครื่องหลักผ่าน Task Scheduler ยังไม่ได้ merge เข้า master และยังไม่ผ่านการ review อิสระ |
| Review app สำหรับ session เก่า | IMPLEMENTED บน branch `claude/anya-ml-v2` | ต้องมี access code, เข้าได้เฉพาะผ่าน Tailscale |
| MQTT | NOT_IMPLEMENTED | ไม่พบโค้ด |
| ESP32-C3 alert device | NOT_IMPLEMENTED | ไม่พบโค้ด |
| Raspberry Pi deployment | NOT_IMPLEMENTED / ไม่ได้ทดสอบ | ผลบน PC ใช้แทนผลบน Pi ไม่ได้ |
| Long-session | IMPLEMENTED_NOT_FULLY_VALIDATED | รันได้ 3 ชม. 51 นาที, 1 ชม. และ 10 นาที แต่ source failure สูง และยังไม่มีข้อมูลการรันหลายวัน |

## Model Status
| โมเดล | Class | ข้อมูลฝึก | ข้อควรระวังในการประเมิน | ต่อกับ inference ไหม |
|---|---|---|---|---|
| **RGB-native V4** `outputs/models/rgb_native_v4/model.joblib` (**FROZEN**) | TUP, TLB | DriveNet v6, 3 คน | 2 class เท่านั้น, โดเมนคนขับรถ | demo รุ่นเก่าเท่านั้น (Pose V2 ไม่เรียก V4) |
| MultiPosture V1 / upper-body V2 (research) | 5 ท่า | MultiPosture 13 คน | landmark ไม่ใช่ RGB จริง | ไม่ต่อ |
| upper_body_posture_v2 candidate | 5 ท่า + UNKNOWN | MultiPosture / DriveNet | ทดลอง | ตัวเลือกใน runner (ต้องสั่งเอง) |
| **webcam_p001_fiveclass_exp_v1** (`4fbc8d82…`, ไม่อยู่ใน Git) | 5 ท่า + UNKNOWN (threshold 0.5) | P001 คนเดียว, 257 obs | holdout คนเดียวกัน 0.890 แต่คนใหม่ใน MultiPosture ~0.54 | ต้องสั่งด้วย `--posture-model webcam-p001-fiveclass-exp-v1` และ dashboard ใช้ตัวนี้ |
| RGB-native V5 | — | — | NOT_TRAINED | ไม่ต่อ |
| Sprint candidate BFHA31 + logreg (branch ML) | 5 ท่า | MultiPosture 13 คน | คนใหม่ใน MultiPosture 0.80 (nested 0.75) แต่ **แพ้ของเดิมเมื่อข้ามกล้อง (P001)** | ไม่ deploy |

**V4 SHA-256 (ตรวจแล้ว 2026-09-30):** `af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd`

## Dataset Status (สรุป — รายละเอียดใน [DATASETS.md](DATASETS.md))
| สถานะ | Dataset |
|---|---|
| USED_FOR_TRAINING | MultiPosture, DriveNet v6 (V4), Webcam P001 Human GT v1 |
| USED_FOR_VALIDATION | P001 holdout / multiview (exposed), DriveNet participant 3 |
| USED_FOR_RUNTIME_TEST | SitStraightNet, CCTV learning pool, Pose V2 telemetry |
| REFERENCE_ONLY | SitPose, MMD-MSD subset, MOPED25, MSR DailyActivity3D, CCTV Exam Monitor |
| PSEUDO_LABEL_ONLY | ANYA Kaggle (`Predicted_Label`, `clusterid`, `normclusterid` **ไม่ใช่ Human GT**) |
| NOT_READY_FOR_TRAINING | Controlled CCTV v2 (**0 accepted Human GT rows**), Local RGB v1 (ไม่มีวิดีโอ), dashboard sessions ที่ยังไม่ได้ review หรือไม่มีความยินยอม |

**ช่องว่างของ Human GT:** CCTV five-class GT = 0 และคนใหม่จากกล้อง ANYA ที่ตรวจแล้วและมีความยินยอม = 0

## Tests / Benchmarks (ตามหลักฐานที่มี)
- **2026-09-30 บน master:** `pytest` 6 ไฟล์ (pose_v2_stage, autonomous_long_session, yolo_tracking_adapter, anya_live_demo_fiveclass, webcam_human_gt_v1, video_source) → 52 passed
- **Branch dashboard r2:** integrity / dashboard / video test 105 passed (ดู `deliverables/ANYA_DASHBOARD/CODEX_HANDOFF_V2_INTEGRITY_R2.md` ใน branch นั้น)
- **Performance:** ดู `reports/offline_pose_stage_benchmark.md` และส่วน PERFORMANCE ใน audit ยังไม่มี benchmark บน Raspberry Pi

## Current Major Gaps (เรียงตามลำดับที่ต้องทำก่อน)
1. **ไม่มี Human GT จากคนใหม่ด้วยกล้อง ANYA:** เป็นสาเหตุที่ยัง validate โมเดลใดไม่ได้
2. **ความต่อเนื่องของตัวตนในโหมดหลายคน:** ยังไม่ได้วัด
3. **โมเดลไม่ generalize ข้ามคนและข้ามกล้อง:** โดยเฉพาะ TUP / TLF / TLB
4. **Branch ยังไม่ได้รวมเข้า master:** dashboard และ ML อยู่นอก master และยังไม่ผ่าน review อิสระ
5. **ยังไม่ได้ทำ:** MQTT, ESP32, Raspberry Pi, และการรันยาวหลายวัน

## Readiness
**Research prototype / controlled supervised pilot** เท่านั้น **ยังไม่พร้อมใช้งานจริง (not production-ready)**

## Branch / worktree ที่มีงานล่าสุด (local เท่านั้น)
| Branch | Worktree | เนื้อหา |
|---|---|---|
| `claude/anya-dashboard-v2-integrity-r2` | `office-posture-claude-dash4` | dashboard ที่ใช้งานจริง: authority ลงลายเซ็น, binding state machine, exact export loaders, runtime supervisors (`tools/runtime/`) |
| `claude/anya-ml-v2` | `office-posture-claude-ml` | review app, offline re-analysis, dataset builder, การทดลอง model sprint |
| `claude/anya-dashboard-v2-fixes`, `...-integrity` | dash2, dash3 | dashboard รุ่นก่อนหน้า (dash2 = rollback) |
| `claude/multiview-gt-schema` | `office-posture-claude-mv` | multiview Human GT collector |
| `codex/model-first-protocol-20260925` | `office-posture-codex-model-first` | ต่อจาก `claude/anya-ml-v2` (`2ff6fac4`) และเพิ่ม `3c931247` ที่แก้ลำดับ target binding ใน collection protocol |
