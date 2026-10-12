# ANYA / SitDee — Continuation Guide

เอกสารนี้เขียนสำหรับนักพัฒนาหรือ AI agent ที่ไม่เคยเห็นโปรเจกต์นี้ อ่านไฟล์นี้แล้วควรทำงานต่อได้โดยไม่ต้องย้อนอ่านบทสนทนาเก่า

## 1. Start Here
```powershell
Set-Location 'C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets'
git status
git branch --show-current
git log -5 --oneline
git worktree list        # งานล่าสุดอยู่ใน worktree/branch อื่น (ดู §5)
```
- **Environment:**
  - `.venv` ใช้กับงานทั่วไปและ test
  - `.venv_yolo` ใช้กับ PyTorch CUDA + Ultralytics (ตัวรันเพิ่ม site-packages ของ `.venv_yolo` เข้า `sys.path` เอง)
- **Test ที่ปลอดภัย** (ไม่เปิดกล้อง ไม่ฝึกโมเดล):
```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_pose_v2_stage.py tests\test_autonomous_long_session.py tests\test_yolo_tracking_adapter.py tests\test_anya_live_demo_fiveclass.py tests\test_webcam_human_gt_v1.py tests\test_video_source.py
```
- **อ่านก่อน:**
  1. [PROJECT_STATUS.md](PROJECT_STATUS.md)
  2. [DATASETS.md](DATASETS.md)
  3. [SECURITY.md](SECURITY.md)
  4. `reports/ANYA_CURRENT_STATE_AUDIT.md` (ถ้ามีในเครื่อง)
  5. `reports/webcam_p001_fiveclass_exp_v1.md`
  6. `reports/upper_body_posture_v2.md`

## 2. Current Architecture
`VideoSource (file/USB/RTSP)` → `YOLO26n + ByteTrack` → `CandidateSelector / target selection` → `overlap + padded-crop safety` → `PoseV2Stage (MediaPipe crop, POSE_* states)` → `posture classifier (experimental)` → `temporal/UI` → (dashboard)

| ส่วน | สถานะ |
|---|---|
| input → Pose V2 | ใช้งานได้จริง (active) |
| posture classifier | experimental ต้องสั่งเลือกเอง |
| dashboard | อยู่บน branch dash4 |
| MQTT, ESP32-C3, Raspberry Pi | **ยังไม่ได้สร้าง (planned)** |

โค้ดหลัก:
- `src/posture_ai/inference/` (`video_source.py`, `pose_v2_stage.py`, `webcam_fiveclass_exp.py`)
- `src/posture_ai/multiperson/`
- `src/posture_ai/monitoring/`
- `src/posture_ai/features/upper_body_posture_v2.py`

## 3. Critical Design Decisions (ต้องรักษาไว้)
- ใช้แกน AI แบบ modular **ชุดเดียว** สำหรับทั้ง Personal และ Multi-person mode
- รองรับอินพุตแบบ webcam, RTSP และไฟล์วิดีโอผ่าน API เดียวกัน
- **ไม่ใช้ face recognition**; track ID เป็นแบบชั่วคราว **ไม่ใช่ตัวตน**
- **Human GT ≠ AI prediction** ต้องแยก field `label_source` เสมอ
- ถ้าไม่แน่ใจให้ตอบ UNKNOWN / UNCERTAIN ห้ามบังคับเดา
- สถานะรายบุคคลห้ามย้ายไปอยู่กับ track อื่นโดยไม่รู้ตัว (binding จบเมื่อ target หาย, ถูกเปลี่ยน หรือกล้องค้าง)
- ปิดการบันทึกวิดีโอดิบเป็นค่าเริ่มต้น (dashboard บันทึกเมื่อมีความยินยอมเท่านั้น)
- dataset, feature schema และโมเดล ต้องมีเวอร์ชันกำกับ
- **V4 frozen** (`af61482e…`)

## 4. DO NOT BREAK THESE
- ห้ามเขียนทับหรือ retrain V4 และห้ามเปลี่ยน threshold ของโมเดลที่ใช้งานอยู่โดยไม่มีหลักฐาน
- ห้ามฝึกโมเดลด้วย `Predicted_Label` / `clusterid` / `normclusterid` หรือป้ายจาก AI หรือการกดสด ในฐานะ GT
- ห้ามแบ่ง train/test ปนกันระหว่างคนหรือ session (ห้ามสุ่มเฟรม ให้แบ่งตามคนหรือ session)
- ห้ามใช้ชุดทดสอบที่เคยถูกเปิดดูแล้ว (exposed holdout) เป็นชุดทดสอบสุดท้ายซ้ำ
- ห้ามเปิดเผย `.env` / รหัสกล้อง / RTSP URL ที่มี credential
- ห้ามย้ายประวัติท่าทางของคนหนึ่งไปให้อีกคนโดยไม่รู้ตัว
- ห้ามอ้างว่า POSE_OK = จำแนกท่าถูก, ห้ามอ้างว่า confidence = accuracy
- ห้ามถือว่าความแม่นยำบน webcam = บน CCTV หรือผลบน PC = บน Raspberry Pi
- ห้ามอ้างว่าเป็นเครื่องมือทางการแพทย์
- ห้ามใช้ข้อมูลผู้เข้าร่วมที่ไม่มีความยินยอมตามวัตถุประสงค์ (training / evaluation แยกกัน)

## 5. Latest Verified Milestones
**master**

| Commit | งาน |
|---|---|
| `ce03f90f` | experimental five-class webcam demo |
| `7dece91f` | target selection feedback |
| `efb53a33` / `f872b084` | blind webcam Human GT collector |
| `b11f9977` | unified file / USB / RTSP input |

**Branch นอก master** (อยู่ในเครื่องหลัก ยังไม่ merge และยังไม่ผ่าน review อิสระ)

| Branch @ commit | งาน |
|---|---|
| `claude/anya-dashboard-v2-integrity-r2` @ `92698439` | dashboard ที่ใช้งานจริงบนเครื่องหลัก |

รายละเอียดของ branch นี้:
- authority แบบ HMAC hash-chain และ signed ledger head (`3ce08aef`)
- binding state machine และ exact export loaders (`3ce08aef`)
- camera loop ที่ทนต่อ error (`129fd3e2`)
- supervisors ผ่าน Task Scheduler (`92698439`)
- test 105 passed

| Branch @ commit | งาน |
|---|---|
| `claude/anya-ml-v2` @ `1c91548b` | review app (Tailscale + access code), offline re-analysis, dataset builder ที่มีด่านตรวจ consent/identity, การทดลองโมเดล |
| `codex/model-first-protocol-20260925` @ `3c931247` | แก้ลำดับ binding ใน collection protocol |

ผลการทดลองล่าสุด (รายงาน `deliverables/ANYA_ML/MODEL_INTELLIGENCE_SPRINT_20260925_TH.md` ใน branch ML):

| ชุดทดสอบ | v1 | BFHA31 + logreg | หมายเหตุ |
|---|---|---|---|
| คนใหม่ใน MultiPosture | 0.543 | **0.801** (nested 0.753) | |
| P001 (ข้ามกล้อง) | — | แพ้ feature เดิม (0.51 vs 0.64) | **จึงไม่ deploy** |

## 6. Outstanding Work
- **P0 — correctness**
  - รวม branch dashboard r2 + ML เข้า master หลัง review อิสระ (ต้องให้ผู้ใช้อนุมัติ)
  - ยืนยันว่า smoke test ของ dash4 ครบทั้งสาย: record → finalize → review → export
- **P1 — data / model validation**
  - สร้างชุดประเมิน Human GT จากกล้อง ANYA ที่ผู้เข้าร่วมยินยอมแล้ว (≥3 คน, หลายมุม)
  - ประเมิน v1 เทียบ BFHA31
  - ทำ CCTV identity pilot (A/B crossing)
- **P2 — product integration:** MQTT, temporal alert แบบมี false-alert rate
- **P3 — deployment:** Raspberry Pi benchmark, การรันยาวหลายวัน, ESP32-C3

## 7. NEXT SINGLE MILESTONE — `ANYA_CAMERA_VERIFIED_EVALUATION_V1`
- **Objective:** สร้างชุดประเมิน Human GT ชุดแรกจากกล้อง ANYA ที่ผ่านการตรวจโดยคนและมีความยินยอมจริงสำหรับ EVALUATION แล้วประเมิน frozen v1 เทียบ BFHA31 แบบแยกผู้เข้าร่วม
- **Why now:**
  - ทุกการปรับปรุงโมเดลตอนนี้ยืนยันได้แค่บนข้อมูลภายนอก และแพ้เมื่อข้ามกล้อง
  - ไม่มีข้อมูล LEVEL C จึงตัดสินใจเรื่องโมเดลไม่ได้
- **Input required (จากมนุษย์):**
  - review session `Sun1` ใน dash4 ให้เป็น VERIFIED
  - บันทึกความยินยอมจริงของ R002 / R003 และอนุมัติขั้นตอนกู้ข้อมูลเก่าใน review app
  - อัดเพิ่มตาม protocol จนมี ≥3 คน
- **Files likely involved** (อยู่ใน branch `claude/anya-ml-v2` และ dash4 **ยังไม่อยู่บน master**):
  - `scripts/ml/evaluate_on_reviewed.py`
  - `scripts/ml/build_verified_dataset.py`
  - `src/posture_ai/ml_review/store.py`
  - `sessions.eligible_observations` / `load_evaluation_observations` ของ dash4
- **Pass conditions:**
  - ≥3 ผู้เข้าร่วมอิสระที่ผ่าน export gate สำหรับ EVALUATION
  - รายงานผลรายคนพร้อม confusion matrix และ coverage
  - ไม่มี leakage
  - ตัดสินตามเกณฑ์ที่ลงทะเบียนไว้ล่วงหน้า
- **Out of scope:** retrain / deploy โมเดลใหม่, แก้ dashboard, MQTT/ESP32/Pi

## 8. How to Work on This Project
`Planner/Gatekeeper → primary implementation → focused tests → independent review → evidence gate → next milestone`
- ทำ **1 milestone ต่อรอบ**
- ทำงานใน branch / worktree แยก ห้าม merge master โดยไม่ได้รับอนุมัติ
- รายงานผลพร้อมป้ายระดับหลักฐาน (VERIFIED / HYPOTHESIS / NOT VERIFIED)

## 9. Definition of Ground Truth
**Human GT** คือป้ายที่ต้องผ่านเงื่อนไขทั้งหมดนี้:
- มนุษย์ตัดสินจากการดูวิดีโอหรือภาพจริง
- มีการยืนยันตัวผู้เข้าร่วมในช่วงนั้น
- ผู้เข้าร่วมยินยอมตามวัตถุประสงค์การใช้
- มี provenance (session, frame, เวลา, ผู้ตรวจ) และผ่านด่าน export

สิ่งที่**ไม่ใช่** GT:
- source annotation ของ dataset ภายนอก (ใช้ได้ภายใต้ license ของมัน แต่ต้องระบุว่าเป็น source label)
- pseudo-label / cluster
- AI prediction
- ป้ายจากการกดสดก่อน review

## 10. Handoff Checklist (ทำก่อนแก้โค้ด)
- [ ] `git status` สะอาด หรือรู้ว่ามีการแก้ไขของผู้ใช้ค้างอยู่ตรงไหน (ห้าม stage)
- [ ] `git worktree list` และรู้ว่า branch ไหนคือ dashboard ที่ใช้งานจริง
- [ ] `sha256sum outputs/models/rgb_native_v4/model.joblib` = `af61482e…`
- [ ] รัน test ที่ปลอดภัยใน §1 แล้วผ่าน
- [ ] ไม่แตะ `.env`, `datasets/`, `outputs/` (นอกจากไฟล์ที่อนุญาต)
- [ ] ระบุ milestone เดียวที่จะทำ และเงื่อนไขผ่าน ก่อนเริ่ม
