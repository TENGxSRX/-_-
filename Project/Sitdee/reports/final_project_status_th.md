# สถานะโครงการฉบับสุดท้าย

## อัปเดต person-centric monitoring — 10 กันยายน 2026

**PERSON_CENTRIC_PIPELINE_READY** — generic multi-person tracking เป็นโหมดหลัก
แล้ว ส่วน seat-aware ยังคงเป็น optional mode. ทุก `PERSON_XX` มี activity,
V4/Geometry, temporal และ lifecycle state แยกกัน

- `SITTING` → เปิด frozen V4 (`TUP/TLB`) และ Geometry indicators
- `STANDING` → `POSTURE MONITOR PAUSED`; ไม่จัดเป็น TUP/TLB
- `UNKNOWN` → `ACTIVITY UNCERTAIN`; ไม่สะสม sitting-risk duration
- activity gate เป็น `GEOMETRY_EXPERIMENTAL`, ไม่ใช่ Sit/Stand AI
- Sit/Stand AI ยัง `NOT_TRAINED`; controlled activity collection ถูกเตรียมไว้ แต่ยังไม่มีการเก็บข้อมูลอัตโนมัติ
- lifecycle events รองรับ appeared, active, temporary lost, reacquired และ exited
- generic ID slot ถูกจำกัดตาม `max_people`; reuse ได้เมื่อ track หมดอายุ และไม่ใช่ persistent real-world identity
- test suite รอบนี้ผ่าน `168 passed`

Short RTSP test ล่าสุด (`HIKVISION_MAIN`, `max_people=4`, 8 วินาที) ตรวจพบ pose ทุกเฟรมที่ infer, core-valid ทุก pose, detected people เฉลี่ย `2.45`, สูงสุด `3`, active tracks สูงสุด `4`, source `43.15 FPS`, inference `3.39 FPS`, decode/reconnect `0/0`. Activity gate ให้ `UNKNOWN` ทุก observation เพราะยังไม่มี lower-body evidence ที่พอ จึง pause V4 sitting evaluation อย่างปลอดภัย. รายละเอียดอยู่ที่ `reports/person_centric_rtsp_test_th.md`.

รายละเอียด: `reports/person_centric_posture_monitoring_th.md`

## อัปเดต hierarchical CCTV intelligence — 10 กันยายน 2026

สถานะซอฟต์แวร์: **HIERARCHICAL_CCTV_PIPELINE_READY_AS_CONTRACT** — มีเส้นทาง
VideoSource → MediaPipe multi-pose → anonymous tracks → V4/Geometry/Temporal
แยกต่อคน และมี decision gate activity → sitting quality → subtype แล้ว แต่ยัง
ไม่ใช่หลักฐานว่าโมเดลชั้นใหม่ train พร้อมใช้งานจริง

- **YOLO:** optional candidate A/B/C ถูกเตรียมไว้ แต่ยังไม่ติดตั้ง/ไม่ benchmark; venv ไม่มี `torch`/`ultralytics` และ C: เหลือประมาณ 10.91 GiB
- **GPU:** NVIDIA GeForce GTX 1660 SUPER 6 GB ตรวจพบ; CUDA/PyTorch runtime ยังไม่มีใน environment นี้
- **Activity / sit-stand:** `STANDING_DATA_INSUFFICIENT`
- **Sitting quality:** `SITTING_POSTURE_DATA_INSUFFICIENT`
- **Subtype:** ยังไม่ train; Geometry เป็น auxiliary indicator เท่านั้น
- **Canonical schema:** `cctv_canonical_v1`, hip-centered และ optional torso scale, ยังไม่พิสูจน์ viewpoint invariance และไม่ส่งเข้า V4 โดยอัตโนมัติ
- **CCTV V5:** ยังไม่ train เพราะยังไม่มี human-labelled CCTV V2 samples
- **V4:** frozen และ hash ยังตรง `af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd`
- **Tests:** `165 passed`; ไม่มีการเปิดกล้อง/RTSP ในรอบเตรียมนี้

ผลนี้ยืนยัน software readiness เท่านั้น ไม่อ้าง person-count accuracy, tracking IDF1,
model accuracy ของ activity/sitting/subtype หรือ real-room production readiness.

อัปเดตล่าสุด 2026-09-09 บนเครื่อง `DESKTOP-ALH6CUB`: เตรียม pipeline CCTV
หลายคนบน RTSP เพิ่มแล้ว โดยยังไม่ train โมเดล ไม่เปิด webcam ไม่แก้ raw data
และไม่เขียนทับ V1/V2/V4 ชุดทดสอบเต็มล่าสุดผ่าน `153 passed` ดูรายละเอียดที่
`reports/controlled_ai_robustness_baseline_th.md`,
`reports/controlled_collection_protocol_th.md` และ
`reports/cctv_multiperson_architecture_audit_th.md` ส่วนรายละเอียดเก่าด้านล่างคงวัน
ตรวจเดิมของแต่ละหัวข้อไว้

## สถานะ CCTV Multi-Person ปัจจุบัน

**CCTV_MULTIPERSON_PIPELINE_READY** — รองรับ `max_people=1..4`, ค่าเริ่มต้น
ของ CCTV คือ 3, ใช้ MediaPipe multi-pose, anonymous `PERSON_XX`, และแยก V4 /
Geometry / Temporal ต่อคนผ่าน `VideoSource` เดิม ระบบทดสอบ RTSP รอบล่าสุด
แบบ max_people=3 ตรวจพบ 1 pose และ core-valid 1 pose แต่ยังไม่มี manual ground truth จึงยังไม่ใช่
การยืนยัน person-count accuracy, IDF1 หรือความพร้อมของห้องจริง

**PENDING_TRUSTED_MULTIPERSON_LABELS** — ยังไม่ train CCTV_V5 หรือโมเดล
hierarchical ใหม่ ตัวเก็บข้อมูลแบบมีผู้ควบคุมพร้อมใช้ที่
`scripts/collect_controlled_cctv.py` และบังคับติดป้ายด้วยคนเท่านั้น

ตรวจสอบวันที่ 2026-09-05 เครื่อง `DESKTOP-ALH6CUB`

## สถานะ Controlled RGB Robustness รอบล่าสุด

**READY_FOR_CONTROLLED_DATA_COLLECTION** — เตรียม AI Lab แบบ single-person
สำหรับ RTSP/file/USB และตัวเก็บข้อมูลที่ให้ผู้ควบคุมกด `1=TUP`, `2=TLB` และ
`SPACE` เพื่อเริ่ม/หยุดการเก็บ accepted frames แล้ว ค่าเริ่มต้นไม่บันทึก raw
video และแยก `human_label` ออกจาก `v4_prediction` อย่างเด็ดขาด

ยังไม่มี sample human-labelled ใหม่ใน `data/controlled_rgb_v5/` จึงยังไม่
ประเมิน V4 บนข้อมูลใหม่และยังไม่ train V5 เครื่องมือประเมิน frozen V4 จะหยุด
เองเมื่อข้อมูลยังไม่พร้อม รายละเอียดอยู่ที่
`reports/controlled_ai_robustness_baseline_th.md` และ
`reports/controlled_collection_protocol_th.md`

## สถานะ RTSP Low-Latency AI Lab ล่าสุด

**RTSP_REAL_BENCHMARK_PASS** — ทดสอบจริงแบบสั้นทั้ง `HIKVISION_MAIN` และ
`HIKVISION_SUB` โดยไม่มี recording/collection/training ระบบ latest-frame มี
buffer เพียง 1 ช่องและทิ้งเฟรมเก่าที่รอไม่ทัน อายุเฟรมอยู่ราว 0–16 ms ตามรอบวัด,
decode failure 0 และ reconnect 0. รอบเปิดหน้าจอจริง MAIN ได้ AI/display `8.73 FPS`,
median render `20.9 ms` หลังย่อ display เป็น 1280×720

รอบล่าสุด MAIN วัดได้ 2560×1440, source 20.39 FPS, AI 14.12 FPS, median inference
23.9 ms และ core acceptance 31.7% ส่วน SUB วัดได้ 640×360, source 20.18 FPS,
AI 14.57 FPS, median inference 62.8 ms และ core acceptance 56.6% ค่า pose/core
ต่ำเป็นผลจากช่วงภาพที่วัดและไม่ใช่ accuracy claim จึงต้องทดสอบซ้ำขณะมีคนจริง
ในท่านั่ง

ตั้ง `RTSP_LOW_LATENCY` และ inference copy MAIN 960×540 เป็นค่าเริ่มต้นแบบ
อนุรักษ์นิยม รายละเอียดอยู่ที่ `reports/rtsp_ai_lab_low_latency_v2_th.md`

## ดาวน์โหลดและ audit ANYA

ติดตั้ง Kaggle `2.2.4` ใน `.venv` และตรวจสอบ authentication สำเร็จ ดาวน์โหลด archive จาก Kaggle ทางการ version 1 และเก็บ archive ไว้แล้ว ไฟล์ CSV เดิมทั้ง 4 ชื่ออยู่ที่ `datasets/raw/anya/official-download/` ขนาดรวม `7,966,483,972` bytes และ ZIP ที่เก็บไว้มีขนาด `3,607,095,951` bytes แฮชอยู่ใน `metadata/anya_hashes.sha256` และ `metadata/anya_archive.sha256`

ตรวจแบบ streaming ครบทั้ง 4 ไฟล์แล้ว ทุกไฟล์มี 605,955 แถว 176 คอลัมน์ และ core row sequence เหมือนกัน ต่างกันที่ค่า `Predicted_Label` ซึ่งเป็นผลพยากรณ์ของแต่ละโมเดล ไม่ใช่ ground truth ที่ยืนยันได้ อีกทั้งไม่พบคอลัมน์ TRUE/FALSE หรือ target ที่ตรวจสอบได้ จึงยังฝึก ergonomic classifier อย่างถูกต้องไม่ได้

การสืบค้นต่อจาก paper และแหล่งทางการจัดให้ release ปัจจุบันเป็น **B — PSEUDO_LABEL_ONLY** paper รายงานว่ามี expert-labeled subset 144,000 frames จาก recording 10 ชุดแรก แต่ Kaggle version 1, ZIP, metadata ในเครื่อง และ repository ของผู้เขียนที่ตรวจแล้ว ไม่มีไฟล์ subset หรือ mapping จาก cluster ไป TRUE/FALSE ที่ join กับแถวในเครื่องได้ อีกทั้งจำนวน 432,000 entries ที่ paper รายงานไม่ตรงกับ 605,955 แถวต่อไฟล์ที่วัดได้จริง รายละเอียดอยู่ใน `reports/anya_ground_truth_investigation.md`

ผมเปรียบเทียบ prediction ทั้ง 4 คอลัมน์โดยไม่ฝึกโมเดล พบตรงกันทุกโมเดล 459,841 แถว (75.887%), ไม่ตรงกัน 146,114 แถว (24.113%) และเป็น tie แบบ 2 ต่อ 2 จำนวน 55,135 แถว (9.099%) ตัวเลขนี้เป็นเพียง pseudo-label agreement diagnostic ไม่ใช่ความถูกต้องเทียบ ground truth ดู `reports/anya_pseudolabel_consensus.md`

## สถานะการฝึกโมเดล

**MULTIPOSTURE MODEL: READY** — HistGradientBoosting, hybrid 126 features, grouped-CV macro F1 `0.9891 ± 0.0060`, validation macro F1 `0.9910` และ final subject-held-out test macro F1 `1.0000` ผล test มีผู้เข้าร่วมเพียง 2 คน จึงต้องตีความอย่างระมัดระวัง

**ANYA MODEL: NOT READY — TRAINING BLOCKED** — release ปัจจุบันเป็น `B — PSEUDO_LABEL_ONLY` มี cluster IDs และผล prediction จากหลาย algorithm แต่ไม่มี target supervised ที่ยืนยันได้ จึงไม่ได้สร้าง `outputs/models/anya_v1/` และไม่ได้สร้าง metric ขึ้นเอง

## สถานะการประเมิน

MultiPosture ยังคงประเมินแบบแยก subject: train 9 คน, validation 2 คน, final test 2 คน และ GroupKFold 5 folds ภายใน train ส่วน ANYA ทำ schema audit, quality audit, เปรียบเทียบ 4 ไฟล์, ตรวจ grouping และความเสี่ยง temporal leakage เสร็จแล้ว แต่ยังไม่ทำ model metrics/calibration/error analysis เพราะไม่มี target ที่ยืนยันได้

## สถานะ pipeline ภาพ/วิดีโอออฟไลน์

**OFFLINE IMAGE/VIDEO PIPELINE: READY AND TESTED** — รองรับ image file, image directory และ prerecorded video มี canonical landmarks, geometry indicators, temporal risk states, event log, session summary และ annotated video เพิ่มจาก MediaPipe Pose, visibility rejection, strict feature-order และ confidence ไม่ได้เรียก webcam และใช้ SitStraightNet เฉพาะ smoke/preprocessing test โดยไม่ใช้ label เป็น ground truth

**ANYA REAL-IMAGE ADAPTER: NOT VALIDATED** — ยังพิสูจน์ไม่ได้ว่า index/order ของ landmark, anatomy mapping และ normalization ของ ANYA เหมือนกับ MediaPipe/MultiPosture ดังนั้นห้ามต่อ adapter เข้ากับ ANYA โดยเดาตามตำแหน่ง

## สถานะสื่อ RGB ภายนอก

**EXTERNAL RGB MEDIA SEARCH: COMPLETED** — ดาวน์โหลด DriveNet v6 จาก Mendeley Data ทางการไว้ที่ `validation_data/external/` จำนวน 3 ไฟล์ MP4 RGB ขนาดภาพ 1920×1440 รวม `387,761,277` bytes โดยเก็บไฟล์ต้นฉบับไม่แก้ไข `SS_1.mp4` map เป็น `TUP`, `LB_2.mp4` map เป็น `TLB` และ `MB_1.mp4` เก็บเป็น `UNLABELED` hash ตรงกับ metadata และ license คือ CC BY 4.0

**EXTERNAL OFFLINE RGB SMOKE: COMPLETED** — MediaPipe ตรวจพบ pose `900/900` เฟรมที่ประมวลผล (คลิปละ 300 เฟรม) แต่ strict 33-landmark adapter รับได้ `0/900` เพราะ landmark ขา/เท้าและแขนบางส่วน visibility ต่ำกว่า 0.5 จากมุมกล้องด้านข้าง/เฉียง จึงไม่รายงาน accuracy หรือ confidence ของ classifier ผล annotated และ telemetry อยู่ที่ `outputs/validation/external/` พร้อม manifest/audit ที่ `validation_data/external/`

**EXTERNAL RGB DOMAIN COMPATIBILITY: NOT PROVEN** — ชื่อและลำดับ 126 features ตรงกันในเชิงกลไก แต่ coordinate/projection และ visibility ของ MediaPipe จากภาพภายนอกยังไม่เทียบเท่ากับ representation ที่ใช้ฝึก MultiPosture ดู `reports/external_rgb_domain_shift.md`

**ADAPTER ROBUSTNESS + UPPER-BODY V2: COMPLETED** — ทดลอง 4 policy กับ DriveNet 900 เฟรมชุดเดิม พบว่า strict full-body รับ `0/900` แต่ core-only, tiered และ quality รับ `900/900` โมเดล V2 แยกจาก V1 ใช้ core 5 จุด (`nose`, ไหล่สองข้าง, สะโพกสองข้าง), hybrid 30 features และเลือก HistGradientBoosting ได้ grouped-CV macro F1 `0.9925 ± 0.0087`, subject-held-out macro F1 `1.0000` ตัวเลขนี้เป็นผลจาก source เดียวกัน ไม่ใช่ real-camera generalization V2 adapter รับ DriveNet `900/900` แต่ยังไม่ถือว่า semantic validation ผ่าน เพราะคลิปที่ map label ได้มี aggregate macro F1 `0.0000` ซึ่งชี้ว่า coordinate/label semantics ยังไม่ตรงกัน รายละเอียดอยู่ใน `reports/adapter_robustness_v2_th.md`, `reports/landmark_visibility_analysis.md`, `reports/domain_shift_root_cause.md` และ `reports/multiposture_landmark_mapping_verified.md`

**COORDINATE ADAPTER V3: PARTIALLY VALIDATED** — forensic จาก source สนับสนุนว่าใช้ MediaPipe Pose Heavy World Landmarks หน่วยเมตร และ origin ที่กึ่งกลางสะโพก แต่ release ที่วัดได้จริงมี 99 coordinate columns และไม่มี StandardScaler ของ model ใน paper ให้ตรวจสอบ Hip midpoint residual ใกล้ศูนย์ทั้งใน CSV และ world output ของ MediaPipe และยังไม่พบหลักฐานการ scale ทางเรขาคณิต Candidate ที่มีหลักฐานรองรับที่สุด `adapter_v3_a_world_hip_centered` ลด V2 features ที่ OOD เกินครึ่งจาก `6/30` เหลือ `5/30` แต่ทุก external row ยังมี OOD อย่างน้อยหนึ่ง feature และ SS_1 ยังถูกทำนายเป็น `TLL` `299/300` เฟรม จึงยังไม่ validate semantic compatibility ของ RGB จริงเต็มรูปแบบ รายละเอียดอยู่ใน `reports/coordinate_adapter_v3.md`, `reports/multiposture_coordinate_forensics.md`, `reports/adapter_v3_distribution_comparison.md` และ `reports/ss_tup_to_tll_root_cause.md`

## สถานะ ERGONOMIC POSTURE ENGINE V1

- **MULTIPOSTURE DIRECTION MODEL: READY** — รองรับ Upright, Lean Forward,
  Lean Backward, Lean Left และ Lean Right
- **MULTIPOSTURE FEATURE ADAPTER: READY ด้าน schema/order; REAL-IMAGE
  SEMANTIC COMPATIBILITY NOT VALIDATED** — 126 features ตรงกับโมเดล แต่ยัง
  พิสูจน์ความเทียบเท่าของ coordinate scale จากภาพจริงไม่ได้
- **GEOMETRY ENGINE: READY** เป็น deterministic indicators; head-forward,
  shoulder-tilt และ slouch ยังเป็น **EXPERIMENTAL**
- **TEMPORAL RISK ENGINE: READY** มี NORMAL, OBSERVE, WARNING,
  HIGH_DURATION_RISK และ POSE_LOST
- **OFFLINE IMAGE PIPELINE: READY** ด้านการประมวลผลไฟล์ แต่ smoke test ภาพ
  SitStraightNet 756 ภาพตรวจ pose ได้ 0 ภาพ
- **OFFLINE VIDEO PIPELINE: READY** สร้าง annotated MP4, event log และ session
  report จาก video smoke test 2 frames ได้
- **EXTERNAL RGB VALIDATION MEDIA: READY FOR POSE SMOKE ONLY** — ตรวจวิดีโอ
  DriveNet แบบ offline พร้อม manifest และ hash แล้ว แต่ adapter gate พบข้อจำกัด
  เรื่อง visibility จากมุมกล้องด้านข้าง จึงยังไม่รายงาน accuracy
- **UPPER-BODY V2: EXPERIMENTAL READY FOR CONTROLLED STUDY** — export อยู่ที่
  `outputs/models/multiposture_upperbody_v2/` และไม่แก้ V1
- **YOLO POSE: NOT YET BENCHMARKED** — มีเพียง interface

ค่าการเตือน 15 วินาทีและ high-duration 60 วินาทีเป็น project settings ไม่ใช่
clinical/medical thresholds ดูรายละเอียดใน `reports/posture_engine_v1_design.md`

## สถานะแทร็ก RGB-native V4 (2026-09-05)

**RGB-NATIVE V4: PILOT TRAINED — รองรับ 2 คลาสที่ mapping ได้เท่านั้น** ดาวน์โหลดวิดีโอ DriveNet v6 ทางการแบบรายไฟล์ครบ 6 คลิป (`LB_1..3` และ `SS_1..3`) ตรวจ SHA-256 ตรงกับ metadata รวม `907,778,242` bytes ใช้ตัวอย่างแบบ deterministic คลิปละ 300 เฟรม รวม 1,800 แถว และทุกแถวตรวจ pose/core landmark ผ่าน

V4 ใช้ pipeline เดียวกันตอนฝึกและอนุมานออฟไลน์: MediaPipe World Landmarks → hip-midpoint translation → V4 features → classifier โดยเลือก Logistic Regression กับ normalized-world 15 features ได้ grouped-CV macro F1 `1.0000 ± 0.0000` และ holdout ผู้เข้าร่วมคนที่ 3 macro F1 `1.0000` ผลนี้เป็น pilot สองคลาสในโดเมนวิดีโอขับรถ ไม่ใช่ผล six-class หรือ office-camera generalization โมเดลอยู่ที่ `outputs/models/rgb_native_v4/`

mapping ที่ใช้ฝึกมีเพียง `SS → TUP` และ `LB → TLB`; `MB` และ `RP` ไม่ถูก map เพราะชื่อแหล่งข้อมูลยังไม่พิสูจน์ว่าเป็น sustained posture จึงไม่เดา label ส่วน CCTV Exam Monitor และ SitStraightNet ยังใช้ได้แค่ audit/smoke เนื่องจาก snapshot ที่ตรวจไม่มี sidecar label รายภาพที่ join ได้ และ MVSP/LGCSPNet ยังไม่มี archive/license ที่ตรวจยืนยันในเครื่อง

มี offline smoke inference จากไฟล์จริงแล้ว และ frozen re-test ถูกระบุเป็น same-source overlap diagnostic ไม่ใช่ external test อิสระ V1/V2/V3, ANYA และ raw data ไม่ถูกแก้ไข ไม่ติดตั้ง YOLO และมี session webcam จริงที่บันทึกแยกไว้เพื่อวิเคราะห์ calibration

## สถานะเว็บแคม

**LIVE WEBCAM VALIDATION: REAL SESSION COMPLETED — CALIBRATION NEEDED**

ทำ user-guided session จริงกับ Camera 0 แล้ว session `20260907T155515Z` มี 1,249 records นาน 103.617 วินาที มี telemetry จริง แต่ยังไม่ใช่ผลรับรองความแม่นยำของ posture รายละเอียดอยู่ที่ `reports/live_hardware_calibration_analysis_th.md`

**LIVE WEBCAM SOFTWARE: READY_FOR_HARDWARE_TEST** — เตรียม `check_camera.bat`
สำหรับตรวจเฉพาะกล้องหมายเลข 0, 1, 2 เมื่อผู้ใช้สั่ง และ
`run_webcam_demo.bat` สำหรับ validation/calibration แบบมีคำแนะนำ ระบบใช้ V4
สองคลาสเดิม แยก geometry indicators ออกจาก AI posture ใช้ค่าจาก
`configs/posture_engine.yaml` และบันทึก telemetry/session summary ใน
`outputs/live/` ออกจากโปรแกรมด้วย Q หรือ ESC ได้อย่างปลอดภัย

**LIVE WEBCAM HARDWARE: PASS** — Camera 0 เปิด อ่านภาพ และ cleanup สำเร็จ
MediaPipe detection `88.86%`, five-core acceptance `66.30%`, mean FPS `10.95`
แต่ core quality และ geometry ยังเป็น `NEEDS_CALIBRATION`, V4 TUP/TLB ยังเป็น
`NEEDS_VALIDATION` และ temporal warning ยังเป็น `NEEDS_RETEST`

ชุดทดสอบล่าสุด รวม live-prototype, Acer lightweight, room planner,
multi-person และ dashboard tests ที่ไม่ใช้ฮาร์ดแวร์ ผ่าน `100/100`

## ความถูกต้องของข้อมูลและ tests

แฮช ANYA ยังตรงกับ manifest ก่อน audit และแฮช raw MultiPosture ยังไม่เปลี่ยน ไฟล์ raw ใน `datasets/raw/` ไม่ได้ถูกเขียนทับหรือลบ และ V1/V2/V4 ยังแยกจากกัน ชุดทดสอบปัจจุบันผ่าน `144 passed` ดู hash snapshot ล่าสุดแบบอ่านอย่างเดียวที่ `reports/overnight_system_prep_integrity.md`

## blocker และการทดลองถัดไป

ต้องได้ source-backed expert label mapping หรือ ground-truth sidecar ที่ join กับแถว/cluster ชุดนี้ได้ ผมเตรียมร่างคำขอภาษาอังกฤษไว้ใน `reports/anya_author_request.md` แล้วและยังไม่ได้ส่ง สำหรับ V1/V2 ให้ทดสอบกล้องจริงซ้ำด้วย neutral calibration ใหม่, framing ที่เผื่อสะโพก, stable-ready gate และ step-8 reacquisition และต้องแก้ความไม่แน่ชัดของ release 99 columns เทียบกับ paper 33 inputs/StandardScaler ก่อน retrain หลักฐานกล้องจริงรอบนี้เหมาะกับการวิเคราะห์ calibration แต่ยังไม่พอยืนยัน classifier จากภาพจริง YOLO ยังไม่จำเป็น เพราะ blocker หลักคือคุณภาพ input และ coordinate semantics ไม่ใช่การเปิดกล้อง

## การขยายชุดข้อมูล RGB-native V5 (2026-09-05)

**V5: ยังไม่ฝึกโมเดล — ติด data-readiness gate อย่างถูกต้อง** ขณะนี้มีข้อมูลใน
manifest ที่ตรวจย้อนกลับได้ 452 แถว: DriveNet 417 แถวจากการสุ่ม 1 fps ของ 6 คลิป
และ MMD-MSD 35 ภาพจาก public subset ทางการ MediaPipe World Landmark และฟีเจอร์
ตามสคีมา V4 เดิมสร้างได้ครบทั้ง 452 แถว ป้ายกำกับที่ใช้ได้คือ TUP 244 แถวและ
TLB 183 แถว ส่วน 25 แถวของ MMD-MSD เก็บไว้เป็น audit/ambiguous

ยังไม่มีข้อมูลที่มีหลักฐานเพียงพอสำหรับ TLF, TLL และ TLR จึงยังไม่ทำ grouped CV,
holdout, leave-one-dataset-out, model comparison หรือ error analysis และไม่ได้
สร้าง classifier V5 ขึ้นมา V4 ไม่ถูกฝึกซ้ำหรือเขียนทับ hash ของไฟล์ต้นฉบับ 44
ไฟล์อยู่ที่ `outputs/metrics/rgb_native_v5_source_hashes.json` รายละเอียดอยู่ใน
`reports/rgb_native_v5.md`

ชุดทดสอบอัตโนมัติผ่าน `89 passed` ขั้นต่อไปที่ถูกต้องคือหา RGB source ที่มี
license และ traceability ชัดเจนสำหรับ TLF/TLL/TLR (MVSP/LGCSPNet ยังเป็นเพียง
candidate เพราะต้องยืนยัน license/archive) หรือเก็บชุดข้อมูล RGB ในสำนักงานที่
ติดป้ายกำกับเองและแยก participant/session/video

## ระบบเก็บข้อมูล Local RGB V1 (2026-09-05)

สร้างโครงสร้างว่างไว้ที่ `datasets/raw/local_rgb_v1/` แล้ว ประกอบด้วย participant
แบบนิรนาม 10 คน × 2 sessions × 5 โฟลเดอร์ท่า รวม 100 recording slots มี
`participants.csv`, `sessions.csv` และ master manifest แยกที่มี 100 แถว ทุกช่อง
ข้อมูลวิดีโอยังว่างและทุก slot เป็น `WAITING_FOR_VIDEO` ไม่มีวิดีโอจริงหรือข้อมูล
เฟรมปลอมถูกสร้างขึ้น

เครื่องมือพร้อมใช้สำหรับตรวจชื่อไฟล์/โฟลเดอร์, SHA-256, MediaPipe World Landmark
pre-flight, การสกัด feature ตาม V4/V5 เดิม, สรุป dataset และวาง participant split
สถานะปัจจุบันคือ `WAITING_FOR_LOCAL_RGB_DATA` และยังไม่ได้ฝึก V5

## อัปเดต Overnight (2026-09-06)

รอบค้นหาแหล่งข้อมูลภายนอกและ benchmark V4 แบบ frozen เสร็จแล้ว ไม่มีการดาวน์โหลด raw archive ใหญ่ใหม่ วิดีโอ DriveNet ที่ยืนยันไว้ทั้ง 6 คลิปอ่านครบ `12,395` source frames และ dense pass ที่ `10 FPS` ประมวลผล `4,134` เฟรม โดย MediaPipe pose detection และ V4 five-core acceptance ได้ `100%` ทั้งคู่ ตารางย่อย `1/2/5 FPS` ก็ได้ pose/core `100%` และ `POSE_LOST` เป็น `0` ที่ `5 FPS` มี raw class switch รวม `3` ครั้ง และลดเหลือ `1` หลังใช้ temporal smoother ผลทั้งหมดเป็น same-source diagnostic ไม่ใช่ external accuracy อิสระ

Frozen V4 feature audit พบ shift สูงสุดระหว่าง participant 3 กับ train domain ที่ `nose_y_world_hip_centered_scaled` absolute SMD `1.212`, hip-y ทั้งสองค่า `0.932` และ right-shoulder-y `0.925` ส่วนค่าสัมประสิทธิ์สูงสุดของ frozen LogisticRegression อยู่ที่ shoulder-x, nose-x และ shoulder-y จึงยืนยันว่าข้อจำกัดหลักยังเป็น camera/domain semantics แม้ pose detection จะดีมาก

ผลค้นหาชุดข้อมูลคือ **NO_VALID_EXTERNAL_5CLASS_DATA** โดย MVSP/LGCSPNet ยังเป็น manual-access candidate แต่ยังยืนยัน Baidu archive, label join, participant/view metadata และ license ไม่ได้ จึงไม่มีการ bypass หรือดาวน์โหลดอัตโนมัติ V5 และ ANYA ยังไม่ฝึก มีการเตรียมและ smoke-test hybrid V4 + geometry demo แยกต่างหากที่ 60 เฟรม แต่ไม่ใช่โมเดลใหม่และไม่ใช่ clinical score

artifact ใหม่อยู่ใน `reports/overnight_run_summary_th.md`, `reports/overnight_external_dataset_search_th.md`, `reports/overnight_performance_benchmark.md`, `reports/v4_feature_audit.md`, `reports/reproducibility_audit.md`, `reports/overnight_progress.md` และ `outputs/validation/overnight/` test suite ผ่านแล้ว `65 passed` raw data และ V1–V4 ไม่เปลี่ยนแปลง คำแนะนำตอนนี้คือใช้ MediaPipe ต่อ, ยังไม่ใช้ YOLO และหา RGB source ที่มีหลักฐานครบทั้ง 5 คลาส หรือเก็บข้อมูล office RGB จากหลายคนก่อนฝึก V5

## ระบบนำเข้า MVSP แบบ Manual (2026-09-06)

เตรียมโฟลเดอร์วาง archive ไว้ที่ `datasets/incoming/mvsp/` และโฟลเดอร์ปลายทางคือ `datasets/raw/mvsp/` กับ `datasets/processed/mvsp/` ตอนนี้ incoming ยังว่าง จึงมีสถานะ **WAITING_FOR_MANUAL_MVSP_ARCHIVE** โฟลเดอร์เดิม `datasets/raw/mvsp/official-repository/` ถูกเก็บไว้และไม่ได้ถูกแก้หรือลบ

เตรียมสคริปต์แล้ว 4 ตัว: `scripts/inspect_mvsp_archive.py`, `scripts/import_mvsp_dataset.py`, `scripts/audit_mvsp_labels.py` และ `scripts/preflight_mvsp_rgb.py` รองรับ archive `.zip`, `.tar`, `.tgz`, `.tar.gz`, `.tar.bz2`, `.tar.xz` โดย inspector จะคำนวณ hash, ตรวจ path traversal/link และดูรายการไฟล์โดยไม่แตก archive importer จะเก็บ archive ต้นฉบับ ตรวจ collision และแตกเฉพาะไป raw/mvsp ส่วน label audit จะไม่เอาชื่อโฟลเดอร์หรือชื่อไฟล์ไปเป็น label อัตโนมัติ และต้องมี source-backed evidence ก่อนใช้สถานะ `EXACT` หรือ `COMPATIBLE`

รายงานอยู่ที่ `reports/mvsp_archive_inspection.md`, `reports/mvsp_inventory.md` และ `reports/mvsp_inventory_th.md` test suite หลังเพิ่ม import gate ผ่าน `70 passed` รอบนี้ไม่มีการดาวน์โหลดหรือแตก MVSP และไม่มีการฝึก V5
