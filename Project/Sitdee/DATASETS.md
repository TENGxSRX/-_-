# ANYA / SitDee — Datasets

**ไม่มี dataset ใดถูกเก็บใน Git** (`datasets/`, `validation_data/`, `*.csv`, `*.zip` ถูก ignore)
ใน Git มีเฉพาะ manifest, hash และ registry ได้แก่ `data/dataset_registry_v2.json`, `metadata/*.sha256`, `metadata/*_record.json` และ `metadata/source_inventory.md`

## ประเภทของป้าย (ห้ามปนกัน)
| ประเภท | ความหมาย |
|---|---|
| **Human Ground Truth** | มนุษย์ตัดสินจากภาพหรือวิดีโอจริง พร้อมยืนยันตัวผู้เข้าร่วม, ความยินยอม และ provenance |
| **Source annotation** | ป้ายที่ผู้สร้าง dataset ภายนอกให้มา ใช้ได้ตาม license และต้องระบุว่าเป็น source label |
| **Pseudo-label** | ป้ายที่สร้างด้วยอัลกอริทึมหรือ cluster **ไม่ใช่ GT** |
| **AI prediction** | ผลของโมเดล **ไม่ใช่ GT** |
| **Unknown** | ไม่มีป้าย หรือยังตรวจไม่ได้ |

## ตาราง dataset (ตรวจกับเครื่องหลัก DESKTOP-ALH6CUB, 2026-09-30)
| Dataset | Location | มีในเครื่อง? | Type | ผู้เข้าร่วม | Classes | ประเภทป้าย | ใช้ทำ | พร้อมฝึก? | แจกต่อได้? | หมายเหตุ |
|---|---|---|---|---|---|---|---|---|---|---|
| MultiPosture (Zenodo 14230872) | `datasets/raw/multiposture/data.csv` | ✔ | MediaPipe world landmarks 33 จุด (ไม่มีวิดีโอ) | 13 | TUP/TLF/TLB/TLL/TLR + ท่าขา | Source annotation (label ด้วยมือ) | ฝึก / ประเมินแบบแยกคน | ✔ (landmark เท่านั้น) | CC BY 4.0 (ต้องอ้างอิง) | 4,794 แถว; ตรวจแล้วว่า 33 จุดตรงกับ MediaPipe จริง |
| DriveNet v6 | `datasets/raw/rgb_native/drivenet_v6/` | ✔ | RGB video | 3 | SS→TUP, LB→TLB | Source annotation | ฝึก V4 / ตรวจภายนอก | บางส่วน (2 class) | CC BY 4.0 | โดเมนคนขับรถ |
| Webcam P001 Human GT v1 | `datasets/derived/webcam_human_gt_v1/` | ✔ | features + world landmarks (ไม่มีภาพ) | 1 | 5 ท่า | **Human GT** | ฝึก v1 / holdout | ✔ แต่มีแค่คนเดียว | ข้อมูลส่วนบุคคล ห้ามแจก | 257 obs, 30 segments; holdout ถูกใช้ดูผลแล้ว (exposed) |
| Multiview P001 GT v2 | worktree `office-posture-claude-mv` | ✔ | features + landmarks | 1 | TUP | **Human GT** | ตรวจข้ามมุมกล้อง | — | ส่วนบุคคล | 55 obs, exposed |
| ANYA Kaggle v1 | `datasets/raw/anya/official-download/` | ✔ (~11 GB) | CSV features | ไม่ทราบ | — | **Pseudo-label / AI prediction** (`Predicted_Label`, `clusterid`, `normclusterid`) | audit เท่านั้น | ✘ | Apache-2.0 (ตาม metadata) | **ห้ามใช้เป็น GT** |
| SitPose | `datasets/raw/sitpose/` | ✔ | Kinect joints | ไม่ระบุ | 7 ท่าที่ไม่ตรงกับ ANYA | Source annotation | อ้างอิง | ✘ | license ข้อมูลไม่ระบุ | |
| SitStraightNet | `datasets/raw/sitstraightnet/` | ✔ | 756 JPG | ไม่ระบุ | ไม่มีป้าย | Unknown | smoke test | ✘ | repository เป็น MIT (ภาพยังไม่ชัด) | |
| MMD-MSD subset | `datasets/raw/rgb_native_v5/mmd_msd/` | ✔ | 35 JPG + CSV | 5 กลุ่ม | จำกัด | Source (ไม่ครบ) | อ้างอิง | ✘ | ไม่ระบุ | |
| CCTV Exam Monitor | `datasets/raw/rgb_native/cctv_exam_monitor/` | ✔ | 8,156 JPG | ไม่ระบุ | ไม่มี join ป้าย | Unknown | อ้างอิง | ✘ | ไม่ชัด | |
| MVSP / LGCSPNet | `datasets/raw/mvsp/` | บางส่วน (README) | — | — | — | — | — | ✘ | ไม่ระบุ | ต้องเข้า Baidu เอง |
| MOPED25, MSR DailyActivity3D | `data/external/...` | archive | — | — | — | — | อ้างอิง | ✘ | ดู registry | ยังไม่ได้ audit |
| Local RGB v1 | `datasets/raw/local_rgb_v1/` | แค่โครงสร้างโฟลเดอร์ | — | 0/10 | — | — | — | ✘ | — | ไม่มีวิดีโอ |
| Controlled CCTV v2 | `data/controlled_cctv_v2/` | ✔ | telemetry | — | — | ไม่มี | runtime test | ✘ | ส่วนบุคคล | **Human GT ที่ยอมรับ = 0 แถว** |
| CCTV learning pool v1 | `outputs/learning_pools/` | ✔ | features ไม่มีป้าย | — | — | Unknown / AI diagnostics | QA | ✘ | ส่วนบุคคล | |
| Dashboard sessions (6 session เก่า + dash4 `Sun1`) | worktree dash / dash2 / dash4 | ✔ | video + telemetry | หลายคน (รหัสนามแฝง) | 5 ท่า | ร่างจากการกดสด / review บางส่วน | review | ✘ ตอนนี้ | **วิดีโอส่วนบุคคล ห้ามอัปโหลด** | ยังไม่มีบันทึกความยินยอม หรือ `Sun1` ยังเป็น UNREVIEWED ทำให้ export ได้ 0 แถว |
| Kaggle melsmm/posture-keypoints-detection | ไม่ได้ดาวน์โหลด | ✘ | ภาพ + spine keypoints 4 จุด | ไม่ระบุ | ไม่มีป้าย 5 ท่า | Source keypoints | — | ✘ | Apache 2.0 (ไม่ระบุที่มาของภาพ) | RESEARCH_ONLY |

**สรุปช่องว่าง:** Human GT จากคนใหม่ด้วยกล้อง ANYA หรือ CCTV ที่มีความยินยอมและผ่านด่าน export = **0**

## วิธีกู้คืน dataset (ไม่อยู่ใน Git)
1. **MultiPosture:** ดาวน์โหลด `data.csv` จาก https://zenodo.org/records/14230872 ไปไว้ที่ `datasets/raw/multiposture/` แล้วตรวจ SHA-256 กับ `metadata/multiposture_zenodo_record.json` / `reports/multiposture_coordinate_forensics.md`
2. **DriveNet v6:** ตาม `metadata/drivenet_v6_source_videos.sha256` (Mendeley, CC BY 4.0) ไปไว้ที่ `datasets/raw/rgb_native/drivenet_v6/`
3. **ANYA Kaggle:** ตาม `metadata/anya_kaggle_record.json` และ `metadata/anya_hashes.sha256` (ใช้ audit เท่านั้น)
4. **dataset อื่น ๆ:** ดู `data/dataset_registry_v2.json` ซึ่งมี source, สถานะ license และข้อห้ามของแต่ละชุด
5. **ข้อมูลที่โปรเจกต์เก็บเอง** (P001 GT, dashboard sessions):
   - แจกจ่ายไม่ได้ ต้องเก็บใหม่ด้วย collector หรือ dashboard พร้อมความยินยอม
   - ตามโปรโตคอล `deliverables/ANYA_ML/DATA_COLLECTION_PROTOCOL_TH.md` (branch ML)

## โมเดลที่ไม่อยู่ใน Git
| โมเดล | Path ที่คาดไว้ | SHA-256 | วิธีกู้คืน |
|---|---|---|---|
| webcam_p001_fiveclass_exp_v1 | `outputs/models/webcam_p001_fiveclass_exp_v1/` | `4fbc8d82719f6f36bb45bf0db1e51a67bc617615a178f0ef48e35ccd4829e82a` (`model.joblib`) | `scripts/train_webcam_p001_fiveclass_exp_v1.py` + P001 GT (ข้อมูลส่วนบุคคล) หรือคัดลอกจากเครื่องหลัก |
| multiposture_v1 / multiposture_upperbody_v2 | `outputs/models/...` | ดู metrics ในเครื่อง | `python -m src.posture_ai.train --dataset multiposture --seed 42 --cv-splits 5` |
| YOLO26n weights | `outputs/yolo_tracking/weights/yolo26n.pt` | `9b09cc8b…` | ดาวน์โหลดจาก Ultralytics |
| MediaPipe pose_landmarker_heavy | `metadata/pose_landmarker_heavy.task` | `64437af8…` | ดาวน์โหลดจาก Google MediaPipe (ดู `reports/environment.md`) |

**V4** (`outputs/models/rgb_native_v4/model.joblib`) อยู่ใน Git และ frozen: `af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd`
