# AUTONOMOUS_CCTV_LEARNING_POOL_V1

Starting commit: `7ef4590ea2437d9c81f820df69cd7954d897d37d`. One milestone only: observe real CCTV and build a bounded **UNLABELED** feature pool. No posture training, threshold changes, pseudo labels, source-dataset merging, human labeling, manual posing, tracker/pose-stack redesign or medical claims.

## Pipeline and schema

Existing private HIKVISION_MAIN RTSP -> latest-frame input -> existing YOLO26n person detection (confidence=0.35, GPU 0) -> existing ByteTrack -> existing CandidateSelector (0.5s representative/event intervals) -> existing MediaPipe Heavy/PoseV2Stage (2 global pose attempts/s, original overlap/crop gates) -> six upper-body V2 NO_HIP features -> quality gate -> temporal/feature dedup -> deterministic representative indices -> bounded JSONL/manifest. No GUI or operator confirmation is needed for collection. Diagnostic sitting predictions do not assert a verified seated context.

Dataset identity/schema: `CCTV_LEARNING_POOL_V1`. Runtime root: `outputs/learning_pools/CCTV_LEARNING_POOL_V1/SESSION_<UTC>`. Every record has `label_status=UNLABELED`, `ground_truth=null`, `label_source=null`, `source_domain=HIKVISION_CCTV`. V1/V2 metadata is nested under `diagnostics` with `label_source=MODEL_PREDICTION`. There is no automatic promotion to GT; nested non-null GT, human label/pseudo-label fields and HUMAN_GROUND_TRUTH values are rejected. Existing verified human labels are not imported in this milestone.

Pose observations contain UTC timestamp, relative monotonic seconds, session/sample/frame IDs, transient anonymous track ID, bbox/detector confidence, normalized image center, bbox height/area fraction, 3x3 zone, NEAR/MID/FAR scale group, pose/ambiguity/quality/tracking state, five anchors (nose, left/right shoulder, optional left/right hip), finite world/crop-image XYZ or null, coordinate-presence and visibility-based availability masks, visibility, and presence_metric=null (existing pose evidence does not expose a separate detection-presence metric).

Feature schema `upper_body_posture_v2/NO_HIP`, ordered: shoulder_tilt_rad, shoulder_yaw_cos, shoulder_yaw_sin, nose_shoulder_x_width, nose_shoulder_y_width, nose_shoulder_z_width. Nose/shoulders are required, hips optional. No knees/ankles/heels/feet/lower legs are read or stored. Both V1/V2 predictions/reasons and OOD booleans are diagnostics. Confidence=null because existing refusal-capable runners expose no calibrated confidence. `range_excess_max` is max(max(training_min-x, x-training_max, 0)) over the model's features; a diagnostic raw range excess, not a calibrated OOD probability or a new rejection threshold.

## Quality, dedup and representative selection

Unusable/non-finite/degenerate nose/shoulder evidence, unavailable pose, severe overlap and required crop-image anchors outside [-0.05,1.05] are rejected from candidates, with explicit statistics and diagnostic records. The crop check is a contamination proxy, not proof of person assignment. Original PoseV2Stage minimum padded crops 80x120 and overlap IoU>=0.25 remain intact. Pending budget records are counted rather than redundantly logged. Overlap records have a 10s per-track cooldown; all overlap observations remain counted. Missing hips never fail the quality gate; their tag is a confidence-based occlusion proxy, not verified desk/monitor occlusion.

The raw observations stream retains quality-valid features even when OOD and records rejected attempts/events separately. Candidate indices reference this single payload, avoiding duplicate landmark/feature storage. Per-track retention requires a 5s minimum gap plus feature RMS distance>=0.10, or primary-status/zone/scale/30-minute-period change, or a 300s refresh. New tracks are considered, but an ACTIVE track matching another recent retained track within 60s and RMS<0.05 is deduplicated. This is feature redundancy suppression, not reidentification. Track memory and recent feature-cell history are bounded; old regions can be marked novel again after eviction.

Logical tags: VALID_NORMAL, OOD, UNCERTAIN, NOVEL_DIVERSE, OVERLAP_OR_OCCLUSION, TRACKING_EVENT. Primary status follows frozen V2: OOD rejection -> OOD; other UNKNOWN -> UNCERTAIN; accepted -> VALID_NORMAL. These are distribution/diagnostic buckets, never posture labels. NOVEL_DIVERSE uses fixed-width 0.15 feature cells. Normal-domain here means accepted by the source-trained refusal policy; it does not establish posture correctness or guaranteed in-distribution membership.

Representative selection keeps one retained reference per feature cell. At the cap, it favors less-covered scale/zone/primary-status/30-minute strata and less-represented transient tracks; SHA256(sample_id) deterministically breaks ties. No preference for maximum OOD score, no clustering labels, no model fitting, no vector database. It cannot create normal/uncertain/spatial coverage absent from natural data. Default cap=512 representatives (configurable 1-4096); bounded O(cap) replacement scan. Replaying the same session order/IDs yields the same selection.

NEAR: bbox_height/image_height>=0.5; MID: >=0.25; FAR: <0.25. Zones use clamped thirds of normalized bbox center. These are image-scale proxies, not calibrated physical distance. Coverage is reported for quality-valid raw samples and retained candidates separately.

## Storage and unattended operation

Existing RotatingJsonlWriter is reused: 8MiB files, rotate then gracefully STOP, never delete or overwrite earlier sessions. Default 512MiB total across the pool root including previous sessions and a 1MiB finalization reserve; default 64 JSONL files per session (48 observations / 16 candidate indices), plus bounded manifest/representative JSON. Default free-disk floor=512MiB, checked at preflight and at most every 5 seconds while writing. A root-level exclusive collector lock prevents concurrent writers; it is removed on normal/exception/keyboard shutdown. After an OS kill/crash a stale lock intentionally requires inspection before manual removal; prior data are preserved.

Manifest checkpoints every 30s; clean finalization closes source/pose/writers, writes state=FINALIZED and stopped reason, sample/quality/rejection/spatial counts, storage limits/files/bytes, label/privacy/selection policies, model fingerprints and runtime statistics. Retained counters count completed candidate-index writes; valid/rejection counters count observations reaching the quality decision, including a last observation if capacity stops its append. Writer records_written/bytes_written report exact completed storage. Tracking is limited to 65,536 distinct transient IDs per session and stops explicitly at the bound. A requested long duration can end earlier on storage/disk/source/memory safety limits; it does not silently continue or erase data.

Continuous raw video and images are OFF. Short event clips are **deferred**: adding pixel buffering/encoding and separate clip budgets is unnecessary for the feature-pool milestone. No names, face identities, images, clips or secret-bearing URLs are stored. OpenCV/FFmpeg camera log output is suppressed; exception text is omitted, preserving only safe exception types/reasons. Runtime data remain ignored by git.

Ten-minute qualification from project root:

```powershell
.venv\Scripts\python.exe scripts\collect_cctv_learning_pool_v1.py --duration-minutes 10
```

Unattended 16-hour collection (headless, no interaction after launch):

```powershell
.venv\Scripts\python.exe scripts\collect_cctv_learning_pool_v1.py --duration-minutes 960 --max-reconnects 1000 --max-pool-mib 1024 --max-files 128
```

The long command raises the existing client's bounded retry allowance and explicitly budgets a 1GiB root / 128 JSONL files (96 observations, 32 indices), avoiding the smaller default file allowance becoming the first limit for longer natural sessions. Camera/network configuration and posture thresholds stay unchanged. Qualification used the default five-reconnect allowance and 512MiB/64-file budget. The released collector's 65,536-ID bound supports natural fragmentation in longer runs; qualification remained far below both this bound and the initial 8,192-ID guard. Neither a full 16-hour run nor exhaustion of real-camera storage was claimed; those capacity branches are covered by focused tests.

## Review and tests

Ponytail review: use existing JSONL writer/atomic manifest, a single stored payload with lightweight candidate/representative references, fixed feature grid rather than a clustering service or database, bounded incremental state, and deferred video. GT/prediction separation, privacy, capacity stops, reproducibility and manifests remain. Final code review and real qualification results are recorded below after collection finishes.

Focused tests cover automatic/nested GT rejection, credential/name/pseudo-label rejection, feature/core/overlap/crop quality, optional hips, temporal/feature/state/refresh/cross-track dedup, representative balance, scale boundaries, file/root-byte/disk capacity, manifest/reference generation, idempotent clean close, exclusive lock, and actual collector-loop shutdown on KeyboardInterrupt and credential-bearing RuntimeError. Relevant regressions cover frozen V2, pose stage, RTSP and existing autonomous long-session utilities.

V1/V2 artifacts and thresholds remain frozen; V4 SHA256 must remain `af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd`.

Files: `.gitignore`, `scripts/collect_cctv_learning_pool_v1.py`, `src/posture_ai/monitoring/cctv_learning_pool.py`, `tests/test_cctv_learning_pool_v1.py`, this report. Pre-existing untracked live-demo files are excluded.

## Real qualification: MEASURED

Session: SESSION_20260917T070204941409Z. Actual existing HIKVISION_MAIN stream, ordinary room activity, no manual pose or labels. DURATION_COMPLETED; source/pose/writers closed, manifest FINALIZED, root lock removed. Duration 600.1027s (10.00 minutes), processed FPS about 9.20. Three RTSP reconnects recovered during the run; no collector crash or capacity stop. 16-hour stability was not measured. The PowerShell tool wrapper reports exit 1 when native stderr is redirected even for sys.exit(0), reproduced with a harmless stderr-only check; completed session state and collector return path were independently verified.

| Metric | Measured |
|---|---:|
| Processed frames | 5,519 |
| Person detections (sum over frames) | 47,368 |
| Distinct transient track IDs (not unique people) | 470 |
| Maximum simultaneous detections | 14 |
| Sampled pose attempts | 841 |
| Quality-valid upper-body poses | 493 |
| Rejected sampled poses | 348 |
| Rejected: pose unavailable | 347 |
| Rejected: crop contamination proxy | 1 |
| Other sampled rejection reasons | 0 |
| Valid OOD (frozen V2 diagnostic) | 485 |
| Valid UNCERTAIN | 1 |
| VALID_NORMAL (diagnostic only) | 7 |
| Dedup rejected | 274 |
| Retained candidate references | 219 |
| Representative references | 217 |
| Representative OOD / normal / uncertain | 213 / 3 / 1 |
| Representative transient tracks | 85 |
| Overlap observations, pre-attempt gate | 2,471 |
| Stored cooldown-limited overlap records | 391 |
| Tracking NEW / LOST / REACQUIRED events | 470 / 1,881 / 1,419 |
| Tracking-event records total | 3,770 |
| Raw observation/event JSONL records | 5,002 |
| Pose-observation records, including overlap | 1,232 |
| Pose pending budget records (counted only) | 3,668 |
| Storage total, including manifest/representatives | 4,438,513 bytes / 4.23 MiB |
| Session files | 4 |

Tracking loss/reacquisition counts describe detector-frame ID gaps, not verified human departure/return or identity continuity. Ambiguous overlap was rejected BEFORE sampling and is excluded from 841 sampled attempts; it is separately counted/stored. Pose backend reported 494 POSE_OK; the crop quality check rejected one, leaving 493 candidates eligible for dedup. Missing hips are not a rejection condition. Features stay unlabeled even when a model emits a sitting output.

| Scale proxy | Raw quality-valid | Retained | Representative |
|---|---:|---:|---:|
| NEAR | 98 | 46 | 46 |
| MID | 382 | 161 | 159 |
| FAR | 13 | 12 | 12 |

Dedup removed 55.58% of quality-valid poses; representatives retain 44.02% of raw valid features in this short session. The 512-reference cap was not reached; larger multi-hour pools are bounded by that cap. All scale groups appeared naturally, but FAR usable-pose coverage remains small, normal/uncertain buckets are rare, and the 10-minute session covers one 30-minute time stratum. No room coverage, physical desk occlusion, posture accuracy, participant-independent GT or long-duration performance is invented.

Post-run integrity audit PASS: every raw/index record validates UNLABELED/null GT; all 493 feature vectors regenerate from only available upper-body anchors to atol=1e-12; all 219 references resolve to quality-valid observations; all 217 representatives resolve to retained indices; replay with released selection logic exactly matches saved representatives; frozen V1/V2/V4 hashes match; no pixels/identities/URLs were stored. Runtime manifests/data remain ignored by git.

Final Ponytail review: Lean already. Ship. No additional infrastructure or duplicate feature payload; safety/scientific safeguards retained. Relevant focused/regression suite: 42 passed. Command:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_cctv_learning_pool_v1.py tests/test_upper_body_posture_v2.py tests/test_pose_v2_stage.py tests/test_rtsp_tools.py tests/test_autonomous_long_session.py -q
```

V4 verified unchanged: af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd.

CCTV_LEARNING_POOL_READY. Stop here: no human labeling, posture training or subsequent milestone.
