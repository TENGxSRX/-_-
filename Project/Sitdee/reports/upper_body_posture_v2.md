# upper_body_posture_v2 — experimental, 2026-09-17

Starting commit: `e83cbe6336f5716e07f12e9eb4a10b0277bc3f55`. One milestone: verified-data upper-body robustness benchmark and one short paired CCTV diagnostic. No autonomous pool, new human GT, tracker/selector/frontend changes, continuous training, raw media recording, or medical diagnosis. Existing untracked live-demo work was left intact.

## Landmark compatibility audit

The local MultiPosture CSV contains canonical 33 XYZ groups, but released preprocessing/scaler provenance is unresolved. DriveNet's existing `world` cache contains exactly five hip-translated XYZ anchors: nose, LS, RS, LH, RH. Hip translation cancels in the relative upper-body features; it does not require live hips. Name compatibility is defensible; cross-source numerical distribution equivalence is **not proven**, so datasets are never pooled or adapted using labels.

| Landmark | MultiPosture | DriveNet accepted cache | Real CCTV MediaPipe | Shared schema |
|---|---|---|---|---|
| nose (0) | XYZ; no visibility column | XYZ; original visibility >=0.5 | XYZ and visibility, when detected | Yes, experimental head anchor; not a neck/trunk measurement |
| left_shoulder (11) | XYZ | XYZ; original visibility >=0.5 | XYZ and visibility | Yes; required |
| right_shoulder (12) | XYZ | XYZ; original visibility >=0.5 | XYZ and visibility | Yes; required |
| left_hip (23) | XYZ | XYZ; original visibility >=0.5 | May be obscured/unreliable | Optional pair only |
| right_hip (24) | XYZ | XYZ; original visibility >=0.5 | May be obscured/unreliable | Optional pair only |
| left_ear / right_ear (7/8) | Named XYZ | Not persisted | Potentially available | No |
| left_elbow / right_elbow (13/14) | Named XYZ | Not persisted | Potentially available | No |
| eyes / mouth / wrists | Named XYZ | Not persisted | Potentially available | No |
| neck | No direct landmark | No direct landmark | No direct landmark | No; not invented |

Official [MediaPipe documentation](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python) distinguishes image landmarks from world landmarks. This experiment uses world XYZ only. See existing local `multiposture_coordinate_forensics.md`, `multiposture_landmark_mapping_verified.md`, and `train_rgb_native_v4.py` for release/cache provenance.

## Exact feature conditions

Let S=RS-LS, M=(LS+RS)/2, H=RH-LH, T=M-(LH+RH)/2, L=||T||, W=||S||, Q=hypot(Sx,Sz), N=(nose-M)/W. Angles are radians. Translation and positive uniform scale cancel; camera/view dependence remains. No lower-body point enters features, inference validity or seated-context decisions.

- **CURRENT_BASELINE (8):** `trunk_lateral_rad=atan2(Tx,-Ty)`, `trunk_depth_rad=atan2(Tz,-Ty)`, `shoulder_tilt_rad=atan2(Sy,Q)`, `shoulder_yaw_cos=Sx/Q`, `shoulder_yaw_sin=Sz/Q`, `shoulder_width_torso=W/L`, `hip_width_torso=||H||/L`, `side_length_asymmetry=(||LS-LH||-||RS-RH||)/L`. Exact unchanged V1 transform; both hips required.
- **NO_HIP (6):** `shoulder_tilt_rad`, `shoulder_yaw_cos`, `shoulder_yaw_sin`, `nose_shoulder_x_width=Nx`, `nose_shoulder_y_width=Ny`, `nose_shoulder_z_width=Nz`. Nose/shoulders required; hips never read for feature values. Head direction is not proof of torso posture.
- **HIP_DROPOUT_ROBUST (12):** the six NO_HIP features plus `trunk_lateral_rad`, `trunk_depth_rad`, `shoulder_width_torso`, `hip_width_torso`, `side_length_asymmetry`, `hips_available`. If either hip is missing/non-finite/visibility<0.5, or optional torso geometry is degenerate, omit the hip pair: five optional values=0 and flag=0. No synthetic hip position is invented. Otherwise flag=1.

Dropout adds one both-hip-missing copy per **training** sample, preserving labels. Partial hip loss has the same pair-omission representation, so extra identical augmented views are unnecessary. Each fit/scaler sees only its training participants. Original validation/holdout data remain unchanged. Left/right/both-hidden stress views are copied solely for inference; no parameter or range fitting uses them.

Rejection remains **EXPERIMENTAL**: image visibility>=0.5, valid required anchors, finite/nondegenerate upper-body geometry, training min/max and unanimity of final plus grouped-fold models. Missing hip dimensions are omitted from the OOD check; observed optional ranges exclude dropout sentinel zeros. There is no probability calibration or CCTV-fitted threshold. Ambiguity, unavailable/pending pose and invalid nose/shoulders remain UNKNOWN. Product sitting output additionally requires operator-confirmed seated context per transient track; otherwise UNKNOWN. No lower-body sitting classifier runs.

## Data, evaluation and selection

Only MultiPosture source `upperbody_label` and verified DriveNet SS→TUP / LB→TLB protocol joins are supervised GT. No ANYA predicted labels, cluster IDs, geometry outputs, predictions, pseudo labels or camera telemetry are training labels. Binary mapping TUP→CORRECT and other supported directions→INCORRECT follows the requested product convention, not independent ergonomic GT.

MultiPosture: 13 participants, 4,794 rows (TUP=1,615, TLF=1,897, TLB=442, TLL=420, TLR=420). Development subjects 1–11: 4,254 rows, 5-fold GroupKFold; fixed holdout 12/13: 540 rows. DriveNet: 3 participants, 1,800 accepted rows (TUP=900, TLB=900). Development 1/2: 1,200 rows, 2-fold GroupKFold/LOSO; holdout 3: 600 rows. Historical nose/hip filtering creates selection bias: simulated masking tests feature availability, not the full occlusion effect on MediaPipe.

Exactly three fixed model families per condition/source: standardized balanced Logistic Regression (C=1, max_iter=2000), standardized balanced RBF-SVM (C=1, gamma=scale), balanced HistGradientBoosting (100 iterations, 15 leaves, L2=1, early_stopping=False), seed=42 where supported. Existing dependencies only. No random frame split, early-stopping split, probability-calibration split or tuning sweep.

Selection rule declared before evaluation: use separate RGB-native DriveNet candidates for live compatibility, maximize minimum of normal/fully-hidden development OOF macro F1; candidates within 0.01 prefer LR→SVM→HGB, then NO_HIP→HIP_DROPOUT_ROBUST. Baseline is only a reference. Holdout/CCTV do not select or tune; the persisted model/committee retain development-only fits. Corpora were used historically, so this remains exploratory.

Full per-class precision/recall/F1, confusion matrices, binary balanced accuracy, per-participant metrics, fold audits, holdout refusal coverage and source/frozen hashes for all 18 runs are persisted in `outputs/models/upper_body_posture_v2/metrics.json`.

## Benchmark results

| Source | Condition | Model | OOF macro F1 | Both-hidden OOF F1 | Holdout F1 | Normal holdout coverage | Binary holdout BA |
|---|---|---|---:|---:|---:|---:|---:|
| MultiPosture | CURRENT_BASELINE | LogisticRegression | 0.9516 | None | 0.9915 | 92.41% | 0.9795 |
| MultiPosture | CURRENT_BASELINE | RBF_SVM | 0.9079 | None | 0.9555 | 86.11% | 0.9205 |
| MultiPosture | CURRENT_BASELINE | HistGradientBoosting | 0.9810 | None | 0.9840 | 95.19% | 0.9890 |
| MultiPosture | NO_HIP | LogisticRegression | 0.6174 | 0.6174 | 0.6527 | 42.78% | 0.7455 |
| MultiPosture | NO_HIP | RBF_SVM | 0.6210 | 0.6210 | 0.7879 | 39.26% | 0.8089 |
| MultiPosture | NO_HIP | HistGradientBoosting | 0.6184 | 0.6184 | 0.7008 | 52.41% | 0.7470 |
| MultiPosture | HIP_DROPOUT_ROBUST | LogisticRegression | 0.9283 | 0.6123 | 0.9900 | 91.30% | 0.9760 |
| MultiPosture | HIP_DROPOUT_ROBUST | RBF_SVM | 0.9166 | 0.6339 | 0.9744 | 95.00% | 0.9706 |
| MultiPosture | HIP_DROPOUT_ROBUST | HistGradientBoosting | 0.9535 | 0.6312 | 0.9554 | 84.44% | 0.9801 |
| DriveNet | CURRENT_BASELINE | LogisticRegression | 0.9908 | None | 1.0000 | 41.33% | 1.0000 |
| DriveNet | CURRENT_BASELINE | RBF_SVM | 0.9750 | None | 0.9583 | 30.83% | 0.9583 |
| DriveNet | CURRENT_BASELINE | HistGradientBoosting | 0.9883 | None | 1.0000 | 17.50% | 1.0000 |
| DriveNet | NO_HIP | LogisticRegression | 0.7468 | 0.7468 | 0.9398 | 57.33% | 0.9400 |
| DriveNet | NO_HIP | RBF_SVM | 0.8423 | 0.8423 | 0.7433 | 60.67% | 0.7583 |
| DriveNet | NO_HIP | HistGradientBoosting | 0.8083 | 0.8083 | 0.8293 | 54.00% | 0.8333 |
| DriveNet | HIP_DROPOUT_ROBUST | LogisticRegression | 0.9517 | 0.7361 | 1.0000 | 26.33% | 1.0000 |
| DriveNet | HIP_DROPOUT_ROBUST | RBF_SVM | 0.9608 | 0.8446 | 0.8921 | 31.00% | 0.8933 |
| DriveNet | HIP_DROPOUT_ROBUST | HistGradientBoosting | 0.9087 | 0.8175 | 0.9633 | 15.33% | 0.9633 |

None for baseline hidden hips means impossible inference / 0% coverage, not an accuracy score. NO_HIP left/right/both-hidden values equal the original by construction. HIP_DROPOUT partial and fully-hidden views share pair omission.

### Cost / benefit within the same model family

| Source | Model | NO_HIP minus baseline OOF (pp) | HIP_DROPOUT minus baseline normal OOF (pp) | HIP_DROPOUT hidden minus NO_HIP hidden (pp) |
|---|---|---:|---:|---:|
| MultiPosture | LogisticRegression | -33.41 | -2.33 | -0.52 |
| MultiPosture | RBF_SVM | -28.70 | +0.87 | +1.29 |
| MultiPosture | HistGradientBoosting | -36.27 | -2.76 | +1.28 |
| DriveNet | LogisticRegression | -24.40 | -3.92 | -1.07 |
| DriveNet | RBF_SVM | -13.27 | -1.42 | +0.24 |
| DriveNet | HistGradientBoosting | -18.00 | -7.96 | +0.92 |

### Selected candidate

`upper_body_posture_v2`: DriveNet / NO_HIP / RBF_SVM; six features: `shoulder_tilt_rad`, `shoulder_yaw_cos`, `shoulder_yaw_sin`, `nose_shoulder_x_width`, `nose_shoulder_y_width`, `nose_shoulder_z_width`.

NO_HIP SVM robust OOF=0.8423 versus optional-hip SVM=0.8446, only 0.24 pp apart; NO_HIP wins the declared simplicity tie. Same-family baseline normal OOF=0.9750: cost=-13.27 pp. Compared with live V1 LR OOF=0.9908: cost=-14.86 pp. It supports UPRIGHT and BACKWARD_OR_RECLINED only; other direction subtypes remain unsupported. The MultiPosture five-class feasibility result near 0.62 warns against claiming general five-class robustness.

Holdout participant 3 raw F1=0.7433, binary BA=0.7583; accepted-only F1=0.9261, BA=0.9144; coverage=60.67% (236 UNKNOWN: 66 OOD, 170 disagreement). All three hip-missing holdout views are identical. Coverage TUP=72.67%, TLB=48.67%; accepted-only scores must not be mistaken for overall performance.

| Selected holdout class | Precision | Recall | F1 |
|---|---:|---:|---:|
| TLB | 1.0000 | 0.5167 | 0.6813 |
| TUP | 0.6742 | 1.0000 | 0.8054 |

Confusion matrix, rows=true, columns=predicted, order TLB/TUP: `[[155,145],[0,300]]`. Accepted-only matrix: `[[121,25],[0,218]]`.

| Selected candidate participant | Evaluation | Samples | Macro F1 |
|---|---|---:|---:|
| 1 | grouped OOF | 600 | 0.7922 |
| 2 | grouped OOF | 600 | 0.8878 |
| 3 | fixed holdout | 600 | 0.7433 |

The 0.7922?0.8878 development participant variation and 0.7433 holdout F1 show a real unseen-person limitation. Backward recall=0.5167 on the full holdout is not strong balanced performance. Selection is experimental rather than a production replacement. Complete participant variations for every family/condition/source are in metrics.json.

Candidate SHA256: `f103daf16ac52c6e3ed4902a1a9502f27326c20aa48db465b828e82bc37d3b6a`. V1 artifacts unchanged relative to training-start hashes. Frozen V4 SHA256: `af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd`.

## Same-session real CCTV diagnostic

Existing private HIKVISION_MAIN config; RTSP -> unchanged YOLO26n -> unchanged ByteTrack -> unchanged CandidateSelector -> MediaPipe Heavy (nose/shoulder core) -> frozen V1 and new V2 on the SAME evidence. Both retain visibility>=0.5, training-range OOD and unanimous committee policy. No threshold changes, human labels, raw images or video.

Session: outputs/upper_body_v2_sessions/20260917T062440773506Z; DURATION_COMPLETED; 60.02s; 573 frames, 71 sampled attempts, 45 valid upper-body poses; 17 transient IDs (not unique humans), maximum 10 simultaneous detections.

| Sampled-attempt metric | V1 | V2 |
|---|---:|---:|
| CORRECT_SITTING | 0 | 0 |
| INCORRECT_SITTING | 0 | 0 |
| UNKNOWN | 71 | 71 |
| OOD | 45 | 45 |
| Rejected due missing hips | 0 | 0 |
| Rejected due other missing core | 0 | 0 |
| Pose unavailable | 26 | 26 |
| UNKNOWN rate | 100% | 100% |
| OOD rate / all attempts | 63.38% | 63.38% |
| OOD rate / valid poses | 100% | 100% |

Product outputs also all UNKNOWN: no seated context confirmed in this headless diagnostic, and raw classifiers already rejected every attempt. Pose pending=218 and ambiguous/overlap=178 telemetry records are excluded from attempt denominators.

Hip visibility proxy: visible=45, partly hidden=0, both hidden=0, unobservable because pose unavailable=26. Those unavailable poses cannot be labeled hidden hips. Range bbox proxy: near=0, mid=48 (45 valid/OOD, 3 unavailable), far=23 (all unavailable). Multiple people were present, but no successful pose with verified desk/monitor-hidden hips or near-range pose was obtained.

Additional hidden-hip cases handled=0. Optional-hip behavior is demonstrated offline and by persisted real-runner tests; hidden hips stopping live rejection is NOT demonstrated in this session. No UNKNOWN/OOD improvement or CCTV accuracy is claimed. MediaPipe confidence is not physical-occlusion GT.

Remaining limits: camera viewpoint/crop domain OOD; far pose/crop availability; overlap gating; weak unseen-person backward recall; two RGB-native GT classes only; synthetic landmark masking does not reproduce occlusion-dependent pose errors. No threshold tuning, collection or subsequent milestone was started.

## Ponytail review and verification

Review removed three duplicate shoulder features from HIP_DROPOUT and identical dropout feature views, retaining one pair-omission training copy. Reused V1 transforms/metrics and existing source/pose/tracker/selector. Batch rejection shares the live policy. No new dependencies, abstractions or tuning sweep. Final review: Lean already. Ship. Grouped evaluation, verified GT, UNKNOWN and freeze safeguards retained.

Focused tests: no lower body, optional hips, deterministic features, training-only dropout with unchanged labels/validation, sentinel OOD protection, persisted reload, actual PoseV2Stage with absent hips, persisted runner classification without hips, invalid nose/shoulders and ambiguity/OOD rejection, unchanged V1/V4. Relevant regressions also cover pose, V4, tracker and RTSP tools. Result: **34 passed** with `.venv/Scripts/python.exe -m pytest tests/test_upper_body_posture_v2.py tests/test_upper_body_posture_v1.py tests/test_pose_v2_stage.py tests/test_rgb_native_v4.py tests/test_yolo_tracking_adapter.py tests/test_rtsp_tools.py -q`.

Files changed/created: .gitignore; scripts/train_upper_body_posture_v2.py; scripts/run_upper_body_posture_v2.py; src/posture_ai/features/upper_body_posture_v2.py; src/posture_ai/inference/upper_body_posture_v2.py; tests/test_upper_body_posture_v2.py; this report; outputs/models/upper_body_posture_v2/candidate.joblib; outputs/models/upper_body_posture_v2/metrics.json. Session telemetry remains ignored/private. Pre-existing untracked live-demo files excluded.

Run from project root; click a confirmed seated track to enable product sitting output; Q stops:

```powershell
.venv\Scripts\python.exe scripts\run_upper_body_posture_v2.py --profile HIKVISION_MAIN --max-seconds 60
```

Add --no-display for paired headless diagnostic. Training command: .venv\Scripts\python.exe scripts\train_upper_body_posture_v2.py (refuses a nonempty versioned output directory).

UPPER_BODY_V2_NEEDS_CCTV_DATA. Stop after this milestone; no autonomous collection.
