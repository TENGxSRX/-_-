# Upper-body posture v1 — 2026-09-17

Starting commit: `cd32ff1d31778066b1e6fb5830fd0b3f17e84abd`. Ponytail 4.10.0 full is active in this thread: injected SessionStart instructions, active flag `full`, enabled plugin, trusted SessionStart/UserPromptSubmit/SubagentStart hashes. No reinstall.

**Experimental candidates trained; office CCTV needs human ground truth.** No medical interpretation, no V4 replacement. The source-domain results below are not live CCTV accuracy.

## Exact schema: upper_body_posture_v1

Required points, in order: left shoulder (11), right shoulder (12), left hip (23), right hip (24). Head/ears, elbows and wrists are omitted: these eight features do not need them. Knees 25/26, ankles 27/28, heels 29/30, feet 31/32 and lower legs are explicitly excluded from features, validity and activity gates. Non-core points may be missing/non-finite.

Let LS/RS/LH/RH be those XYZ points; T=(LS+RS-LH-RH)/2, S=RS-LS, H=RH-LH, L=||T||, Q=hypot(Sx,Sz). Features are ordered exactly as below; angles are radians. Reject non-finite core coordinates, L<=1e-8, Q<=1e-8 or ||H||<=1e-8; never impute them.

| Feature | Exact definition | Reason |
|---|---|---|
| trunk_lateral_rad | atan2(Tx,-Ty) | lateral trunk inclination |
| trunk_depth_rad | atan2(Tz,-Ty) | forward/back depth inclination |
| shoulder_tilt_rad | atan2(Sy,Q) | shoulder alignment |
| shoulder_yaw_cos | Sx/Q | view orientation without angle wrap |
| shoulder_yaw_sin | Sz/Q | complementary view orientation |
| shoulder_width_torso | ||S||/L | torso-normalized shoulder geometry |
| hip_width_torso | ||H||/L | torso-normalized lower torso anchor |
| side_length_asymmetry | (||LS-LH||-||RS-RH||)/L | bilateral torso displacement |

Translation and positive uniform-scale invariant; camera-axis/view dependent, not viewpoint invariant. RGB uses MediaPipe world coordinates (x/y/z convention retained, no image-normalized coordinates). MultiPosture keeps its source XYZ convention. Equivalent axes/provenance between these sources are NOT proven; do not merge their rows or flip axes to improve labels.

Pose V2 default nose-inclusive core contract is preserved. Only the new diagnostic explicitly requests the four-point core. The new classifier rechecks finite features and shoulder/hip visibility >=0.5 (inherited pose-quality setting, not probability calibration). No lower-body sitting detector runs; an operator must confirm seated context per anonymous track, otherwise UNKNOWN.

## Dataset audit

| Source | Actual participants / samples | Labels and grouping | Coordinate compatibility / use |
|---|---|---|---|
| MultiPosture | 13 / 4,794 | verified source human/expert GT: TUP/TLF/TLB/TLL/TLR; subject IDs | source processed MediaPipe-like XYZ; separate five-class benchmark only |
| DriveNet RGB-native V4 pilot cache | 3 / 1,800; six videos, 300 frames each | verified compatible SS→TUP and LB→TLB protocol mapping; filename participant IDs | cached hip-centered MediaPipe world 15-point-value family; drop nose, retain four anchors; separate two-class training |
| Controlled CCTV v2 | 0 / 0 | human-label header exists; no samples | no training data |
| Local RGB v1 | 0 recorded / 100 planned manifest rows | 10 planned IDs, five classes; all WAITING_FOR_VIDEO | not actual labelled samples; excluded |
| ANYA | participant IDs unverified / 605,955 rows per CSV | Predicted_Label, clusterid, normclusterid are predictions/clusters, NOT GT | anatomical order/normalization and groups unverified; excluded |
| Other registry sources | SitPose 5,746 rows, SitStraightNet 756 images, MMD subset 35 images | missing groups, file-label joins or verified mapping | excluded; no blind feature fusion |

MultiPosture balance: TUP=1,615, TLF=1,897, TLB=442, TLL=420, TLR=420. DriveNet balance: TUP=900, TLB=900. MultiPosture has all desired direction subtypes but no SLOUCH. RGB-native GT lacks FORWARD_LEAN/LATERAL_LEFT/LATERAL_RIGHT. Correctness is the requested product mapping from direction, not independently measured ergonomic ground truth. The RGB cache was historically filtered for nose visibility too; this creates selection bias, although new inference does not require a head.

## Fixed benchmark and participant-independent validation

Three existing sklearn families only: standardized Logistic Regression (C=1, balanced), standardized RBF-SVM (C=1, gamma=scale, balanced, no probability calibration), HistGradientBoosting (100 iterations, 15 leaves, L2=1, balanced, early_stopping=False). No new dependencies or deep models. Scalers are fitted only within training folds.

MultiPosture: development subjects 1–11, 4,254 samples; 5-fold GroupKFold; fixed holdout subjects 12/13, 540 samples. Development counts TUP=1,469, TLF=1,758, TLB=307, TLL=360, TLR=360; holdout counts TUP=146, TLF=139, TLB=135, TLL=60, TLR=60.
DriveNet: development subjects 1/2, 1,200 samples; 2-fold GroupKFold (leave-one-subject-out); fixed holdout subject 3, 600 samples. Development TUP=600/TLB=600; holdout TUP=300/TLB=300. Neither final candidate is refitted on holdout. No participant overlaps; no random frame split, including internal early stopping or SVM probability calibration.

Selection uses pooled development OOF macro F1 only, fixed simpler-family tie break. Fold variability is shown below. Holdout is evaluated only after selection. These are exploratory reused corpora with historical experiments, not prospectively blinded participants.

| Family | MultiPosture OOF F1 / fold SD | DriveNet OOF F1 / fold SD |
|---|---:|---:|
| LogisticRegression | 0.9516 / 0.0216 | 0.9908 / 0.0092 |
| RBF_SVM | 0.9079 / 0.0282 | 0.9750 / 0.0000 |
| HistGradientBoosting | 0.9810 / 0.0166 | 0.9883 / 0.0117 |

Selected source-only five-class model: HistGradientBoosting; holdout macro F1=0.9840. Selected live diagnostic model: RGB-native Logistic Regression, two classes; holdout macro F1=1.0000. Inference including grouped committee: about 39.6 ms and 1.03 ms/person respectively, CPU on this host.

### Participant-level macro F1 (each model; development OOF, then selected holdout)

| Source / subject | Samples | LR | RBF-SVM | HGB |
|---|---:|---:|---:|---:|
| MultiPosture / 1 OOF | 540 | 0.9457 | 0.9124 | 0.9750 |
| MultiPosture / 2 OOF | 450 | 0.9686 | 0.8340 | 0.9807 |
| MultiPosture / 3 OOF | 480 | 0.9096 | 0.8788 | 0.9907 |
| MultiPosture / 4 OOF | 480 | 0.9089 | 0.8681 | 0.7791 |
| MultiPosture / 5 OOF | 444 | 0.7899 | 0.7811 | 0.7989 |
| MultiPosture / 6 OOF | 480 | 0.9410 | 0.8949 | 0.9828 |
| MultiPosture / 7 OOF | 300 | 0.9875 | 0.8762 | 0.9973 |
| MultiPosture / 8 OOF | 270 | 0.9954 | 0.9954 | 1.0000 |
| MultiPosture / 9 OOF | 270 | 0.9809 | 0.9641 | 0.9966 |
| MultiPosture / 10 OOF | 270 | 0.8942 | 0.8623 | 0.9963 |
| MultiPosture / 11 OOF | 270 | 0.9798 | 0.9754 | 1.0000 |
| MultiPosture / 12 holdout | 270 | — | — | 0.9714 |
| MultiPosture / 13 holdout | 270 | — | — | 0.9969 |
| DriveNet / 1 OOF | 600 | 1.0000 | 0.9750 | 1.0000 |
| DriveNet / 2 OOF | 600 | 0.9817 | 0.9750 | 0.9767 |
| DriveNet / 3 holdout | 600 | 1.0000 | — | — |

### Selected holdout per-class results (before refusal)

| Source / class | Count | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| MultiPosture / TLB | 135 | 0.9926 | 1.0000 | 0.9963 |
| MultiPosture / TLF | 139 | 1.0000 | 1.0000 | 1.0000 |
| MultiPosture / TLL | 60 | 1.0000 | 1.0000 | 1.0000 |
| MultiPosture / TLR | 60 | 1.0000 | 0.9000 | 0.9474 |
| MultiPosture / TUP | 146 | 0.9603 | 0.9932 | 0.9764 |
| DriveNet / TLB | 300 | 1.0000 | 1.0000 | 1.0000 |
| DriveNet / TUP | 300 | 1.0000 | 1.0000 | 1.0000 |

Confusion matrices: rows=true, columns=prediction. MultiPosture order TLB/TLF/TLL/TLR/TUP; DriveNet order TLB/TUP.

```text
MultiPosture: [[135, 0, 0, 0, 0], [0, 139, 0, 0, 0], [0, 0, 60, 0, 0], [0, 0, 0, 54, 6], [1, 0, 0, 0, 145]]
DriveNet: [[300, 0], [0, 300]]
```

Binary before refusal: MultiPosture CORRECT recall=0.9932, INCORRECT recall=0.9848, balanced accuracy=0.9890; DriveNet all three=1.0000. These are source holdout values, not CCTV measurements.

## UNKNOWN and transfer evidence

Reject bad/missing shoulders or hips, low visibility, multi-person ambiguity, pending/unavailable pose, unverified seated context, marginal training-range OOD or disagreement between final and grouped-fold models. No invented probability threshold. Range-box and unanimous-vote policy is **EXPERIMENTAL**, uncalibrated and may both miss joint OOD and over-reject valid people. Unsupported poses may still fall inside the two-class domain; rejection cannot guarantee recognition of missing subtypes.

MultiPosture holdout: coverage=95.19%, 26 UNKNOWN (25 disagreement / 1 OOD); accepted macro F1=0.9778. TLR coverage only 60%. End-to-end CORRECT recall=0.9932 / INCORRECT recall=0.9213 when refusal counts as not resolved.
DriveNet holdout: coverage=41.33%, 352 UNKNOWN, all range OOD; accepted binary recalls and balanced accuracy=1.0000 on 248 accepted samples. **End-to-end CORRECT recall=0.0100 / INCORRECT recall=0.8167**. Per-class coverage TUP=1% / TLB=81.67%. This is a severe coverage imbalance; the threshold was NOT relaxed using holdout results.
Frozen MultiPosture candidate evaluated on 1,800 separately cached RGB rows: no correct subtype predictions, binary balanced accuracy=0.0389 before refusal; all 1,800 rejected as OOD. It is not used as the live RGB candidate. No adapter was tuned against source labels.

## Real-camera qualifications

Initial existing Pose V2 path: 30.65 s, 308 frames, max nine people, 15 IDs; MediaPipe executes, 22 attempts, 2 POSE_OK, 20 unavailable, 24 pending, 6 ambiguous. Source closes and DURATION_COMPLETED. All 52 telemetry records have v4_prediction=null. Regression tests specifically verify unselected overlapping boxes block pose without spending budget.
New RGB candidate diagnostic: 302 frames / 23 IDs in a 35-second bounded run; 41 attempts, 19 POSE_OK, 16 unavailable, 6 invalid, 177 pending, 25 ambiguous. All 243 predictions UNKNOWN: 19 valid poses rejected as OOD and the rest pose states. GUI displays track ID and CORRECT/INCORRECT/UNKNOWN plus subtype when accepted. Lower-body landmarks never enter features or validity. Seated context is operator-confirmed per ID; none were confirmed during automated qualification.
A single transient real frame was inspected: desks/monitors hide lower-body regions and portions of torsos/hip anchors; a near foreground person is cropped. The temporary frame was deleted. This is a domain observation, not labelled posture accuracy. No continuous raw video, retained images, credentials or face data in committed artifacts. Camera clock overlay differs from host date; configuration was not changed.

## Blind human-GT collector and protocol

The same small RTSP script has a collector mode that does not load ANY posture classifier, display predictions, or call V4. Anonymous IDs P001–P999 (P000 invalid); reuse the same ID for the same participant across sessions. Click one verified seated person, SPACE to freeze their exact successfully extracted pose frame, inspect that target, then choose: 1 UPRIGHT, 2 FORWARD_LEAN, 3 BACKWARD_OR_RECLINED, 4 participant anatomical LEFT, 5 anatomical RIGHT, 0 UNKNOWN/SKIP. If sitting, hip anchors, direction, or person association is unclear, skip; do not infer posture from head-only movement. SLOUCH is not offered.
UPRIGHT: trunk approximately neutral/upright. FORWARD_LEAN: torso leans toward the workspace. BACKWARD_OR_RECLINED: torso leans away/back into the seat. LEFT/RIGHT: torso lateral lean relative to the participant, not image screen direction. This is direction annotation; annotators must agree on these cases before treating office labels as distinguishable. Compound/occluded poses use UNKNOWN/SKIP. No numeric ergonomic or medical cutoff is claimed.
Frozen images stay in RAM; only four-point world XYZ, eight features, visibility, host/frame metadata and a human-selected label are saved. Human rows are HUMAN_GROUND_TRUTH; diagnostic rows are MODEL_PREDICTION in separate files. No model value can populate human labels. Collector smoke: 157 frames / 13 IDs, clean close, model_loaded=false, zero human labels (no operator annotation attempted). Small single-frame participant examples can be collected; no demand for hundreds of clips.

```powershell
& '.\.venv_yolo\Scripts\python.exe' scripts/run_upper_body_posture.py --max-seconds 60
& '.\.venv_yolo\Scripts\python.exe' scripts/run_upper_body_posture.py --collect-human-gt --participant P001 --max-seconds 300
& '.\.venv\Scripts\python.exe' scripts/train_upper_body_posture_v1.py --dataset drivenet --output outputs/models/upper_body_posture_v1_drivenet_repeat
& '.\.venv\Scripts\python.exe' scripts/train_upper_body_posture_v1.py --dataset multiposture --output outputs/models/upper_body_posture_v1_multiposture_repeat
```

Run from repository root. Existing output directories are refused, preserving artifacts. Diagnostic click toggles confirmed seated context. No new frontend/dashboard/network/Tailscale changes.

## Verification / Ponytail review / freeze

23 focused tests pass: upper-body schema/missing lower body, scale/translation invariance, core degeneration, mapping, ambiguity, visibility, disagreement/OOD, blind-GT separation, off-ontology confusion counts, default and four-point backend cores, Pose V2 global budget, all-box regressions and long-session integration/shutdown. V4 remains unchanged; SHA256 `af61482ed98a3c71f1200933a981afbbf67853b26ea54691166a71edaf9855fd`.
Ponytail review: reused existing source, tracker, CandidateSelector and Pose V2; one script shares diagnostic/collector plumbing; no dependency/framework/frontend additions. Removed explicit default SVM probability argument. Grouped validation, GT/prediction separation, UNKNOWN, privacy and V4 guards retained. Lean already. Ship.
The 4 KB RGB candidate and its metrics are versioned separately. Larger five-class experiment/metrics remain local under outputs/models/upper_body_posture_v1_multiposture_exp. No pre-existing untracked file or Z.ai worktree was changed. Full machine metrics include per-participant/class counts and confusion matrices. Next evidence needed: small participant-labelled desk-camera examples and a prospectively held-out participant set; only then validate rejection coverage and add missing RGB-native directions.

UPPER_BODY_POSTURE_NEEDS_MORE_GROUND_TRUTH
