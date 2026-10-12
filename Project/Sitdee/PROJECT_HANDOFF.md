# Roles

ChatGPT = Planner / Gatekeeper  
Gemini Flash = Researcher  
Codex = Primary Engineer  
Z.ai / ZCode = Secondary Engineer + Reviewer

# Current Goal

Determine seated posture from upper body and hip anchors only, including when
desks hide knees, legs and feet. Outputs: CORRECT_SITTING / INCORRECT_SITTING /
UNKNOWN. This is ergonomic monitoring, not medical diagnosis.

# Current Pipeline

Hikvision RTSP
→ YOLO person detection
→ ByteTrack
→ CandidateSelector
→ optional budgeted Pose V2 MediaPipe crop backend (all-box ambiguity checks)
→ experimental upper_body_posture_v1 with UNKNOWN refusal

# Current Tracking Evidence

YOLO/CUDA works on GTX 1660 SUPER.

Current detector: `yolo26n.pt`

Observed performance is approximately 14 FPS and 25–40 ms processing latency.
Natural scenes created many short tracker IDs, but confirmed human-labelled ID
switches have not yet been measured. A controlled 2-person ByteTrack benchmark
tool now exists.

# Current Gate

Upper-body candidates were trained on 2026-09-17. The five-class MultiPosture
candidate does not transfer to RGB world coordinates. The separate RGB-native
two-class Logistic Regression candidate is diagnostic only: CCTV valid poses
were all rejected as OOD. Collect small blind participant-labelled CCTV samples
before validating or changing rejection policy. See reports/upper_body_posture_v1.md.
The identity-continuity gate before Target Lock still applies.

# Important Model Direction

V4 is a frozen TUP/TLB baseline, not the final model.

Verified MultiPosture subtypes: UPRIGHT, FORWARD_LEAN, BACKWARD_OR_RECLINED,
LATERAL_LEFT, LATERAL_RIGHT. RGB-native local GT supports only UPRIGHT and
BACKWARD_OR_RECLINED. No distinct SLOUCH ground truth exists.
UPRIGHT maps to CORRECT_SITTING; verified other sitting subtypes map to
INCORRECT_SITTING. Unverified seated context, bad pose, overlap, model disagreement
or possible OOD maps to UNKNOWN. ANYA predictions/clusters and Geometry are not GT.
Only shoulders (11/12) and hips (23/24) are used in upper_body_posture_v1;
no knees, ankles, heels, feet or lower legs. Head is not required.

# Do Not Touch

- `.env` secrets
- Camera/network configuration
- Raw datasets
- Frozen V4
- Human-labelled ground truth
