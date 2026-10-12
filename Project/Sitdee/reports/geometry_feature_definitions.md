# Geometry feature definitions

These are deterministic geometric indicators, not clinically validated rules.
All coordinates come from the active pose backend. Angles use the image-plane
x/y axes; the deployment camera orientation matters.

Let `S_L`, `S_R` be shoulder points, `H_L`, `H_R` hip points, `N` the nose,
`S=(S_L+S_R)/2`, `H=(H_L+H_R)/2`, `v=S-H`, and `L=||v||`.

| Feature | Landmarks | Equation / units | Interpretation | Limitation |
|---|---|---|---|---|
| `trunk_inclination_deg` | shoulders, hips | `atan2(v_x, abs(v_y))`, degrees | Signed torso lean relative to image vertical | Camera viewpoint and seated orientation affect it; not a clinical flexion measure |
| `shoulder_tilt_deg` | left/right shoulders | `atan2(S_R,y-S_L,y, S_R,x-S_L,x)`, degrees | Shoulder-line tilt in image plane | Does not establish shoulder pathology or clinical imbalance |
| `hip_tilt_deg` | left/right hips | Same line-angle equation, degrees | Hip-line tilt in image plane | Camera roll and landmark noise affect it |
| `torso_lateral_displacement` | shoulder/hip midpoints | `v_x / L`, dimensionless | Lateral shoulder displacement from hip midpoint | Not a validated lateral-load measure |
| `torso_length` | shoulder/hip midpoints | `L`, backend units | Scale reference for ratios | Backend coordinate scale may vary |
| `head_to_shoulder_x/y/z` | nose, shoulder midpoint | `(N - S) / L`, dimensionless | Relative head offset | Nose is a proxy point; z sign and camera geometry are backend-dependent |
| `head_forward_proxy` | nose, shoulder midpoint | `head_to_shoulder_z` | Camera-relative depth proxy | Experimental; not clinically validated forward-head posture |
| `shoulder_height_asymmetry` | shoulders | `(S_L,y-S_R,y)/L` | Relative left/right shoulder height | Viewpoint and pose rotation confound it |
| `hip_height_asymmetry` | hips | `(H_L,y-H_R,y)/L` | Relative left/right hip height | Viewpoint and landmark noise confound it |
| `shoulder_to_hip_length_asymmetry` | shoulders, hips | `(||S_L-H||-||S_R-H||)/L` | Side-to-side torso geometry difference | Not a clinical asymmetry score |

## Indicator policy

The trained model's direction labels (`forward_trunk`, `backward_trunk`,
`left_lateral_trunk`, `right_lateral_trunk`) are core model states. The
following are explicitly experimental: `head_forward_proxy`,
`shoulder_tilt_imbalance`, and `slouch_proxy`. Their thresholds in
`configs/posture_engine.yaml` are project settings for observation/warning
behavior only.

No C7 landmark is invented, and no claim of clinically confirmed forward-head,
slouch, rounded shoulder, or imbalance posture is made.

