# V4 vs V5

| Attribute | V4 | V5 current state |
|---|---|---|
| Purpose | Frozen RGB-native two-class pilot | RGB-native multi-class expansion gate |
| Sources | DriveNet v6 | DriveNet v6 + MMD-MSD public subset |
| Sampled rows | 1,800 in the V4 experiment | 452 in the prepared manifest |
| Labels | TUP/TLB | TUP 244, TLB 183; 25 audit-only/ambiguous rows |
| Core landmarks | Five | Same five, by contract |
| Representation | MediaPipe World Landmarks, V4 adapter | Same V4-compatible extraction, no schema change |
| Model | Frozen V4 artifact | Not trained; missing TLF/TLL/TLR |
| Evaluation | Existing V4 grouped result remains unchanged | CV/holdout/LODO not run |
| Isolation | `outputs/models/rgb_native_v4/` | `outputs/models/rgb_native_v5/NOT_TRAINED.md` |

V4 remains unchanged and is not replaced. V5 currently provides a traceable
multi-source manifest and extracted feature arrays, but no classifier artifact.
The gate is intentional: the available RGB sources do not yet provide all five
direction classes with defensible labels and independent groups.
