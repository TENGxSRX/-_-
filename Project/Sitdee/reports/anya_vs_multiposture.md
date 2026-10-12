# ANYA versus MultiPosture

| Attribute | ANYA | MultiPosture |
|---|---|---|
| Intended purpose | Ergonomic posture analysis | Sitting-posture direction classification |
| Local samples | 605,955 rows in each of four identical-core files | 4,794 rows |
| Participants | Not present as an explicit field; 32 recording-like filenames | 13 subject IDs |
| Source modality | Pose-derived CSV release; original image/video not in the archive | 33-landmark XYZ CSV |
| Landmark slots | 33 raw indexed slots; separate 13-slot normalized block | 33 named MediaPipe landmarks |
| Landmark identities | Not supplied in the CSV; anatomy/order unverified | Named and mapped in the project |
| Coordinates | Raw x/y/z plus visibility; separate normalized-looking x/y/z | Source XYZ, project normalization/geometry tracks |
| Labels | No verified ground-truth target; `Predicted_Label` is model output; cluster IDs are not targets | Five upper-body direction labels and seven lower-body labels |
| Grouping | `filename` only; participant independence not proven | `subject` available; grouped split used |
| Leakage risk | High if rows are split randomly because frame blocks are temporal | Reduced with subject grouping; still only 13 subjects |
| Ergonomic relevance | Intended by source context, but no validated local target | Directional posture relevance, not a medical/ergonomic diagnosis |
| Real-camera compatibility | Not validated; index mapping and transform unknown | Offline adapter implemented; exact source normalization equivalence not proven |
| License | Apache-2.0 from Kaggle metadata | CC BY 4.0 from Zenodo record |
| Current model performance | None honestly reportable | Grouped-CV macro-F1 `0.9891 ± 0.0060`; held-out subject test macro-F1 `1.0000` with only 2 test subjects |

## Common adapter decision

A common adapter is **not proven feasible**. Both releases contain 33 pose slots, but ANYA does not provide anatomical names, index order, normalization formula, or target semantics. The existing MultiPosture adapter must not be applied to ANYA by position alone. The two-model architecture remains conceptual until a source-backed mapping and a controlled cross-dataset adapter test are available.

