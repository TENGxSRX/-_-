# Verified MultiPosture landmark mapping

## Measured local schema

The actual local CSV has **102 columns**: `subject`, two labels, and **99 coordinate columns**. The coordinate names form 33 groups of `x/y/z` using the canonical MediaPipe landmark names. This is an observed file fact from `C:\Users\ADMIN\Documents\ChatGPT\AnyaTrainModel\office-posture-datasets\datasets\raw\multiposture\data.csv`.

The Zenodo metadata stored locally describes “11 key body joints” and “33 input dimensions,” which conflicts with the actual 99 coordinate columns. Therefore this report does not silently replace the measured schema with the prose count. The local column names support a 33-slot name/index mapping, but they do not independently prove how every slot was generated.

## Mapping table

`EXACT_MATCH` below means exact column-group name/index identity with the canonical MediaPipe name. It does **not** mean that source camera scale, projection, z-axis convention, visibility, or model-version semantics have been validated.

| MultiPosture feature group | Anatomical landmark by column name | MediaPipe equivalent | Status | Evidence/limitation |
|---|---|---|---|---|
| `nose_x, nose_y, nose_z` | nose | nose | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_eye_inner_x, left_eye_inner_y, left_eye_inner_z` | left_eye_inner | left_eye_inner | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_eye_x, left_eye_y, left_eye_z` | left_eye | left_eye | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_eye_outer_x, left_eye_outer_y, left_eye_outer_z` | left_eye_outer | left_eye_outer | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_eye_inner_x, right_eye_inner_y, right_eye_inner_z` | right_eye_inner | right_eye_inner | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_eye_x, right_eye_y, right_eye_z` | right_eye | right_eye | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_eye_outer_x, right_eye_outer_y, right_eye_outer_z` | right_eye_outer | right_eye_outer | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_ear_x, left_ear_y, left_ear_z` | left_ear | left_ear | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_ear_x, right_ear_y, right_ear_z` | right_ear | right_ear | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `mouth_left_x, mouth_left_y, mouth_left_z` | mouth_left | mouth_left | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `mouth_right_x, mouth_right_y, mouth_right_z` | mouth_right | mouth_right | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_shoulder_x, left_shoulder_y, left_shoulder_z` | left_shoulder | left_shoulder | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_shoulder_x, right_shoulder_y, right_shoulder_z` | right_shoulder | right_shoulder | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_elbow_x, left_elbow_y, left_elbow_z` | left_elbow | left_elbow | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_elbow_x, right_elbow_y, right_elbow_z` | right_elbow | right_elbow | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_wrist_x, left_wrist_y, left_wrist_z` | left_wrist | left_wrist | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_wrist_x, right_wrist_y, right_wrist_z` | right_wrist | right_wrist | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_pinky_x, left_pinky_y, left_pinky_z` | left_pinky | left_pinky | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_pinky_x, right_pinky_y, right_pinky_z` | right_pinky | right_pinky | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_index_x, left_index_y, left_index_z` | left_index | left_index | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_index_x, right_index_y, right_index_z` | right_index | right_index | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_thumb_x, left_thumb_y, left_thumb_z` | left_thumb | left_thumb | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_thumb_x, right_thumb_y, right_thumb_z` | right_thumb | right_thumb | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_hip_x, left_hip_y, left_hip_z` | left_hip | left_hip | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_hip_x, right_hip_y, right_hip_z` | right_hip | right_hip | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_knee_x, left_knee_y, left_knee_z` | left_knee | left_knee | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_knee_x, right_knee_y, right_knee_z` | right_knee | right_knee | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_ankle_x, left_ankle_y, left_ankle_z` | left_ankle | left_ankle | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_ankle_x, right_ankle_y, right_ankle_z` | right_ankle | right_ankle | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_heel_x, left_heel_y, left_heel_z` | left_heel | left_heel | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_heel_x, right_heel_y, right_heel_z` | right_heel | right_heel | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `left_foot_index_x, left_foot_index_y, left_foot_index_z` | left_foot_index | left_foot_index | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |
| `right_foot_index_x, right_foot_index_y, right_foot_index_z` | right_foot_index | right_foot_index | EXACT_MATCH | actual local column group; name/index match only, coordinate provenance still caveated |

## Source facts and engineering decision

- Source metadata says coordinates are normalized relative to hip center and were extracted using MediaPipe Pose Heavy; the actual CSV values are finite and the local loader confirms the exact 102-column order.
- The CSV has explicit `subject` IDs and upper/lower labels; V2 uses only `upperbody_label`.
- The disagreement between the “11 joints” prose and the 33 named coordinate groups means full semantic provenance is **UNKNOWN** even though the name-level mapping is exact.
- V2 is therefore an experimental feasibility track, not proof of real-camera semantic compatibility. V1 remains unchanged.
