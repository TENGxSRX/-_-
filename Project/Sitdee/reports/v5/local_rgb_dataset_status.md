# Local RGB V1 dataset status

Status: **WAITING_FOR_LOCAL_RGB_DATA**

## Collection inventory

- Participants: **0 / 10** with videos
- Sessions: **0 / 20** with videos
- Videos: **0 / 100**
- Usable frames after pose pre-flight: **0**
- Camera views observed: `none yet`

| Posture | Videos present | Independent participants |
|---|---:|---:|
| TUP | 0 / 20 | 0 |
| TLF | 0 / 20 | 0 |
| TLB | 0 / 20 | 0 |
| TLL | 0 / 20 | 0 |
| TLR | 0 / 20 | 0 |

## Quality summary

- Pose detection: `{'mean': None, 'minimum': None}`
- Five-core acceptance: `{'mean': None, 'minimum': None}`
- Duration: `{'minimum': None, 'mean': None, 'maximum': None}`
- Raw media modification by validation: `False`

## Local V5 gate

- `all_five_classes_have_video`: WAITING/FAIL
- `each_class_has_three_participants`: WAITING/FAIL
- `at_least_six_participants`: WAITING/FAIL
- `participant_ids_available`: WAITING/FAIL
- `grouped_split_possible`: WAITING/FAIL
- `pose_detection_acceptable`: WAITING/FAIL
- `core_acceptance_acceptable`: WAITING/FAIL
- `raw_hashes_recorded`: PASS
- `source_labels_missing`: WAITING/FAIL
- `obvious_confounding`: WAITING/FAIL

Training is intentionally **not performed** by this collection system. The
first future experiment must be LOCAL ONLY, with participant-grouped splitting;
public data may be compared later after the local-only baseline.
