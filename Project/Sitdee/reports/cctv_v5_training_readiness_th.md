# CCTV V5 training readiness audit

สถานะ: **NOT_READY_FOR_CCTV_V5_TRAINING**

เอกสารนี้เป็น readiness gate เท่านั้น ไม่ train V5 และไม่เปลี่ยน V4

- samples: 0 | sessions: 0 | participants: 0
- labels: `{}`
- expected people values: `[]`
- explicit positions: `[]`
- grouping: `PARTICIPANT_INDEPENDENT_SUBJECT_GROUPED; no random frame split`

## Blockers

- NO_HUMAN_LABELLED_CCTV_V2_SAMPLES

## Metrics status

- detection: `NO_GROUND_TRUTH_FRAMES`
- target core: `NO_GROUND_TRUTH_PERSON_ROWS`
- frozen V4: `NO_VALID_LABELLED_V4_ROWS`
- temporal: `NO_LABELLED_TEMPORAL_SEQUENCE`
- tracking: `NO_TRACK_EVENTS`

## Integrity rules

- human labels must be selected by an operator
- V4 prediction, Geometry and Temporal are diagnostic fields only
- no random frame split; use participant/group split
- do not start V5 until this report is READY and a separate training review approves it
