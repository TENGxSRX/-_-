# ANYA leakage analysis

Status: **COMPLETE; participant-independent evaluation is not currently provable**.

## Verified grouping evidence

The only recording-like field is `filename`. It contains 32 values of the form `sample_N_front.csv`; there are no explicit participant, subject, person, user, session, video, clip, or timestamp columns. `clusterid` and `normclusterid` are cluster IDs, not people.

Within the combined ordered table:

- 32 filename blocks and 31 filename transitions were observed.
- `frame` has 31 decreases, consistent with a reset at each recording block.
- 605,910 adjacent frame pairs increment by one.
- There are 13 positive frame jumps containing 218 missing frame numbers; no within-file frame decreases were found.
- Exact serialized duplicate rows: 0.
- A full near-duplicate distance search was not claimed; it is unnecessary to establish the already-observed temporal-block risk and would be a separate computational experiment.

## Leakage risk

Rows within a `filename` are temporally ordered frames from the same recording-like block. Randomly splitting individual rows would place highly adjacent frames from one block into both train and test. That would be a leakage-prone diagnostic only, never the primary generalization estimate.

Because participant IDs are absent, a split by `filename` would be recording-independent but **not participant-independent**. If a verified target is later supplied, the minimum defensible grouping key is `filename`; participant independence must remain “not proven” unless a mapping from filenames to participants is supplied.

## Engineering decision

No ANYA classifier was trained and no performance metric was reported because the target itself is not verified. If labels become available, use grouped splits by filename, keep rows from one recording in one split, and report the limitation that participant identity is unavailable. Use block-aware diagnostics only to quantify temporal leakage, not to select an attractive metric.

