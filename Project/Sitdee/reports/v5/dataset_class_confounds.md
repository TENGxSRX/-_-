# V5 dataset/class confound audit

Generated: 2026-09-05T10:51:46.710185+00:00

## Project-labelled coverage

| Project label | Participants | Videos/sessions | Manifest rows |
|---|---:|---:|---:|
| TLB | 3 | 3 | 183 |
| TUP | 8 | 8 | 244 |

## Confound test

```json
{
  "by_source": {
    "DriveNet_v6": [
      "TLB",
      "TUP"
    ],
    "MMD-MSD_public_5_participant_subset": [
      "TUP"
    ]
  },
  "by_label": {
    "TLB": [
      "DriveNet_v6"
    ],
    "TUP": [
      "DriveNet_v6",
      "MMD-MSD_public_5_participant_subset"
    ]
  },
  "one_label_per_source": false,
  "one_source_per_label": false,
  "severe": false
}
```

The current usable labels are unevenly distributed across source identity: TUP
is represented by DriveNet SS plus MMD-MSD corrected/active sitting, while TLB
is represented only by DriveNet LB. The exact boolean `severe` criterion above
is false because the two available sources are not one-to-one with the two
labels; this does not make the dataset sufficient for the intended five-class
experiment. There are no TLF/TLL/TLR samples, so source and posture effects
cannot be separated for those classes. A five-class V5 model must not be
trained from this manifest.

The row-level random split is not used.  Future V5 training must keep participant, session, and video groups intact and must include at least one independent source or participant for each direction.

## Training-readiness coverage

| Class | Participants/groups | Videos/sessions | Samples | Datasets | Status |
|---|---:|---:|---:|---:|---|
| TUP | 8 | 8 | 244 | 2 | Available, but MMD participant mapping is protocol-inferred |
| TLB | 3 | 3 | 183 | 1 | Available only from DriveNet |
| TLF | 0 | 0 | 0 | 0 | Missing |
| TLL | 0 | 0 | 0 | 0 | Missing |
| TLR | 0 | 0 | 0 | 0 | Missing |

## Acquisition confound risk

Adding MVSP only for TLF/TLL/TLR would create a near-direct relationship
between dataset identity and the missing project labels: existing DriveNet/MMD
rows would provide TUP/TLB while MVSP would provide the three new directions.
This is a source-domain confound even if every MVSP label is individually
correct. The next acquisition must include TUP/TLB from the new source or
missing directions from more than one source.

## Engineering decision

All five classes are not represented, so the V5 training gate remains closed.
Participant/video grouping is possible for the existing rows, but a valid
five-class grouped evaluation is not possible until missing-class groups are
acquired and their labels/license are verified.
