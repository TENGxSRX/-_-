# V5 targeted missing-class data search

Date: 2026-09-05

## Decision

**V5_PUBLIC_DATA_INSUFFICIENT.** No new source was admitted to supervised V5
training in this phase. V5 was not trained, no label was inferred visually,
and no pseudo-label was created.

The current verified supervised RGB-native coverage remains:

| Project class | Samples | Independent participants/groups | Datasets | Videos/sessions |
|---|---:|---:|---:|---:|
| TUP | 244 | 8 participants (5 MMD IDs are protocol-based engineering IDs) | 2 | 3 videos + 5 image sessions |
| TLB | 183 | 3 | 1 | 3 videos |
| TLF | 0 | 0 | 0 | 0 |
| TLL | 0 | 0 | 0 | 0 |
| TLR | 0 | 0 | 0 | 0 |

## MVSP/LGCSPNet — highest-priority investigation

### Source facts

The related paper describes a multi-view sitting posture dataset with 7
posture categories, 7 fixed viewpoints and 1 dynamic viewpoint, and reports
49,798 RGB images. The official repository currently contains only a README
that points to a Baidu Pan download link; it does not publish a license file,
annotation manifest, participant table, or archive listing.

Sources: [official repository](https://github.com/LKW123800/multi-view-sitting-posture-dataset),
[related paper](https://www.sciencedirect.com/science/article/pii/S0957417425021050),
[paper PDF](https://pure.hw.ac.uk/ws/portalfiles/portal/151801017/LGCSPNet_posture_management.pdf).

### Published class semantics

| MVSP class | Published meaning | V5 mapping decision |
|---|---|---|
| `upright` | relax and sit upright/straight | COMPATIBLE with TUP at source-semantics level |
| `back` | lean back | COMPATIBLE with TLB at source-semantics level |
| `down` | lean forward and bend over | AMBIGUOUS for TLF; compound “bend over” is not automatically identical to trunk-only forward lean |
| `left` | body leans to the left | COMPATIBLE with TLL at source-semantics level |
| `right` | body leans to the right | COMPATIBLE with TLR at source-semantics level |
| `cheek` | chin/head supported by a hand | INCOMPATIBLE with a direction label |
| `leg` | one leg crossed over the other | INCOMPATIBLE with an upper-trunk direction label |

These are source-level semantics only. They are not admitted V5 samples because
the archive, per-sample label join, participant IDs, and license have not been
verified locally. In particular, `down` is not forced into TLF.

### Access result

MVSP was **not obtained**. The official README points to:

`https://pan.baidu.com/s/1577RDm5dmdXtQqNUjwJusw?pwd=qm7x`

The link cannot be verified as an archive from the available access path. If a
human has legitimate access, the safe procedure is: open the official README,
use the Baidu link normally, complete any required account/CAPTCHA step, save
the original archive under `datasets/raw/rgb_native_v5/mvsp/`, preserve its
filename, compute SHA-256, inspect the archive manifest and labels, and only
then request a new V5 audit. No CAPTCHA or access control was bypassed.

## Other credible candidates

| Dataset | RGB | TLF | TLL | TLR | Participants | IDs / grouping | License | Public access | Decision |
|---|---|---|---|---|---:|---|---|---|---|
| MVSP/LGCSPNet | Yes | AMBIGUOUS (`down`) | COMPATIBLE (`left`) | COMPATIBLE (`right`) | Not verified locally | Not verified | LICENSE_UNCLEAR | Baidu manual link | Not admitted |
| CCTV Exam Monitor | Yes | AMBIGUOUS (`ForwardMove`) | AMBIGUOUS (`LeftSideMove`) | AMBIGUOUS (`RightSideMove`) | Not available locally | No per-image join in local archive | LICENSE_UNCLEAR | Kaggle archive | Audit only |
| SitStraightNet | RGB-D/RGB snapshot | AMBIGUOUS (`hunched over`) | No exact class | No exact class | 11 in paper, not locally joined | No per-file label sidecar | LICENSE_UNCLEAR for image data | GitHub snapshot | Smoke only |
| MMD-MSD public subset | Yes | No | No | No | 5 public-subset participants | Ordered 7-image block mapping inferred | Source/paper terms recorded | Public Google Drive subset | TUP diversity only |
| DriveNet v6 | Yes | No | No | No | 3 | Explicit participant/video metadata | CC BY 4.0 | Mendeley Data | Existing TUP/TLB only |
| MultiPosture | No raw RGB | N/A | N/A | N/A | 13 | Processed keypoints only | Dataset record terms | Zenodo | Cannot supply RGB |
| Smart-cushion posture dataset | No | Sensor label exists | Sensor label exists | Sensor label exists | 12 | Pressure recordings | Not an RGB source | Paper data | Rejected: pressure-only |

Source references: [CCTV Exam Monitor](https://www.kaggle.com/datasets/cctvdataset/cctv-exam-monitor-dataset),
[SitStraightNet repository](https://github.com/realratchet/SitStraightNet),
[SitStraightNet paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8022631/),
[MMD-MSD official resource](http://isr.tu-varna.bg/ergo/index.php/resursi),
[DriveNet v6](https://data.mendeley.com/datasets/v37nxdn2gh/6),
[MultiPosture Zenodo record](https://zenodo.org/records/14230872),
[smart-cushion study](https://pmc.ncbi.nlm.nih.gov/articles/PMC5421679/).

## Local audit result

No new media was downloaded in this phase. The V5 manifest remains at 452
rows. Existing V5 source extraction remains 452/452 pose-detected and core
feature-eligible. The current C: free space is approximately 19.97 GB; no
download larger than 2 GB was started because MVSP access and license were not
resolved.

## Engineering decision

Public RGB data is insufficient for a scientifically defensible five-class V5
model. A download alone would not solve the source-label confound: adding only
MVSP missing-class images would make dataset identity strongly predict TLF/TLL/
TLR. The next valid acquisition must include missing classes plus TUP/TLB
coverage, or a local multi-participant RGB collection with explicit labels and
group metadata.
