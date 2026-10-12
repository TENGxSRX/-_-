# V5 RGB cross-dataset generalization

## Experiment result

**NOT RUN for classifier metrics.** The available sources do not provide a
leakage-safe five-class corpus with joinable labels. There is no valid V5
prediction or accuracy claim yet.

## Data decision

MVSP/LGCSPNet is the most promising missing-class candidate because its
published categories include upright, back, down, left, and right. It remains
excluded until the downloadable archive and license are verified. CCTV and
SitStraightNet remain audit/smoke sources because the local per-image label
join is unavailable. The next run should use leave-one-dataset-out only after
TLF/TLL/TLR labels and group identifiers are present.
