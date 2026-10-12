# External RGB validation-media search and verification

## Scope and decision

This is a small external validation/smoke pack. It does not retrain either model and does not modify `datasets/raw/`. DriveNet v6 was selected because its official Mendeley record provides RGB posture videos, explicit source classes, participant/video metadata, and CC BY 4.0. Only `SS → TUP` and `LB → TLB` are used as project labels. `MB` remains `UNLABELED` because “Move Backward” is not a safe project-ontology mapping.

The source videos are retained unchanged. The runtime benchmark processes only the first 300 frames (about 10 seconds) per clip to keep the validation run small; this is a processing limit, not a source trim.

## Source candidates reviewed

| Candidate | Decision | Reason |
|---|---|---|
| DriveNet v6 (Mendeley Data) | SELECTED | RGB MP4 source videos; four explicit classes; participant/video metadata; CC BY 4.0 |
| HuMoLs (University of Edinburgh) | NOT INCLUDED | Useful sitting RGB videos, but the page says “used freely” without a standard exact license expression; kept as a research lead, not a reusable pack |
| SitStraightNet local repository | SMOKE ONLY / EXISTING | 756 images; no verified posture labels or sidecars; prior run had 0/756 MediaPipe detections, so no accuracy |
| SitPose | NOT RGB VALIDATION | Depth/Azure Kinect dataset; unsuitable for ordinary RGB media validation |
| MultiPosture | NOT MEDIA | Landmark CSV source for the trained direction model; no RGB validation media in the project source |

## Downloaded source table

| source_file | verified_label | source_class | participant | source_frames | source_duration_s | bytes | SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| SS_1.mp4 | TUP | SS | 1 | 2131 | 71.04991164605075 | 153589862 | f4161e3bb9faf61831422e0d88c253f9746423a86be1deac8f997212a4426f79 |
| LB_2.mp4 | TLB | LB | 2 | 1058 | 35.27842614204735 | 77562475 | baa2dbb0c1d4dec761389ab4941114e27c61928ac8132c6eae0abcd84ce677ac |
| MB_1.mp4 | UNLABELED | MB | 1 | 2074 | 69.14024735806915 | 156608940 | 51c8b6790f0094672779fed1fa7e3cadeea1911b04e5256ceff74fe6ef40c34f |

Total local media size: **387761277 bytes** (369.80 MiB).

## Required verification points

1. **Trustworthy source:** DriveNet v6 official Mendeley Data record, DOI `10.17632/v37nxdn2gh.6`, source URL `https://data.mendeley.com/datasets/v37nxdn2gh/6`.
2. **RGB and sitting content:** visual inspection confirms ordinary color video of a person seated in a vehicle; this is a driving-seat domain, not an office desk domain.
3. **Labels:** `SS` and `LB` are explicit source classes. `SS → TUP` and `LB → TLB` are the only semantic mappings used. `MB` is not mapped.
4. **License:** source record and README state **CC BY 4.0**. Reuse requires attribution and a link to the license; this project retains the source name, URL, and exact hashes.
5. **Integrity:** local file hashes match the source file hashes: **True**.
6. **Scope:** three clips were downloaded: two labeled clips and one smoke-only clip. The source class `RP` was not downloaded because it was not needed to demonstrate the pipeline and its project mapping is ambiguous.
7. **Media properties:** the source videos are 1920×1440 MP4 files at about 30 FPS, with the exact per-file values in the table above.
8. **MediaPipe pre-screen:** pose detection succeeded on **900/900** processed frames (**all_detected=True**).
9. **Adapter/classifier gate:** the safe 33-landmark adapter accepted **0** frames at visibility ≥ 0.5 (**all_adapter_zero=True**). Low-visibility legs/feet from the side view caused rejection; therefore no external classifier accuracy is reported.
10. **Engineering decision:** this pack is sufficient for offline RGB pose-extraction and failure-mode smoke testing, but **not sufficient to validate real-image MultiPosture classification**. A future camera-view-compatible, full-landmark sample or manually recorded five-posture set is still needed for adapter validation.

## Observed quality and limitations

- The videos contain real RGB frames and visible seated people, but their camera setup is a side/three-quarter vehicle view rather than the intended office-camera view.
- MediaPipe produced landmarks, but several lower-limb landmarks were below the safety visibility threshold. Feeding those low-confidence values into the 126-feature classifier would be an invalid experiment.
- Because the adapter accepted no frames, there are no labeled accuracy, confusion-matrix, confidence, or domain-generalization claims from this external pack.
- The annotated outputs show skeletons and the verified source label for the two labeled clips; the smoke clip is annotated as `UNLABELED`.
- No webcam device was opened and no `--source 0` run was attempted.

## Artifact locations

- Manifest: `validation_data/external/external_manifest.csv`
- Source metadata: `validation_data/external/source_metadata/DriveNet_v6/`
- Annotated videos: `outputs/validation/external/videos/`
- Per-video telemetry: `outputs/validation/external/metrics/`, `outputs/validation/external/summaries/`, and `outputs/validation/external/logs/`
- Machine-readable audit: `outputs/validation/external/external_media_audit.json`

## Status

- **Labeled external accuracy:** NOT AVAILABLE — safe adapter had zero eligible frames.
- **Unlabeled smoke:** COMPLETED for one DriveNet clip; MediaPipe detection passed, adapter gate exposed the side-view/full-landmark limitation.
- **Real-image adapter:** NOT VALIDATED.
- **Live webcam:** NOT TESTED — hardware not currently available.
