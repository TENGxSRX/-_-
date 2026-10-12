# V5 leave-one-dataset-out

## Experiment result

**NOT RUN.** Only two direction classes have defensible supervision: TUP and
TLB. TLF, TLL, and TLR have zero source-backed rows. A leave-one-dataset-out
score would therefore either omit required classes or measure source/camera
identity instead of five-class posture generalization.

## Engineering decision

Freeze this experiment until at least two independently traceable RGB sources
cover each project direction, with participant/session/video grouping metadata.
No placeholder scores are reported.
