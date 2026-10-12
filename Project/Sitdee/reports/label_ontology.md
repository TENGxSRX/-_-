# Label ontology

## Current supervised target

The current model predicts MultiPosture's five upper-body source labels:

| Code | Display meaning |
|---|---|
| `TUP` | Upright |
| `TLF` | Lean forward |
| `TLB` | Lean backward |
| `TLL` | Lean left |
| `TLR` | Lean right |

These are posture classes, not medical diagnoses, pain states, or ergonomic risk scores. The seven lower-body labels were preserved in the audit but not mixed into this first target.

SitPose labels (`forward`, `healthy`, `lay`, `left`, `right`, `stand`, `wrong`) are not assumed to be equivalent to the MultiPosture codes. ANYA's ontology remains unknown until its files and documentation are locally verified. Any future ergonomic-risk layer must define its own labels and evidence rather than relabeling directional posture classes as disease or injury.
