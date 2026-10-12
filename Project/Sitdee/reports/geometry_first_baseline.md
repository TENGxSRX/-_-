# Geometry-first baseline

Temporary `StandardScaler + LogisticRegression` using `5` geometry features only. No model was exported.

- Grouped CV macro-F1: `0.9623 ± 0.0353`
- Validation macro-F1: `0.9429`
- Subject-held-out macro-F1: `0.9756`

This baseline tests whether simple geometry is competitive without depending on raw coordinate magnitudes. It is not automatically selected for deployment.
