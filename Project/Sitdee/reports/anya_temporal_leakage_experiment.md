# ANYA temporal leakage experiment

Status: **DIAGNOSTIC AUDIT COMPLETE; predictive A/B comparison not run**.

The downloaded data has 32 contiguous filename blocks, frame counters that reset 31 times, and 605,910 adjacent frame increments. This is strong evidence that rows are temporally related within recording-like blocks. Exact serialized duplicate rows were zero, but temporal similarity does not require exact duplicates.

The requested naive-random versus leakage-resistant predictive comparison cannot be run honestly because there is no verified ground-truth target. Reporting a random-row score against `Predicted_Label` would measure agreement with one source model and would also mix adjacent frames across splits.

Primary future design: group all rows by `filename`, evaluate only on held-out filename groups, and explicitly state that participant independence is unavailable.

