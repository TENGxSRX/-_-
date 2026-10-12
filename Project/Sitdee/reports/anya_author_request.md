# Draft request to ANYA authors

Status: **DRAFT ONLY — NOT SENT**

Suggested contacts from the checked paper/repository: `trakanjai@gmail.com`
and `cjundang@gmail.com`.

## Subject

Request for the expert-labeled ground-truth subset or cluster mapping for the ANYA dataset

## Email draft

Dear ANYA authors,

I am working on an academic office-posture machine-learning project and am
carefully auditing the official ANYA release before training an ergonomic
posture classifier. I downloaded the official Kaggle dataset
(`chanankornjandaeng/anya-a-dataset-for-ergonomic-posture-analysis`) and read
the paper, “ANYA: A Noteworthy Youth Annotation Dataset for Machine
Learning-Based Ergonomic Posture Analysis” (DOI:
10.1109/ACCESS.2025.3633624).

The paper describes an expert-labeled subset from the first 10 recordings,
reported as 144,000 frames with 60,181 ergonomically correct and 83,819
ergonomically incorrect observations, followed by pseudo-label propagation
using ANN, Random Forest, LightGBM, and XGBoost. The paper also describes a
cluster-to-`TRUE`/`FALSE` expert-label mapping.

In the current Kaggle release, I find four CSV files (`ANN.csv`,
`Random_Forest.csv`, `lgbm.csv`, and `xgb.csv`). They contain the same
605,955-row feature/metadata core and different `Predicted_Label` columns, but
I cannot find a verified `TRUE`/`FALSE` target, the original expert-labeled
subset, or a joinable mapping for `clusterid`/`normclusterid`. I do not want to
mistake model pseudo-labels or cluster identifiers for expert ground truth.

Could you please clarify whether one of the following is available?

1. The original expert-labeled 144,000-row subset, with its exact file/row
   identifiers;
2. A mapping from `filename` + frame and/or `clusterid`/`normclusterid` to the
   expert labels `TRUE`/`FALSE`;
3. An authoritative manifest identifying the initial 10 recordings;
4. Documentation explaining the exact meaning of `clusterid`,
   `normclusterid`, and the numeric values in `Predicted_Label`;
5. A historical or supplementary release showing how the reported 144,000
   expert-labeled observations relate to the current 605,955-row Kaggle files.

If this artifact is available, I would be grateful for the preferred download
location, citation, and any applicable license or usage conditions. I will
preserve the raw Kaggle files unchanged and will clearly distinguish expert
labels from pseudo-labels in any report.

Thank you for your time and for making the ANYA work available.

Best regards,

[Your name]
[Institution / course]
[Project or repository link, if applicable]

