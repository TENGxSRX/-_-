# ANYA model comparison

Status: **NOT RUN — no verified supervised target**.

The four CSVs are model-specific prediction outputs over one shared core observation sequence. Reproducing `ANN.csv` with an ANN, `Random_Forest.csv` with a random forest, `lgbm.csv` with LightGBM, or `xgb.csv` with XGBoost would be a prediction-reproduction/distillation exercise, not a validated ergonomic classifier. The files disagree on 83,370–117,555 rows for several pairs, so no one prediction column is a defensible ground truth.

The planned fair benchmark remains Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting, GradientBoostingClassifier, and a small MLP only if justified. XGBoost/LightGBM may be included when a verified target is supplied. All tuning must use filename-grouped CV and must not use the final test groups.

No ANYA model, metric, calibration result, confusion matrix, or `outputs/models/anya_v1/` export was created.

