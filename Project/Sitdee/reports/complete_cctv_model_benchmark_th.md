# Complete CCTV model benchmark gate

## ผลที่มีจริง

- Existing frozen V4: HistGradient/old V4 artifacts ตามโมเดลเดิม; ไม่ retrain ในรอบนี้
- CCTV activity model: not trained
- sitting correct/incorrect model: not trained
- subtype model: not trained
- YOLO stack B/C: not benchmarked; optional dependency/weights unavailable

## เหตุผลที่ยังไม่ benchmark model ใหม่

`data/controlled_cctv_v2/samples.csv` ยังไม่มี human-labelled samples. Readiness audit จึงคืน `NOT_READY_FOR_CCTV_V5_TRAINING`. การสร้าง score จาก V4 prediction หรือ Geometry จะเป็น target leakage.

## แผน benchmark เมื่อข้อมูลพร้อม

ใช้ Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting และ GradientBoosting บน canonical schema ที่ frozen แล้ว. รายงาน accuracy, balanced accuracy, macro-F1, sensitivity/specificity และ calibration โดยใช้ participant/session/video grouped CV และ final holdout ที่ไม่ overlap.

จนกว่าจะถึงจุดนั้น ให้รายงาน software readiness แยกจาก scientific model readiness.
