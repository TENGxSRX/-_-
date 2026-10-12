# Failure routing และ model promotion policy

## Failure routing

ระบบแยกสาเหตุเป็นคนละชั้น: person detection, pose/core quality, tracking, activity classifier, posture classifier, subtype และ temporal instability. `POSE_LOST` หรือ `SOURCE_STALE` ไม่ถูกนับเป็น posture risk.

## Promotion gate

โมเดลใหม่ต้องมี source/license/label provenance, group integrity, frozen holdout, feature schema metadata, reload test, error analysis และ comparison กับ V4. ห้าม promote เพราะ accuracy สูงจาก random frame split.

## Deprecation

รอบนี้ไม่มีโมเดลใดถูก deprecate หรือเขียนทับ. V4, V1, V2 และ raw sources คงเดิม.
