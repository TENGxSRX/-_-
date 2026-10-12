# MediaPipe / YOLO stack comparison

สถานะนี้เป็น preflight ไม่ใช่ผล benchmark YOLO

- Stack A MediaPipe: **AVAILABLE_EXISTING_RUNTIME**
- Stack B YOLO person crop + MediaPipe: **BLOCKED_OPTIONAL_DEPENDENCY**
- Stack C YOLO Pose: **BLOCKED_OPTIONAL_DEPENDENCY**
- YOLO environment: `{'available': False, 'package': 'ultralytics', 'version': None, 'reason': 'ULTRALYTICS_NOT_INSTALLED'}`
- ไม่ดาวน์โหลด weight อัตโนมัติ และไม่ติดตั้ง package ในขั้นตอนนี้.
- COCO-17 keypoints ของ YOLO Pose ยังไม่ใช่ 33-landmark/world-landmark schema ของ V4 โดยอัตโนมัติ.
- การเลือก stack ต้องรอ benchmark ที่มี local weights, source เดียวกัน และ human ground truth.
