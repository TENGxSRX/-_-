# Canonical CCTV feature schema

Schema: `cctv_canonical_v1`

## รูปแบบ

- input: 33 landmarks × XYZ ที่ระบุ coordinate type ชัดเจน (`mediapipe_world` หรือ `mediapipe_image_normalized`)
- selected landmarks: nose, shoulders, hips, knees, ankles รวม 9 จุด / 27 ค่า
- translation: ลบ hip midpoint
- scale: `none` หรือ `torso_length` แบบ explicit
- rotation: ยังไม่ทำ (`none`)
- missing/non-finite input: reject ไม่เติมค่าเงียบ ๆ

## ข้อจำกัด

schema นี้เป็น common representation สำหรับการทดลองในอนาคต ไม่ใช่หลักฐานว่า MediaPipe image/world output เทียบกับ MultiPosture/V4 ได้. `v4_compatible=False` ถูกบันทึกไว้ใน metadata และ viewpoint invariance เป็น `NOT_PROVEN`.
