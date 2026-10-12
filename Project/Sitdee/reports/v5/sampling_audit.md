# V5 frame sampling audit

```json
{
  "LB_1.mp4": {
    "frame_count": 2131,
    "fps": 29.998,
    "duration_seconds": 71.03806920461363,
    "sampled_at_target_fps": 72,
    "target_fps": 1.0,
    "comparison_1fps": 72,
    "comparison_2fps": 143,
    "comparison_5fps": 356
  },
  "LB_2.mp4": {
    "frame_count": 1058,
    "fps": 29.99,
    "duration_seconds": 35.27842614204735,
    "sampled_at_target_fps": 36,
    "target_fps": 1.0,
    "comparison_1fps": 36,
    "comparison_2fps": 71,
    "comparison_5fps": 177
  },
  "LB_3.mp4": {
    "frame_count": 2226,
    "fps": 29.998,
    "duration_seconds": 74.20494699646643,
    "sampled_at_target_fps": 75,
    "target_fps": 1.0,
    "comparison_1fps": 75,
    "comparison_2fps": 149,
    "comparison_5fps": 371
  },
  "SS_1.mp4": {
    "frame_count": 2131,
    "fps": 29.993,
    "duration_seconds": 71.04991164605075,
    "sampled_at_target_fps": 72,
    "target_fps": 1.0,
    "comparison_1fps": 72,
    "comparison_2fps": 143,
    "comparison_5fps": 356
  },
  "SS_2.mp4": {
    "frame_count": 2126,
    "fps": 29.993,
    "duration_seconds": 70.883206081419,
    "sampled_at_target_fps": 71,
    "target_fps": 1.0,
    "comparison_1fps": 71,
    "comparison_2fps": 142,
    "comparison_5fps": 355
  },
  "SS_3.mp4": {
    "frame_count": 2723,
    "fps": 29.984,
    "duration_seconds": 90.81510138740661,
    "sampled_at_target_fps": 91,
    "target_fps": 1.0,
    "comparison_1fps": 91,
    "comparison_2fps": 182,
    "comparison_5fps": 454
  }
}
```

The selected interval is 1 fps for reproducible anti-redundancy sampling; it was not chosen by validation accuracy.

The selected video sample count is 417 rows across six videos. The 2 fps and
5 fps counts are retained only as audit references (not as extra training
rows), so adjacent-frame redundancy is not silently introduced. The 35 MMD-MSD
images are single still-image samples and are not frame-expanded.
