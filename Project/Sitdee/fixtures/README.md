# Development fixtures

Files in this folder are sanitized `TEST_FIXTURE` inputs for UI, summary,
serialization, and documentation work. They are not labels, ground truth,
participant data, or scientific evaluation results.

No raw video is copied into the Acer package. Use a saved session summary or
the JSON fixture for dashboard development; keep source recordings on Desktop.

`test_room_spec.json` is also a sanitized `TEST_ROOM_ONLY` geometry fixture.
Its dimensions, seats, cameras and obstacle are invented test values and must
never be presented as the real room plan.
