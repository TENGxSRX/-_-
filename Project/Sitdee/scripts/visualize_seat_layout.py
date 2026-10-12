"""Draw a normalized seat layout on a reference image or TEST_FIXTURE canvas.

No camera is opened.  With no --image, the generated canvas is explicitly a
synthetic TEST_FIXTURE and must not be treated as a real-room calibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.multiperson.seating import draw_seat_layout, load_seat_layout  # noqa: E402


def _fixture_canvas(width: int = 960, height: int = 540):
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    cv2.putText(canvas, "TEST_FIXTURE ONLY — no real-room geometry", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (60, 90), (900, 495), (215, 215, 215), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, default=ROOT / "configs" / "seat_layout_example.yaml")
    parser.add_argument("--image", type=Path, help="Reference image file; omitted means synthetic TEST_FIXTURE")
    parser.add_argument("--output", type=Path, default=ROOT / "fixtures" / "seat_layout_example.png")
    args = parser.parse_args()
    camera_id, zones = load_seat_layout(args.layout)
    if args.image is None:
        frame = _fixture_canvas()
    else:
        frame = cv2.imread(str(args.image))
        if frame is None:
            parser.error(f"could not read image: {args.image}")
    output = draw_seat_layout(frame, zones)
    if not cv2.imwrite(str(args.output), output):
        raise OSError(f"could not write visualization: {args.output}")
    print(f"Wrote {len(zones)} seats for {camera_id} to {args.output}")
    if args.image is None:
        print("status=TEST_FIXTURE_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
