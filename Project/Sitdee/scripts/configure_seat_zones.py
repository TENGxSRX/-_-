"""Create a normalized seat-layout YAML from an image file.

Examples:
  python scripts/configure_seat_zones.py --image room.jpg --seat SEAT_01:40,80,250,430 --seat SEAT_02:280,80,490,430
  python scripts/configure_seat_zones.py --image room.jpg --interactive --count 3

This tool only opens the supplied image file.  It never opens a camera.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.multiperson.seating import SeatZone  # noqa: E402


def _pixel_box(value: str, width: int, height: int) -> tuple[str, list[float]]:
    try:
        seat_id, raw_box = value.split(":", 1)
        left, top, right, bottom = (float(item) for item in raw_box.split(","))
    except ValueError as exc:
        raise ValueError("seat must use SEAT_01:left,top,right,bottom") from exc
    if not (0 <= left <= right <= width and 0 <= top <= bottom <= height):
        raise ValueError("pixel seat rectangle must be inside the image")
    return seat_id, [left / width, top / height, right / width, bottom / height]


def _interactive_boxes(image, count: int) -> list[tuple[str, list[float]]]:
    if count < 1:
        raise ValueError("count must be at least 1")
    height, width = image.shape[:2]
    points: list[tuple[int, int]] = []
    window = "Configure seat zones — click two corners, press N for next seat, Q to cancel"
    display = image.copy()

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    boxes: list[tuple[str, list[float]]] = []
    try:
        for index in range(count):
            points.clear()
            while True:
                display = image.copy()
                for point in points:
                    cv2.circle(display, point, 5, (0, 255, 0), -1, cv2.LINE_AA)
                if len(points) == 2:
                    left, top = points[0]
                    right, bottom = points[1]
                    cv2.rectangle(display, (min(left, right), min(top, bottom)), (max(left, right), max(top, bottom)), (0, 255, 0), 2)
                cv2.putText(display, f"SEAT_{index + 1:02d}: click top-left and bottom-right, then N", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(window, display)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    raise KeyboardInterrupt
                if key in (ord("n"), ord("N")) and len(points) == 2:
                    left, top = points[0]
                    right, bottom = points[1]
                    boxes.append((f"SEAT_{index + 1:02d}", [min(left, right) / width, min(top, bottom) / height, max(left, right) / width, max(top, bottom) / height]))
                    break
    finally:
        cv2.destroyWindow(window)
    return boxes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Reference image file; no camera is used")
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "seat_layout_from_image.yaml")
    parser.add_argument("--camera-id", default="CAM01")
    parser.add_argument("--seat", action="append", default=[], help="SEAT_01:left,top,right,bottom in pixels; repeat for each seat")
    parser.add_argument("--interactive", action="store_true", help="Click two corners per seat on the supplied image")
    parser.add_argument("--count", type=int, default=1, help="Number of seats for --interactive")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output YAML")
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"could not read image: {args.image}")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output exists; choose another path or pass --overwrite: {args.output}")
    height, width = image.shape[:2]
    boxes = [_pixel_box(value, width, height) for value in args.seat]
    if args.interactive:
        if boxes:
            parser.error("use either --seat or --interactive, not both")
        boxes = _interactive_boxes(image, args.count)
    if not boxes:
        parser.error("provide at least one --seat or use --interactive")
    zones = []
    for seat_id, roi in boxes:
        zone = SeatZone(seat_id=seat_id, roi=roi, anchor_x=(roi[0] + roi[2]) / 2.0, anchor_y=(roi[1] + roi[3]) / 2.0)
        zones.append(zone)
    payload = {
        "camera_id": args.camera_id,
        "layout_status": "IMAGE_REFERENCE_PENDING_REAL_ROOM_REVIEW",
        "coordinate_convention": "normalized_image_xy_x_right_y_down",
        "reference_image": str(args.image),
        "seats": [{**zone.to_dict(), "id": zone.seat_id, "roi": zone.to_dict()["roi"]} for zone in zones],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {len(zones)} normalized seat zones to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
