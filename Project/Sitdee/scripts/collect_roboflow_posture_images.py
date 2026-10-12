"""Consent-gated local image collection for a future Roboflow pose project."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

import cv2

LABEL_KEYS = {ord("1"): "TUP", ord("2"): "TLF", ord("3"): "TLB", ord("4"): "TLL", ord("5"): "TLR"}
NUMPAD_KEYS = {0x61: "TUP", 0x62: "TLF", 0x63: "TLB", 0x64: "TLL", 0x65: "TLR"}
BUTTONS = ((12, 105, 112, 145, "TUP"), (122, 105, 222, 145, "TLF"),
           (232, 105, 332, 145, "TLB"), (342, 105, 442, 145, "TLL"),
           (452, 105, 552, 145, "TLR"), (562, 105, 732, 145, "SAVE"))


def safe_id(value: str) -> str:
    value = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9_-]{2,32}", value):
        raise ValueError("IDs must contain only A-Z, 0-9, underscore or dash")
    return value


def next_image_sequence(images: Path, participant: str, session: str) -> int:
    """Continue after the highest existing sequence without overwriting images."""
    highest = 0
    prefix = f"{participant}__{session}__"
    for path in images.glob(f"{prefix}*.jpg"):
        parts = path.stem.split("__")
        if len(parts) == 4 and parts[0] == participant and parts[1] == session:
            try:
                highest = max(highest, int(parts[3]))
            except ValueError:
                continue
    return highest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--consent", action="store_true", help="Required: participant consent includes local image collection and intended Roboflow upload")
    parser.add_argument("--output-root", default="datasets/incoming/roboflow_posture")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--countdown-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if not args.consent:
        parser.error("--consent is required; images contain identifiable personal data")
    if args.countdown_seconds < 0:
        parser.error("--countdown-seconds must be zero or positive")
    participant, session = safe_id(args.participant), safe_id(args.session)
    root = Path(args.output_root) / participant / session
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    log = root / "manifest.jsonl"
    capture = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise RuntimeError(f"camera {args.camera_index} could not be opened")
    state = {"selected": None, "save_requested": False,
             "countdown_deadline": None, "countdown_label": None}
    sequence = next_image_sequence(images, participant, session)
    if sequence:
        print(f"Resuming {participant}/{session} after image {sequence}; existing files are preserved.")
    window = "Sitdee Roboflow Collection - 1..5 label, SPACE save, Q exit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, action in BUTTONS:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if action == "SAVE":
                    state["save_requested"] = True
                else:
                    state["selected"] = action
                return

    cv2.setMouseCallback(window, on_mouse)
    try:
        with log.open("a", encoding="utf-8") as handle:
            while True:
                ok, frame = capture.read()
                if not ok:
                    continue
                now = time.monotonic()
                if state["save_requested"]:
                    state["save_requested"] = False
                    if state["selected"] and state["countdown_deadline"] is None:
                        state["countdown_label"] = state["selected"]
                        state["countdown_deadline"] = now + args.countdown_seconds

                if state["countdown_deadline"] is not None and now >= state["countdown_deadline"]:
                    sequence += 1
                    selected = state["countdown_label"]
                    name = f"{participant}__{session}__{selected}__{sequence:06d}.jpg"
                    path = images / name
                    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                        raise OSError(f"failed to write {path}")
                    row = {"image": str(path), "participant_id": participant, "session_id": session,
                           "human_label": selected, "label_source": "HUMAN_KEYBOARD_EXPLICIT",
                           "created_utc": datetime.now(timezone.utc).isoformat(), "roboflow_uploaded": False,
                           "countdown_seconds": args.countdown_seconds}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush()
                    print(f"SAVED {sequence}: {selected} -> {path}", flush=True)
                    state["countdown_deadline"] = None
                    state["countdown_label"] = None
                preview = frame.copy()
                lines = ["1=TUP 2=TLF 3=TLB 4=TLL 5=TLR", f"selected: {state['selected'] or 'NONE'} | saved: {sequence}", "SPACE=save image  Q/ESC=exit | or click buttons"]
                for i, text in enumerate(lines):
                    cv2.putText(preview, text, (12, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
                for x1, y1, x2, y2, action in BUTTONS:
                    active = action == state["selected"]
                    color = (0, 210, 0) if active else (30, 130, 230)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), color, -1)
                    cv2.putText(preview, action, (x1 + 12, y1 + 27), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
                if state["countdown_deadline"] is not None:
                    remaining = max(0, math.ceil(state["countdown_deadline"] - now))
                    message = f"{state['countdown_label']}  PHOTO IN {remaining}"
                    size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
                    x = max(10, (preview.shape[1] - size[0]) // 2)
                    y = max(210, preview.shape[0] // 2)
                    cv2.rectangle(preview, (x - 16, y - 50), (x + size[0] + 16, y + 18), (0, 0, 0), -1)
                    cv2.putText(preview, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
                cv2.imshow(window, preview)
                raw_key = cv2.waitKeyEx(1)
                key = raw_key & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                label = LABEL_KEYS.get(key) or NUMPAD_KEYS.get(key)
                if label:
                    state["selected"] = label
                if key == ord(" "):
                    state["save_requested"] = True
    finally:
        capture.release(); cv2.destroyAllWindows()
    print(json.dumps({"output": str(root), "saved_images": sequence}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
