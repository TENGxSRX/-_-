"""Print a safe human-guided no-video CCTV collection protocol.

The tool intentionally does not auto-label from V4/Geometry and does not
record video.  It gives the operator an exact protocol and a command line
compatible with the existing manual collector.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.posture_ai.data.guided_collection import protocol_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the human-guided no-video CCTV collection protocol")
    parser.add_argument("--source", default="rtsp", choices=("rtsp", "video", "webcam"))
    parser.add_argument("--profile", default="HIKVISION_MAIN")
    parser.add_argument("--max-people", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--subject-id", default="P001")
    parser.add_argument("--clip-id", default="CCTV_GUIDED_STAGEA_01")
    parser.add_argument("--expected-people-count", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--preset", default="RTSP_CCTV")
    parser.add_argument("--output-root", default="data/controlled_cctv_v2")
    parser.add_argument("--write-manifest", default=None)
    parser.add_argument("--execute", action="store_true", help="After printing the protocol, start the existing attended collector")
    args = parser.parse_args()
    manifest = protocol_manifest()
    manifest.update({
        "source": args.source,
        "camera_profile": args.profile,
        "max_people": args.max_people,
        "subject_id": args.subject_id,
        "clip_id": args.clip_id,
        "expected_people_count": args.expected_people_count,
        "preset": args.preset,
    })
    if args.write_manifest:
        path = Path(args.write_manifest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"WROTE_PROTOCOL_MANIFEST={path}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("GUIDED_PROTOCOL_ONLY: operator must confirm labels; no raw video; no prediction-as-label")
    print("Use the existing manual collector for capture after reviewing this protocol.")
    if args.execute:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "collect_controlled_cctv.py"),
            "--source", args.source,
            "--profile", args.profile,
            "--max-people", str(args.max_people),
            "--subject-id", args.subject_id,
            "--clip-id", args.clip_id,
            "--expected-people-count", str(args.expected_people_count),
            "--preset", args.preset,
            "--output-root", args.output_root,
        ]
        print("STARTING_ATTENDED_COLLECTOR: raw video remains disabled; labels remain manual")
        return int(subprocess.run(command, cwd=ROOT).returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
