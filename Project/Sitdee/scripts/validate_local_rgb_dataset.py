"""Validate local RGB V1 videos without changing raw media."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.posture_ai.data.local_rgb_v1 import (  # noqa: E402
    LOCAL_MANIFEST_COLUMNS,
    LOCAL_VIDEO_SUFFIXES,
    expected_manifest_rows,
    validate_manifest_schema,
    validate_video_path,
)

RAW_ROOT = ROOT / "datasets" / "raw" / "local_rgb_v1"
PROCESSED_ROOT = ROOT / "datasets" / "processed" / "local_rgb_v1"
HASH_PATH = ROOT / "outputs" / "metrics" / "local_rgb_v1_hashes.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest() -> list[dict[str, str]]:
    path = PROCESSED_ROOT / "manifest.csv"
    if not path.exists():
        return expected_manifest_rows()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    validate_manifest_schema(rows)
    return rows


def _write_manifest(rows: list[dict[str, str]]) -> None:
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    with (PROCESSED_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOCAL_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def validate(*, minimum_duration_seconds: float = 20.0, write_manifest: bool = True) -> dict:
    rows = _read_manifest()
    by_path = {row["video_path"].replace("\\", "/"): row for row in rows}
    files = sorted(
        path for path in RAW_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in LOCAL_VIDEO_SUFFIXES
    ) if RAW_ROOT.exists() else []
    hashes: dict[str, dict[str, int | str]] = {}
    hash_to_paths: dict[str, list[str]] = {}
    valid_videos: list[dict] = []
    invalid: list[dict[str, str]] = []
    for path in files:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        digest = _sha256(path)
        hashes[relative] = {"bytes": path.stat().st_size, "sha256": digest}
        hash_to_paths.setdefault(digest, []).append(relative)
        parsed, path_errors = validate_video_path(path, RAW_ROOT)
        if path_errors or parsed is None:
            invalid.append({"path": relative, "errors": "; ".join(path_errors)})
            continue
        capture = cv2.VideoCapture(str(path))
        opened = bool(capture.isOpened())
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) if opened else 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if opened else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if opened else 0
        duration = frame_count / fps if fps > 0 else 0.0
        capture.release()
        errors = []
        if not opened:
            errors.append("unreadable_video")
        if fps <= 0:
            errors.append("invalid_fps")
        if frame_count <= 0:
            errors.append("zero_frames")
        if width <= 0 or height <= 0:
            errors.append("invalid_resolution")
        if 0 < duration < minimum_duration_seconds:
            errors.append("SHORTER_THAN_CONFIGURED_MINIMUM")
        status = "VALIDATED" if not errors else "WARNING"
        valid_videos.append({
            "path": relative,
            "parsed": parsed,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "readable": opened,
            "fps": fps,
            "total_frames": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "errors": errors,
            "status": status,
        })
        row = by_path.get(relative)
        if row is None:
            invalid.append({"path": relative, "errors": "No matching expected manifest slot"})
        else:
            row.update({
                "duration_seconds": f"{duration:.6f}" if duration else "",
                "fps": f"{fps:.6f}" if fps else "",
                "width": str(width) if width else "",
                "height": str(height) if height else "",
                "total_frames": str(frame_count) if frame_count else "",
                "status": status,
                "notes": (row.get("notes", "") + " " + ";".join(errors)).strip(),
            })
    for relative in by_path:
        if relative not in {item["path"] for item in valid_videos}:
            by_path[relative]["status"] = "WAITING_FOR_VIDEO"
    duplicate_hashes = {digest: paths for digest, paths in hash_to_paths.items() if len(paths) > 1}
    report = {
        "status": "VALIDATED_WITH_WARNINGS" if invalid or any(item["errors"] for item in valid_videos) else ("WAITING_FOR_LOCAL_RGB_DATA" if not valid_videos else "VALIDATED"),
        "videos_checked": len(valid_videos),
        "invalid_count": len(invalid),
        "invalid": invalid,
        "videos": valid_videos,
        "duplicate_sha256": duplicate_hashes,
        "minimum_duration_seconds": minimum_duration_seconds,
        "raw_media_modified": False,
    }
    HASH_PATH.parent.mkdir(parents=True, exist_ok=True)
    HASH_PATH.write_text(json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8")
    if write_manifest:
        _write_manifest(list(by_path.values()))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-duration-seconds", type=float, default=20.0)
    parser.add_argument("--no-write-manifest", action="store_true")
    args = parser.parse_args()
    report = validate(minimum_duration_seconds=args.minimum_duration_seconds, write_manifest=not args.no_write_manifest)
    output = ROOT / "outputs" / "metrics" / "local_rgb_v1_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "videos_checked": report["videos_checked"], "invalid_count": report["invalid_count"], "hashes": str(HASH_PATH)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
