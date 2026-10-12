"""Inspect JPEG inventory and dimensions without changing image files."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


SOF_MARKERS = set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0))


def jpeg_size(path: Path):
    data = path.read_bytes()
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            break
        marker = data[index]
        index += 1
        if marker in (0xD8, 0xD9):
            continue
        if index + 2 > len(data):
            break
        segment_length = int.from_bytes(data[index:index + 2], "big")
        if marker in SOF_MARKERS and index + 7 <= len(data):
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            return width, height
        index += segment_length
    return None


def main() -> None:
    root = Path(sys.argv[1])
    files = sorted(root.rglob("*.jpg"))
    dimensions = Counter()
    total_bytes = 0
    sha256 = hashlib.sha256()
    unreadable = []
    for path in files:
        total_bytes += path.stat().st_size
        dimensions[str(jpeg_size(path))] += 1
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha256.update(block)
        if jpeg_size(path) is None:
            unreadable.append(str(path))
    sidecars = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".xml", ".txt", ".yaml", ".yml", ".mat", ".mp4", ".avi", ".mov"}
    )
    print(json.dumps({
        "root": str(root),
        "jpg_count": len(files),
        "total_bytes": total_bytes,
        "sha256_concatenated_file_bytes": sha256.hexdigest(),
        "dimensions": dict(dimensions),
        "sidecar_or_video_files": sidecars,
        "unreadable_jpegs": unreadable,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

