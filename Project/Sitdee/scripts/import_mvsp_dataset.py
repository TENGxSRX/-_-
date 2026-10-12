"""Safely import a manually downloaded MVSP archive into raw/mvsp."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

try:
    from .inspect_mvsp_archive import archive_kind, discover_archives, inspect_archive, safe_member_name
except ImportError:  # Direct ``python scripts/import_mvsp_dataset.py`` execution.
    from inspect_mvsp_archive import archive_kind, discover_archives, inspect_archive, safe_member_name

ROOT = Path(__file__).resolve().parents[1]
INCOMING = ROOT / "datasets" / "incoming" / "mvsp"
RAW = ROOT / "datasets" / "raw" / "mvsp"
PROCESSED = ROOT / "datasets" / "processed" / "mvsp"
METADATA = ROOT / "metadata"
REPORT = ROOT / "reports" / "mvsp_import.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entries(path: Path):
    if archive_kind(path) == "zip":
        archive = zipfile.ZipFile(path)
        try:
            for info in archive.infolist():
                name, reason = safe_member_name(info.filename)
                is_dir = info.is_dir() or info.filename.endswith(("/", "\\"))
                is_link = ((info.external_attr >> 16) & 0o170000) == 0o120000
                yield {"raw_name": info.filename, "name": name, "reason": reason or ("symlink" if is_link else None), "is_dir": is_dir, "bytes": int(info.file_size)}
        finally:
            archive.close()
    elif archive_kind(path) == "tar":
        archive = tarfile.open(path, mode="r:*")
        try:
            for info in archive.getmembers():
                name, reason = safe_member_name(info.name)
                is_dir = info.isdir()
                is_link = info.issym() or info.islnk()
                yield {"raw_name": info.name, "name": name, "reason": reason or ("link" if is_link else None), "is_dir": is_dir, "bytes": int(info.size)}
        finally:
            archive.close()
    else:
        raise ValueError(f"Unsupported archive type: {path.name}")


def choose_archive(archive: str | None = None, incoming: Path = INCOMING) -> Path:
    if archive:
        path = Path(archive)
        if not path.is_absolute():
            path = incoming / path
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = discover_archives(incoming)
    if not candidates:
        raise FileNotFoundError(f"No supported MVSP archive found in {incoming}")
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one archive; found {len(candidates)}: {[path.name for path in candidates]}")
    return candidates[0]


def _destination(root: Path, name: str) -> Path:
    target = (root / Path(*name.split("/"))).resolve()
    root_resolved = root.resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Archive member escapes extraction root: {name}") from exc
    return target


def _copy_member(archive: Path, item: dict[str, Any], target: Path) -> None:
    if archive_kind(archive) == "zip":
        with zipfile.ZipFile(archive) as source_archive:
            source = source_archive.open(item["raw_name"])
            try:
                with target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            finally:
                source.close()
    else:
        with tarfile.open(archive, mode="r:*") as source_archive:
            info = source_archive.getmember(item["raw_name"])
            source = source_archive.extractfile(info)
            if source is None:
                raise RuntimeError(f"Could not read archive member: {item['name']}")
            try:
                with target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            finally:
                source.close()


def import_archive(archive: Path, *, raw_root: Path = RAW, processed_root: Path = PROCESSED, metadata_root: Path = METADATA) -> dict[str, Any]:
    inspection = inspect_archive(archive)
    if inspection["archive_type"] == "unsupported" or not inspection["readable"]:
        raise RuntimeError(f"Archive is not readable: {inspection['error']}")
    if inspection["unsafe_paths"]:
        raise RuntimeError(f"Refusing unsafe archive paths/links: {inspection['unsafe_paths'][:5]}")
    raw_root.mkdir(parents=True, exist_ok=True)
    planned = []
    for entry in _entries(archive):
        if entry["reason"]:
            raise RuntimeError(f"Refusing archive member {entry['name'] or '<invalid>'}: {entry['reason']}")
        if not entry["name"]:
            raise RuntimeError("Archive contains an empty member path")
        target = _destination(raw_root, entry["name"])
        if target.exists():
            raise FileExistsError(f"Import would overwrite existing path: {target}")
        planned.append({**entry, "target": target})
    for item in planned:
        target: Path = item["target"]
        if item["is_dir"]:
            target.mkdir(parents=True, exist_ok=True)
    for item in planned:
        target = item["target"]
        if item["is_dir"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_member(archive, item, target)
    inventory = []
    for item in planned:
        target: Path = item["target"]
        if item["is_dir"]:
            continue
        inventory.append({"relative_path": item["name"], "bytes": target.stat().st_size, "sha256": sha256(target)})
    processed_root.mkdir(parents=True, exist_ok=True)
    with (processed_root / "file_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(inventory)
    metadata_root.mkdir(parents=True, exist_ok=True)
    (metadata_root / "mvsp_hashes.sha256").write_text("".join(f"{row['sha256']}  {row['relative_path']}\n" for row in inventory), encoding="utf-8")
    manifest = {"archive": {"filename": archive.name, "path": str(archive), "bytes": archive.stat().st_size, "sha256": sha256(archive)}, "raw_root": str(raw_root), "imported_files": inventory, "preexisting_raw_paths_not_included": "Existing files under raw/mvsp were never overwritten and are not part of this archive inventory."}
    (processed_root / "import_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text("\n".join(["# MVSP safe import", "", f"- Archive: `{archive.name}`", f"- Archive SHA-256: `{manifest['archive']['sha256']}`", f"- Extracted destination: `{raw_root}`", f"- Imported files: `{len(inventory)}`", f"- Imported bytes: `{sum(row['bytes'] for row in inventory)}`", "- Original archive was preserved in the incoming folder.", "- Existing destination paths were checked before extraction; a collision would abort the import.", "- No model training was run."]) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", help="archive filename or absolute path; omit when exactly one archive is present")
    parser.add_argument("--incoming", default=str(INCOMING))
    parser.add_argument("--raw", default=str(RAW))
    parser.add_argument("--processed", default=str(PROCESSED))
    parser.add_argument("--metadata", default=str(METADATA))
    args = parser.parse_args()
    archive = choose_archive(args.archive, Path(args.incoming))
    manifest = import_archive(archive, raw_root=Path(args.raw), processed_root=Path(args.processed), metadata_root=Path(args.metadata))
    print(json.dumps({"archive": manifest["archive"], "imported_files": len(manifest["imported_files"]), "raw_root": manifest["raw_root"]}, indent=2))


if __name__ == "__main__":
    main()
