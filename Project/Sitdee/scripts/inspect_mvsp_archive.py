"""Inspect a manually supplied MVSP archive without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
INCOMING = ROOT / "datasets" / "incoming" / "mvsp"
OUT = ROOT / "outputs" / "validation" / "mvsp"
REPORT = ROOT / "reports" / "mvsp_archive_inspection.md"
SUPPORTED_SUFFIXES = (".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz")
LIST_LIMIT = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_kind(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith((".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz")):
        return "tar"
    return "unsupported"


def discover_archives(incoming: Path = INCOMING) -> list[Path]:
    if not incoming.exists():
        return []
    return sorted(path for path in incoming.rglob("*") if path.is_file() and archive_kind(path) != "unsupported")


def safe_member_name(name: str) -> tuple[str | None, str | None]:
    """Return a normalized relative name or a precise rejection reason."""
    normalized = name.replace("\\", "/")
    if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", normalized):
        return None, "absolute_path"
    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        return None, "path_traversal"
    cleaned = "/".join(part for part in parts if part not in ("", "."))
    if not cleaned:
        return None, "empty_path"
    return cleaned, None


def _category(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
        return "image"
    if suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}:
        return "video"
    if suffix in {".csv", ".json", ".txt", ".xml", ".mat", ".yaml", ".yml"}:
        return "annotation_or_metadata"
    if Path(name).name.lower().startswith(("readme", "license", "copying", "notice")):
        return "readme_or_license"
    return "other"


def _zip_members(path: Path) -> Iterable[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name, reason = safe_member_name(info.filename)
            is_dir = info.is_dir() or info.filename.endswith(("/", "\\"))
            is_symlink = ((info.external_attr >> 16) & 0o170000) == 0o120000
            yield {"raw_name": info.filename, "name": name, "bytes": int(info.file_size), "is_dir": is_dir, "is_symlink": is_symlink, "unsafe_reason": reason or ("symlink" if is_symlink else None)}


def _tar_members(path: Path) -> Iterable[dict[str, Any]]:
    with tarfile.open(path, mode="r:*") as archive:
        for info in archive.getmembers():
            name, reason = safe_member_name(info.name)
            is_dir = info.isdir()
            is_link = info.issym() or info.islnk()
            yield {"raw_name": info.name, "name": name, "bytes": int(info.size), "is_dir": is_dir, "is_symlink": is_link, "unsafe_reason": reason or ("link" if is_link else None)}


def iter_members(path: Path) -> Iterable[dict[str, Any]]:
    kind = archive_kind(path)
    if kind == "zip":
        yield from _zip_members(path)
    elif kind == "tar":
        yield from _tar_members(path)
    else:
        raise ValueError(f"Unsupported archive type: {path.name}")


def inspect_archive(path: Path, *, list_limit: int = LIST_LIMIT) -> dict[str, Any]:
    kind = archive_kind(path)
    result: dict[str, Any] = {"filename": path.name, "path": str(path), "archive_type": kind, "compressed_bytes": path.stat().st_size, "sha256": sha256(path), "readable": False, "unsafe_paths": [], "member_count": 0, "extracted_bytes": 0, "category_counts": {}, "members": [], "listing_truncated": False, "error": None}
    if kind == "unsupported":
        result["error"] = "Unsupported archive type; supported types are zip and tar variants"
        return result
    try:
        for member in iter_members(path):
            result["readable"] = True
            result["member_count"] += 1
            if not member["is_dir"]:
                result["extracted_bytes"] += int(member["bytes"])
                category = _category(member["name"] or member["raw_name"])
                result["category_counts"][category] = result["category_counts"].get(category, 0) + 1
            if member["unsafe_reason"]:
                result["unsafe_paths"].append({"raw_name": member["raw_name"], "reason": member["unsafe_reason"]})
            if len(result["members"]) < list_limit:
                result["members"].append(member)
            else:
                result["listing_truncated"] = True
    except (OSError, tarfile.TarError, zipfile.BadZipFile, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_all(incoming: Path = INCOMING) -> dict[str, Any]:
    archives = [inspect_archive(path) for path in discover_archives(incoming)]
    return {"status": "WAITING_FOR_MANUAL_MVSP_ARCHIVE" if not archives else "INSPECTED", "incoming": str(incoming), "archives": archives, "safety_rule": "Extraction is refused when archive listing fails or any unsafe path/link is present."}


def write_report(payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "archive_inspection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# MVSP archive inspection", "", f"Status: **{payload['status']}**", "", f"Incoming folder: `{payload['incoming']}`", "", "| Archive | Type | Compressed bytes | SHA-256 | Readable | Members | Estimated extracted bytes | Unsafe paths |", "|---|---|---:|---|---|---:|---:|---:|"]
    for item in payload["archives"]:
        lines.append(f"| `{item['filename']}` | {item['archive_type']} | {item['compressed_bytes']} | `{item['sha256']}` | {item['readable']} | {item['member_count']} | {item['extracted_bytes']} | {len(item['unsafe_paths'])} |")
        lines.append("")
        if item["error"]:
            lines.append(f"- Error: `{item['error']}`")
        if item["unsafe_paths"]:
            lines.append(f"- **Extraction blocked**: `{item['unsafe_paths'][:5]}`")
        counts = ", ".join(f"{key}={value}" for key, value in sorted(item["category_counts"].items()))
        lines.append(f"- Categories: {counts or 'none'}")
        if item["listing_truncated"]:
            lines.append(f"- Member listing truncated at {LIST_LIMIT}; counts and safety scan continued.")
        likely = [member["name"] for member in item["members"] if member["name"] and _category(member["name"]) in {"image", "video", "annotation_or_metadata", "readme_or_license"}][:50]
        if likely:
            lines.append("- Likely media/annotation/document paths (sample): " + ", ".join(f"`{name}`" for name in likely))
    lines += ["", "The inspector never extracts or changes the archive. The safe importer must be run separately after this report shows a readable archive with zero unsafe paths/links."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", default=str(INCOMING))
    args = parser.parse_args()
    payload = inspect_all(Path(args.incoming))
    write_report(payload)
    print(json.dumps({"status": payload["status"], "archives": len(payload["archives"]), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
