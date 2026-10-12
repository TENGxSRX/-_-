"""Audit MVSP media, annotation, labels, grouping, and V5 readiness.

The default audit is intentionally conservative: directory names and filename
tokens are evidence to review, not automatic supervised labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw" / "mvsp"
OUT = ROOT / "outputs" / "validation" / "mvsp"
REPORT = ROOT / "reports" / "mvsp_inventory.md"
REPORT_TH = ROOT / "reports" / "mvsp_inventory_th.md"
MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_SUFFIXES = MEDIA_SUFFIXES - {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ANNOTATION_SUFFIXES = {".csv", ".json", ".txt", ".xml", ".mat", ".yaml", ".yml"}
TARGETS = ("TUP", "TLF", "TLB", "TLL", "TLR", "SLOUCH")
SOURCE_TERMS = {"upright", "up", "back", "backward", "left", "right", "down", "forward", "cheek", "chin", "leg", "cross"}


def _relative(path: Path, root: Path = RAW) -> str:
    return path.relative_to(root).as_posix()


def _files(root: Path = RAW) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts)


def _class_candidates(files: list[Path], root: Path = RAW) -> list[dict[str, Any]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        for parent in path.relative_to(root).parents:
            if str(parent) == ".":
                continue
            token = parent.name.lower().replace(" ", "_").replace("-", "_")
            if token in SOURCE_TERMS or any(term in token for term in SOURCE_TERMS):
                grouped[str(parent)].append(path)
    result = []
    for directory, members in sorted(grouped.items()):
        result.append({"directory": directory, "media_files": sum(path.suffix.lower() in MEDIA_SUFFIXES for path in members), "files": len(members), "source_label_candidate": Path(directory).name, "mapping_status": "UNKNOWN", "mapping_note": "Directory naming alone is not accepted as a project label."})
    return result


def _text_evidence(path: Path, root: Path = RAW) -> dict[str, Any]:
    try:
        if path.stat().st_size > 2_000_000:
            return {"path": _relative(path, root), "read": False, "reason": "over_2MB_bound"}
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError) as exc:
        return {"path": _relative(path, root), "read": False, "reason": str(exc)}
    lower = text.lower()
    return {"path": _relative(path, root), "read": True, "bytes": path.stat().st_size, "terms_found": sorted(term for term in SOURCE_TERMS if re.search(rf"\b{re.escape(term)}\b", lower)), "target_terms_found": sorted(target for target in TARGETS if target.lower() in lower), "header": next((line[:500] for line in text.splitlines() if line.strip()), "")}


def _participant_tokens(path: Path, root: Path = RAW) -> list[str]:
    value = _relative(path, root)
    tokens = set()
    patterns = [r"(?:participant|subject|person|user)[ _-]*([A-Za-z0-9]+)", r"(?:^|[/_ -])(p\d{1,4})(?:[/_. -]|$)", r"(?:^|[/_ -])(s\d{1,4})(?:[/_. -]|$)"]
    for pattern in patterns:
        tokens.update(match.lower() for match in re.findall(pattern, value, flags=re.IGNORECASE))
    return sorted(tokens)


def _license_files(files: list[Path], root: Path = RAW) -> list[str]:
    return [_relative(path, root) for path in files if path.name.lower().startswith(("license", "copying", "notice"))]


def _media_counts(files: list[Path]) -> dict[str, Any]:
    images = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]
    videos = [path for path in files if path.suffix.lower() in MEDIA_SUFFIXES - IMAGE_SUFFIXES]
    return {"images": len(images), "videos": len(videos), "image_paths_sample": [str(path) for path in images[:20]], "video_paths_sample": [str(path) for path in videos[:20]]}


def determine_status(payload: dict[str, Any], mapping_review: dict[str, Any] | None = None) -> str:
    media = payload["media"]
    if not media["images"] and not media["videos"]:
        return "MVSP_NOT_RGB_SUITABLE" if payload.get("import_manifest_present") else "WAITING_FOR_MANUAL_MVSP_ARCHIVE"
    if payload["license_status"] != "LICENSE_VERIFIED":
        return "MVSP_LICENSE_BLOCKED"
    if not payload["participant_tokens"]:
        return "MVSP_PARTICIPANT_GROUPING_BLOCKED"
    if not mapping_review:
        return "MVSP_PARTIALLY_USEFUL"
    mappings = mapping_review.get("mappings", {})
    statuses = [str(mappings.get(target, {}).get("status", "UNKNOWN")).upper() for target in TARGETS]
    if any(status == "INCOMPATIBLE" for status in statuses):
        return "MVSP_LABELS_INCOMPATIBLE"
    if all(status in {"EXACT", "COMPATIBLE"} for status in statuses):
        return "MVSP_V5_READY"
    return "MVSP_PARTIALLY_USEFUL"


def audit(root: Path = RAW, mapping_review_path: Path | None = None) -> dict[str, Any]:
    files = _files(root)
    if not files:
        return {"status": "WAITING_FOR_MANUAL_MVSP_ARCHIVE", "raw_root": str(root), "files": 0, "media": {"images": 0, "videos": 0}, "class_candidates": [], "annotations": [], "participant_tokens": [], "license_files": [], "license_status": "LICENSE_UNCLEAR", "mapping_review": None, "import_manifest_present": False, "v5_gate": "WAITING_FOR_MANUAL_MVSP_ARCHIVE"}
    media = _media_counts(files)
    annotation_files = [path for path in files if path.suffix.lower() in ANNOTATION_SUFFIXES]
    doc_files = [path for path in files if path.name.lower().startswith(("readme", "license", "copying", "notice"))]
    participant_tokens = sorted({token for path in files for token in _participant_tokens(path, root)})
    mapping_review = json.loads(mapping_review_path.read_text(encoding="utf-8")) if mapping_review_path and mapping_review_path.exists() else None
    license_files = _license_files(files, root)
    payload = {"status": "AUDITED", "raw_root": str(root), "files": len(files), "total_bytes": sum(path.stat().st_size for path in files), "media": media, "class_candidates": _class_candidates(files, root), "annotations": {"count": len(annotation_files), "files": [_text_evidence(path, root) for path in annotation_files[:1000]]}, "readme_license_files": [_relative(path, root) for path in doc_files], "license_files": license_files, "license_status": "LICENSE_VERIFIED" if license_files else "LICENSE_UNCLEAR", "participant_tokens": participant_tokens, "participant_grouping_status": "CANDIDATE_TOKENS_FOUND_REVIEW_REQUIRED" if participant_tokens else "NOT_RECONSTRUCTED", "mapping_review": mapping_review, "import_manifest_present": (root / ".." / ".." / "processed" / "mvsp" / "import_manifest.json").resolve().exists() if root.resolve() == RAW.resolve() else False, "label_policy": "Only annotation-backed/source-definition-backed mappings may be promoted from UNKNOWN; image and directory names alone are not labels.", "v5_gate": None}
    payload["v5_gate"] = determine_status(payload, mapping_review)
    return payload


def write_reports(payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "label_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# MVSP inventory and label forensics", "", f"Status: **{payload['v5_gate']}**", "", f"Raw root: `{payload['raw_root']}`", f"Total files: `{payload['files']}`", f"Total bytes: `{payload.get('total_bytes', 0)}`", f"License status: **{payload['license_status']}**", f"Participant grouping: **{payload.get('participant_grouping_status', 'UNKNOWN')}**", "", "## Media", "", f"- Images: `{payload['media']['images']}`", f"- Videos: `{payload['media']['videos']}`", "- RGB validity is not accepted from filenames alone; the preflight script checks readable image channels/resolution on a representative sample.", "", "## Candidate class directories", "", "| Directory | Media files | Candidate source label | Mapping status |", "|---|---:|---|---|"]
    for item in payload.get("class_candidates", []):
        lines.append(f"| `{item['directory']}` | {item['media_files']} | `{item['source_label_candidate']}` | `{item['mapping_status']}` |")
    lines += ["", "## Annotation and documents", "", f"- Annotation/metadata files: `{payload['annotations']['count']}`", f"- README/license files: `{payload['readme_license_files']}`", "- Directory names, filenames, and paper terminology remain forensic evidence until a source-backed label definition and join are verified.", "", "## Participants and leakage", "", f"- Candidate participant tokens: `{payload.get('participant_tokens', [])}`", f"- Grouping status: `{payload.get('participant_grouping_status', 'UNKNOWN')}`", "- A participant-independent V5 claim remains blocked unless every imported observation can be assigned to a verified participant/session/video/view group.", "", "## V5 gate", "", f"**{payload['v5_gate']}**", "", "No V5 training is run by this audit. A human-reviewed mapping file may be supplied later, but it must record status `EXACT`, `COMPATIBLE`, `AMBIGUOUS`, `INCOMPATIBLE`, or `UNKNOWN` and cite the source evidence."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    th = ["# รายงาน inventory และ label forensics ของ MVSP", "", f"สถานะ: **{payload['v5_gate']}**", "", f"จำนวนไฟล์: `{payload['files']}`", f"ขนาดรวม: `{payload.get('total_bytes', 0)}` bytes", f"license: **{payload['license_status']}**", f"participant grouping: **{payload.get('participant_grouping_status', 'UNKNOWN')}**", "", "## สื่อ", "", f"- images: `{payload['media']['images']}`", f"- videos: `{payload['media']['videos']}`", "- จะยืนยันว่าเป็น RGB จากชื่อไฟล์อย่างเดียวไม่ได้ ต้องใช้ preflight ตรวจไฟล์จริง", "", "## class candidates", "", "| directory | media | source label candidate | mapping status |", "|---|---:|---|---|"]
    for item in payload.get("class_candidates", []):
        th.append(f"| `{item['directory']}` | {item['media_files']} | `{item['source_label_candidate']}` | `{item['mapping_status']}` |")
    th += ["", "## label และ leakage", "", f"- annotation/metadata files: `{payload['annotations']['count']}`", f"- participant tokens ที่พบ: `{payload.get('participant_tokens', [])}`", "- ชื่อโฟลเดอร์หรือชื่อไฟล์ไม่ถูกเลื่อนเป็น label อัตโนมัติ", "- ถ้ายัง reconstruct participant/session/video/view ไม่ได้ จะอ้าง participant-independent V5 ไม่ได้", "", "## V5 gate", "", f"**{payload['v5_gate']}**", "", "สคริปต์นี้ไม่ฝึก V5 และไม่สร้าง pseudo-label"]
    REPORT_TH.write_text("\n".join(th) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(RAW))
    parser.add_argument("--mapping-review", help="optional human-reviewed JSON mapping; never generated automatically")
    args = parser.parse_args()
    payload = audit(Path(args.raw), Path(args.mapping_review) if args.mapping_review else None)
    write_reports(payload)
    print(json.dumps({"status": payload["v5_gate"], "files": payload["files"], "images": payload["media"]["images"], "videos": payload["media"]["videos"], "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
