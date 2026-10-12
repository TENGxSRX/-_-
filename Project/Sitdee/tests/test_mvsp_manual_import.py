from __future__ import annotations

import json
import zipfile

import pytest

from scripts.audit_mvsp_labels import audit
from scripts.import_mvsp_dataset import import_archive
from scripts.inspect_mvsp_archive import inspect_archive, safe_member_name


def test_archive_inspection_lists_media_and_hash(tmp_path):
    archive = tmp_path / "mvsp.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("README.txt", "upright left right")
        handle.writestr("participant_01/upright/frame.jpg", b"not-an-image")
    result = inspect_archive(archive)
    assert result["readable"]
    assert result["member_count"] == 2
    assert result["category_counts"]["image"] == 1
    assert result["sha256"]


def test_path_traversal_is_rejected_before_import(tmp_path):
    assert safe_member_name("../outside.txt")[1] == "path_traversal"
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "unsafe")
    raw = tmp_path / "raw"
    with pytest.raises(RuntimeError, match="unsafe"):
        import_archive(archive, raw_root=raw, processed_root=tmp_path / "processed", metadata_root=tmp_path / "metadata")
    assert not (tmp_path / "outside.txt").exists()


def test_import_generates_manifest_and_refuses_overwrite(tmp_path):
    archive = tmp_path / "mvsp.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("participant_01/upright/frame.jpg", b"sample")
    raw = tmp_path / "raw"
    manifest = import_archive(archive, raw_root=raw, processed_root=tmp_path / "processed", metadata_root=tmp_path / "metadata")
    assert archive.exists()
    assert manifest["imported_files"][0]["relative_path"] == "participant_01/upright/frame.jpg"
    assert (tmp_path / "processed" / "file_inventory.csv").exists()
    assert (tmp_path / "metadata" / "mvsp_hashes.sha256").exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        import_archive(archive, raw_root=raw, processed_root=tmp_path / "processed2", metadata_root=tmp_path / "metadata2")


def test_label_audit_keeps_directory_names_unknown_and_detects_group_tokens(tmp_path):
    root = tmp_path / "mvsp"
    (root / "participant_01" / "left").mkdir(parents=True)
    (root / "participant_01" / "left" / "frame.jpg").write_bytes(b"sample")
    (root / "LICENSE.txt").write_text("research license", encoding="utf-8")
    (root / "labels.csv").write_text("filename,label\nframe.jpg,left\n", encoding="utf-8")
    result = audit(root)
    assert result["license_status"] == "LICENSE_VERIFIED"
    assert "01" in result["participant_tokens"]
    assert result["class_candidates"][0]["mapping_status"] == "UNKNOWN"
    assert result["v5_gate"] == "MVSP_PARTIALLY_USEFUL"


def test_empty_audit_waits_for_manual_archive(tmp_path):
    result = audit(tmp_path / "empty")
    assert result["v5_gate"] == "WAITING_FOR_MANUAL_MVSP_ARCHIVE"
