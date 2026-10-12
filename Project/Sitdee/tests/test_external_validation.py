from __future__ import annotations

import csv

from scripts.external_validation import MANIFEST_FIELDS, MANIFEST_PATH, MEDIA_ROWS, PROJECT_LABELS


def test_external_manifest_has_required_columns_and_safe_labels():
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert list(rows[0]) == MANIFEST_FIELDS
    assert {row["verified_label"] for row in rows} <= PROJECT_LABELS
    assert {row["verified_label"] for row in rows} == {"TUP", "TLB", "UNLABELED"}


def test_external_pack_keeps_ambiguous_source_classes_unlabeled():
    by_file = {row["source_file"]: row for row in MEDIA_ROWS}
    assert by_file["SS_1.mp4"]["verified_label"] == "TUP"
    assert by_file["LB_2.mp4"]["verified_label"] == "TLB"
    assert by_file["MB_1.mp4"]["verified_label"] == "UNLABELED"
