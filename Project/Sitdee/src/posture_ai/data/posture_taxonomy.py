"""Versioned, human-readable posture-label taxonomy.

The taxonomy separates a label written by a human from any later source-label
mapping.  It deliberately has no implicit aliases: a caller must record both
the original source label and a canonical label when a mapping is justified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class TaxonomyLabel:
    code: str
    display_en: str
    display_th: str
    collection_enabled: bool = False
    notes: str = ""


@dataclass(frozen=True)
class PostureTaxonomy:
    schema_version: str
    labels: dict[str, TaxonomyLabel]
    source_mappings: dict[str, str]

    @property
    def collection_labels(self) -> tuple[str, ...]:
        return tuple(code for code, item in self.labels.items() if item.collection_enabled)

    def validate_human_label(self, value: str) -> str:
        label = str(value).strip().upper()
        if label not in self.collection_labels:
            raise ValueError(f"human label is not enabled for this collection taxonomy: {label!r}")
        return label

    def canonical_for_source_label(self, value: str) -> str | None:
        """Return an explicitly configured mapping, never a guessed alias."""

        source = str(value).strip()
        return self.source_mappings.get(source)


def load_posture_taxonomy(path: str | Path) -> PostureTaxonomy:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("posture taxonomy must be a mapping")
    schema_version = str(raw.get("schema_version", "")).strip()
    if not schema_version:
        raise ValueError("posture taxonomy requires schema_version")
    raw_labels = raw.get("labels")
    if not isinstance(raw_labels, Mapping) or not raw_labels:
        raise ValueError("posture taxonomy requires a non-empty labels mapping")
    labels: dict[str, TaxonomyLabel] = {}
    for key, raw_label in raw_labels.items():
        code = str(key).strip().upper()
        if not code or not isinstance(raw_label, Mapping):
            raise ValueError("each taxonomy label must be a mapping with a non-empty code")
        if code in labels:
            raise ValueError(f"duplicate taxonomy label: {code}")
        labels[code] = TaxonomyLabel(
            code=code,
            display_en=str(raw_label.get("display_en", code)).strip() or code,
            display_th=str(raw_label.get("display_th", code)).strip() or code,
            collection_enabled=bool(raw_label.get("collection_enabled", False)),
            notes=str(raw_label.get("notes", "")).strip(),
        )
    raw_mappings = raw.get("source_mappings", {})
    if not isinstance(raw_mappings, Mapping):
        raise ValueError("source_mappings must be a mapping when present")
    mappings: dict[str, str] = {}
    for source_label, canonical in raw_mappings.items():
        target = str(canonical).strip().upper()
        if target not in labels:
            raise ValueError(f"source mapping points to unknown canonical label: {target}")
        mappings[str(source_label).strip()] = target
    taxonomy = PostureTaxonomy(schema_version=schema_version, labels=labels, source_mappings=mappings)
    if not taxonomy.collection_labels:
        raise ValueError("taxonomy must enable at least one collection label")
    return taxonomy


__all__ = ["PostureTaxonomy", "TaxonomyLabel", "load_posture_taxonomy"]
