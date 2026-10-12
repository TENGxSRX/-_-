"""Strict canonical adapter contract for cross-dataset experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.posture_ai.pose.landmarks import PoseLandmarks


class UnsupportedLabelError(ValueError):
    """Raised when a source label has no evidence-backed canonical mapping."""


class AdapterCompatibilityError(ValueError):
    """Raised when a source representation cannot enter a canonical schema."""


@dataclass(frozen=True)
class CanonicalSample:
    dataset_id: str
    subject_id: str | None
    session_id: str | None
    label_original: str | None
    label_canonical: str | None
    landmarks: PoseLandmarks | None
    coordinate_type: str
    quality_flags: Mapping[str, Any] = field(default_factory=dict)
    feature_schema_version: str = "UNKNOWN"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.dataset_id).strip():
            raise ValueError("dataset_id is required")
        if not str(self.coordinate_type).strip():
            raise ValueError("coordinate_type is required")
        object.__setattr__(self, "dataset_id", str(self.dataset_id))
        object.__setattr__(self, "subject_id", None if self.subject_id is None else str(self.subject_id))
        object.__setattr__(self, "session_id", None if self.session_id is None else str(self.session_id))
        object.__setattr__(self, "label_original", None if self.label_original is None else str(self.label_original))
        object.__setattr__(self, "label_canonical", None if self.label_canonical is None else str(self.label_canonical))
        object.__setattr__(self, "coordinate_type", str(self.coordinate_type))
        object.__setattr__(self, "quality_flags", dict(self.quality_flags))
        object.__setattr__(self, "feature_schema_version", str(self.feature_schema_version))
        object.__setattr__(self, "provenance", dict(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "label_original": self.label_original,
            "label_canonical": self.label_canonical,
            "landmarks": None if self.landmarks is None else self.landmarks.xyz_array().tolist(),
            "coordinate_type": self.coordinate_type,
            "quality_flags": dict(self.quality_flags),
            "feature_schema_version": self.feature_schema_version,
            "provenance": dict(self.provenance),
        }


class DatasetAdapter:
    """Base class requiring explicit representation and label decisions."""

    dataset_id = "UNSPECIFIED"
    coordinate_type = "UNKNOWN"
    feature_schema_version = "UNKNOWN"

    def map_label(self, label_original: Any) -> str:
        raise UnsupportedLabelError(f"No evidence-backed label mapping for {self.dataset_id}: {label_original!r}")

    def adapt_landmarks(self, values: Any) -> PoseLandmarks:
        if values is None:
            raise AdapterCompatibilityError(f"{self.dataset_id} has no landmark representation")
        try:
            return PoseLandmarks.from_any(values)
        except (TypeError, ValueError) as exc:
            raise AdapterCompatibilityError(f"{self.dataset_id} landmark representation is incompatible") from exc

    def adapt(
        self,
        *,
        label_original: Any,
        landmarks: Any,
        subject_id: str | None = None,
        session_id: str | None = None,
        quality_flags: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> CanonicalSample:
        return CanonicalSample(
            dataset_id=self.dataset_id,
            subject_id=subject_id,
            session_id=session_id,
            label_original=None if label_original is None else str(label_original),
            label_canonical=self.map_label(label_original),
            landmarks=self.adapt_landmarks(landmarks),
            coordinate_type=self.coordinate_type,
            quality_flags=quality_flags or {},
            feature_schema_version=self.feature_schema_version,
            provenance=provenance or {},
        )


class MultiPostureAdapter(DatasetAdapter):
    dataset_id = "MultiPosture"
    coordinate_type = "MULTIPOSTURE_SOURCE_XYZ"
    feature_schema_version = "MULTIPOSTURE_126_HYBRID_BASELINE"
    _labels = {
        "TUP": "NORMAL",
        "TLF": "NON_NORMAL",
        "TLB": "NON_NORMAL",
        "TLL": "NON_NORMAL",
        "TLR": "NON_NORMAL",
    }

    def map_label(self, label_original: Any) -> str:
        key = str(label_original).strip().upper()
        if key not in self._labels:
            raise UnsupportedLabelError(f"Unsupported MultiPosture label: {label_original!r}")
        return self._labels[key]


class V4RgbAdapter(DatasetAdapter):
    dataset_id = "RGB_NATIVE_V4"
    coordinate_type = "MEDIAPIPE_WORLD_HIP_CENTERED"
    feature_schema_version = "FEATURE_V4_15"
    _labels = {"TUP": "NORMAL", "TLB": "NON_NORMAL"}

    def map_label(self, label_original: Any) -> str:
        key = str(label_original).strip().upper()
        if key not in self._labels:
            raise UnsupportedLabelError(f"Unsupported RGB-native V4 label: {label_original!r}")
        return self._labels[key]


def canonical_label(dataset_id: str, label_original: Any) -> str:
    adapters = {"MultiPosture": MultiPostureAdapter(), "RGB_NATIVE_V4": V4RgbAdapter()}
    adapter = adapters.get(str(dataset_id))
    if adapter is None:
        raise UnsupportedLabelError(f"Dataset is not approved for canonical supervised mapping: {dataset_id}")
    return adapter.map_label(label_original)


__all__ = [
    "AdapterCompatibilityError",
    "CanonicalSample",
    "DatasetAdapter",
    "MultiPostureAdapter",
    "UnsupportedLabelError",
    "V4RgbAdapter",
    "canonical_label",
]
