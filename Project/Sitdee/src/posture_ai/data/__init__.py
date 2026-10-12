"""Dataset loading and schema validation."""

from .canonical import (
    AdapterCompatibilityError,
    CanonicalSample,
    DatasetAdapter,
    MultiPostureAdapter,
    UnsupportedLabelError,
    V4RgbAdapter,
    canonical_label,
)
from .cctv_collection import (
    CCTV_COLLECTION_SCHEMA_VERSION,
    CCTV_EXTRA_SAMPLE_COLUMNS,
    CCTV_LABEL_KEYS,
    CCTV_MANUAL_LABELS,
    CCTV_NON_SAMPLE_ACTIONS,
    ManualPersonAssignment,
    label_for_key,
    validate_manual_label,
    validate_person_id,
)
from .dataset_registry import DATASET_REGISTRY, DatasetRecord, get_dataset_record, registry_as_dicts
from .cctv_metrics import (
    aggregate_detection_metrics,
    detection_frame_metrics,
    detection_metrics_by_scenario,
    frozen_v4_metrics,
    per_person_core_metrics,
    temporal_stability_metrics,
    tracking_metrics,
)
from .guided_collection import DEFAULT_GUIDED_STAGES, GUIDED_COLLECTION_SCHEMA_VERSION, GuidedStage, protocol_manifest
from .activity_collection import ACTIVITY_COLLECTION_SCHEMA_VERSION, ACTIVITY_EXTRA_SAMPLE_COLUMNS, ACTIVITY_LABELS
from .behavior_collection import (
    BEHAVIOR_COLLECTION_SCHEMA_VERSION,
    BEHAVIOR_EXTRA_SAMPLE_COLUMNS,
    POSTURE_NONE,
    SITTING_POSTURE_LABELS,
    validate_behavior_labels,
)

__all__ = [
    "AdapterCompatibilityError",
    "aggregate_detection_metrics",
    "CCTV_COLLECTION_SCHEMA_VERSION",
    "CCTV_EXTRA_SAMPLE_COLUMNS",
    "CCTV_LABEL_KEYS",
    "CCTV_MANUAL_LABELS",
    "CCTV_NON_SAMPLE_ACTIONS",
    "CanonicalSample",
    "DATASET_REGISTRY",
    "DatasetAdapter",
    "DatasetRecord",
    "MultiPostureAdapter",
    "ManualPersonAssignment",
    "UnsupportedLabelError",
    "V4RgbAdapter",
    "canonical_label",
    "detection_frame_metrics",
    "detection_metrics_by_scenario",
    "frozen_v4_metrics",
    "get_dataset_record",
    "registry_as_dicts",
    "per_person_core_metrics",
    "temporal_stability_metrics",
    "tracking_metrics",
    "label_for_key",
    "validate_manual_label",
    "validate_person_id",
    "DEFAULT_GUIDED_STAGES",
    "GUIDED_COLLECTION_SCHEMA_VERSION",
    "GuidedStage",
    "protocol_manifest",
    "ACTIVITY_COLLECTION_SCHEMA_VERSION",
    "ACTIVITY_EXTRA_SAMPLE_COLUMNS",
    "ACTIVITY_LABELS",
    "BEHAVIOR_COLLECTION_SCHEMA_VERSION",
    "BEHAVIOR_EXTRA_SAMPLE_COLUMNS",
    "POSTURE_NONE",
    "SITTING_POSTURE_LABELS",
    "validate_behavior_labels",
]
