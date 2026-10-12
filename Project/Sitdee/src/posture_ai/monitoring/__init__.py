"""Privacy-aware runtime observation helpers.

These utilities record engineering telemetry only.  They do not create
training labels and they never store video frames unless a caller explicitly
implements a separate, attended capture workflow.
"""

from .shadow import (
    SHADOW_SCHEMA_VERSION,
    UNKNOWN_REASON_CODES,
    RotatingJsonlWriter,
    ShadowReviewQueue,
    analyze_shadow_session,
    build_frame_observation,
    build_subject_observation,
    derive_unknown_reasons,
)

__all__ = [
    "SHADOW_SCHEMA_VERSION",
    "UNKNOWN_REASON_CODES",
    "RotatingJsonlWriter",
    "ShadowReviewQueue",
    "analyze_shadow_session",
    "build_frame_observation",
    "build_subject_observation",
    "derive_unknown_reasons",
]
