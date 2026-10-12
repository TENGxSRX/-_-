"""Non-production anonymous multi-person and optional seat-aware contracts.

The original single-person path remains available. Generic PERSON tracking
and seat-aware workstation assignment are separate opt-in runtimes.
"""

from .demo import parse_source, run_multiperson_demo
from .mediapipe_tasks import MultiPersonMediaPipeBackend, create_multi_pose_landmarker
from .overlay import dashboard_cards, draw_multiperson_overlay
from .runtime import MultiPersonRuntime, MultiPersonRuntimeConfig, resolve_performance_preset, should_process_frame
from .schema import MultiPersonFrameResult, PoseSubject, anonymous_subject_id
from .seating import (
    SeatAwareFrameResult,
    SeatAwareRuntime,
    SeatAwareRuntimeConfig,
    SeatAssignment,
    SeatFrameState,
    SeatZone,
    body_location,
    dashboard_seat_cards,
    draw_seat_layout,
    load_seat_layout,
)
from .telemetry import MultiPersonTelemetry, frame_to_event, legacy_single_person_to_event, seat_frame_to_event
from .target_lock import TargetCandidate, TargetLockController, TargetLockSnapshot
from .tracking import AnonymousPoseTracker, PoseDetection, core_quality_summary
from .yolo_association import AssociationResult, PersonPoseAssociation, associate_pose_items_to_person_boxes

__all__ = [
    "AnonymousPoseTracker",
    "AssociationResult",
    "MultiPersonFrameResult",
    "MultiPersonMediaPipeBackend",
    "MultiPersonRuntime",
    "MultiPersonRuntimeConfig",
    "MultiPersonTelemetry",
    "SeatAwareFrameResult",
    "SeatAwareRuntime",
    "SeatAwareRuntimeConfig",
    "SeatAssignment",
    "SeatFrameState",
    "SeatZone",
    "PoseDetection",
    "PersonPoseAssociation",
    "TargetCandidate",
    "TargetLockController",
    "TargetLockSnapshot",
    "associate_pose_items_to_person_boxes",
    "core_quality_summary",
    "PoseSubject",
    "anonymous_subject_id",
    "create_multi_pose_landmarker",
    "dashboard_cards",
    "dashboard_seat_cards",
    "draw_multiperson_overlay",
    "draw_seat_layout",
    "body_location",
    "frame_to_event",
    "legacy_single_person_to_event",
    "load_seat_layout",
    "parse_source",
    "resolve_performance_preset",
    "run_multiperson_demo",
    "seat_frame_to_event",
    "should_process_frame",
]
