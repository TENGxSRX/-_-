"""Human-guided, no-video CCTV collection protocol state machine."""

from __future__ import annotations

from dataclasses import dataclass


GUIDED_COLLECTION_SCHEMA_VERSION = "guided_cctv_collection_v1"


@dataclass(frozen=True)
class GuidedStage:
    stage_id: str
    instruction: str
    label: str
    seconds: float
    collect: bool

    def __post_init__(self) -> None:
        if self.seconds <= 0:
            raise ValueError("stage duration must be positive")
        if self.collect and self.label not in {"TUP", "TLB"}:
            raise ValueError("only manually verified frozen V4 labels are collectable in this protocol")


DEFAULT_GUIDED_STAGES = (
    GuidedStage("NEUTRAL", "Sit upright and hold still", "DO_NOT_TRAIN", 5.0, False),
    GuidedStage("TUP", "Hold upright posture", "TUP", 8.0, True),
    GuidedStage("TRANSITION", "Change posture; do not collect", "DO_NOT_TRAIN", 3.0, False),
    GuidedStage("TLB", "Lean backward and hold", "TLB", 8.0, True),
)


def protocol_manifest(stages: tuple[GuidedStage, ...] = DEFAULT_GUIDED_STAGES) -> dict:
    return {
        "schema_version": GUIDED_COLLECTION_SCHEMA_VERSION,
        "label_source": "MANUAL_OPERATOR_ONLY",
        "raw_video_saved": False,
        "prediction_used_as_label": False,
        "stages": [stage.__dict__.copy() for stage in stages],
    }


__all__ = ["DEFAULT_GUIDED_STAGES", "GUIDED_COLLECTION_SCHEMA_VERSION", "GuidedStage", "protocol_manifest"]
