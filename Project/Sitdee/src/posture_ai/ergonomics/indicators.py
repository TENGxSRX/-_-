"""Rule-based indicators kept separate from the trained direction model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndicatorThresholds:
    """Project settings, not medical or clinical thresholds."""

    shoulder_tilt_deg: float = 8.0
    head_forward_proxy: float = 0.20
    slouch_trunk_angle_deg: float = 20.0
    lateral_lean: float = 0.20


_DIRECTION_INDICATORS = {
    "TLF": "forward_trunk",
    "TLB": "backward_trunk",
    "TLL": "left_lateral_trunk",
    "TLR": "right_lateral_trunk",
}


def evaluate_indicators(
    geometry: dict,
    direction: str | None,
    *,
    confidence: float | None = None,
    thresholds: IndicatorThresholds | None = None,
) -> list[dict]:
    """Return core direction indicators and explicitly experimental proxies."""

    settings = thresholds or IndicatorThresholds()
    indicators: list[dict] = []
    if direction in _DIRECTION_INDICATORS:
        indicators.append({
            "name": _DIRECTION_INDICATORS[direction],
            "kind": "CORE_MODEL_STATE",
            "status": "SUPPORTED_BY_MULTIPOSTURE_LABEL",
            "direction": direction,
            "confidence": confidence,
        })
    if not geometry.get("valid"):
        return indicators

    shoulder_tilt = geometry.get("shoulder_tilt_deg")
    if shoulder_tilt is not None and abs(float(shoulder_tilt)) >= settings.shoulder_tilt_deg:
        indicators.append({
            "name": "shoulder_tilt_imbalance",
            "kind": "EXPERIMENTAL",
            "status": "GEOMETRIC_INDICATOR_ONLY",
            "value": float(shoulder_tilt),
            "threshold": settings.shoulder_tilt_deg,
            "limitation": "No clinical threshold or independent imbalance labels are available.",
        })
    head_proxy = geometry.get("head_forward_proxy")
    if head_proxy is not None and abs(float(head_proxy)) >= settings.head_forward_proxy:
        indicators.append({
            "name": "head_forward_proxy",
            "kind": "EXPERIMENTAL",
            "status": "GEOMETRIC_INDICATOR_ONLY",
            "value": float(head_proxy),
            "threshold": settings.head_forward_proxy,
            "limitation": "Camera-relative z offset; not clinically validated forward-head posture.",
        })
    trunk_angle = geometry.get("trunk_inclination_deg")
    if trunk_angle is not None and abs(float(trunk_angle)) >= settings.slouch_trunk_angle_deg:
        indicators.append({
            "name": "slouch_proxy",
            "kind": "EXPERIMENTAL",
            "status": "GEOMETRIC_INDICATOR_ONLY",
            "value": float(trunk_angle),
            "threshold": settings.slouch_trunk_angle_deg,
            "limitation": "Trunk inclination is not a validated slouch/hunch label.",
        })
    return indicators


def evaluate_directional_geometry(
    geometry: dict,
    *,
    thresholds: IndicatorThresholds | None = None,
) -> dict[str, bool | float | None]:
    """Return camera-relative geometry flags separate from the V4 AI label.

    These are engineering indicators, not extra V4 classes and not clinical
    measurements.  The left/right sign follows the supplied MediaPipe/world
    coordinate convention and must be checked during the first real-camera
    validation.
    """

    settings = thresholds or IndicatorThresholds()
    lateral = geometry.get("torso_lateral_displacement")
    forward_proxy = geometry.get("head_forward_proxy")
    return {
        "forward": bool(forward_proxy is not None and abs(float(forward_proxy)) >= settings.head_forward_proxy),
        "left": bool(lateral is not None and float(lateral) <= -settings.lateral_lean),
        "right": bool(lateral is not None and float(lateral) >= settings.lateral_lean),
        "trunk_inclination": geometry.get("trunk_inclination_deg"),
        "shoulder_tilt": geometry.get("shoulder_tilt_deg"),
        "lateral_offset": lateral,
        "forward_back_component": forward_proxy,
    }
