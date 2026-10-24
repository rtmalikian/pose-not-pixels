"""Deterministic pose geometry.

Coordinate convention for v0.1.0 landmark JSON (documented in README
and docs/ARCHITECTURE.md): ``x`` is horizontal, ``y`` is vertical with
LARGER = HIGHER (mathematical up). MediaPipe image coordinates use
y-down; the documented on-device integration path flips y when
converting MediaPipe output to this JSON. All angles below are
independent of that choice up to sign, because they use vector geometry.

Landmarks are dicts ``{"x": float, "y": float, "z": float|None,
"visibility": float}`` keyed by MediaPipe landmark name (see NAMES).
Missing / low-visibility landmarks return None instead of a guessed
number — callers degrade gracefully, never invent a joint.
"""

from __future__ import annotations

import math

NAMES: list[str] = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]
assert len(NAMES) == 33

INDEX = {name: i for i, name in enumerate(NAMES)}


def _pt(frame: dict, name: str):
    lm = (frame or {}).get(name)
    if not lm:
        return None
    if lm.get("visibility", 1.0) < 0.3:
        return None
    try:
        return (float(lm["x"]), float(lm["y"]),
                float(lm["z"]) if lm.get("z") is not None else None)
    except (KeyError, TypeError, ValueError):
        return None


def get(frame: dict, name: str):
    return _pt(frame, name)


def dist2d(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_at(a, b, c) -> float | None:
    """Angle in degrees at joint ``b`` between segments b->a and b->c.

    Uses 3D vectors when all three points carry z, otherwise 2D.
    Straight limb = 180 deg.
    """
    if a is None or b is None or c is None:
        return None
    if a[2] is not None and b[2] is not None and c[2] is not None:
        v1 = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
        v2 = (c[0] - b[0], c[1] - b[1], c[2] - b[2])
        n1 = math.sqrt(sum(v * v for v in v1))
        n2 = math.sqrt(sum(v * v for v in v2))
        if n1 == 0 or n2 == 0:
            return None
        cosang = max(-1.0, min(1.0, sum(x * y for x, y in zip(v1, v2)) / (n1 * n2)))
        return math.degrees(math.acos(cosang))
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosang))


def signed_knee_deviation(frame: dict, leg: str) -> dict | None:
    """Frontal hip-knee-ankle (HKA) proxy for one leg.

    Returns {"angle_deg", "deviation_deg", "signed_deviation_deg",
    "direction"} where deviation = 180 - angle (0 = straight). The
    signed value is the knee's offset from the hip-ankle line: positive
    = knee displaced toward the body's lateral side is NOT asserted
    from 2D alone, so ``direction`` is reported only as
    "lateral_offset" / "medial_offset" relative to the hip-ankle line
    for that leg's image side, and flags fire on magnitude only.
    """
    hip = _pt(frame, f"{leg}_hip")
    knee = _pt(frame, f"{leg}_knee")
    ankle = _pt(frame, f"{leg}_ankle")
    ang = angle_at(hip, knee, ankle)
    if ang is None:
        return None
    # Signed offset: cross product of (ankle-hip) with (knee-hip), z.
    cross = (ankle[0] - hip[0]) * (knee[1] - hip[1]) - (ankle[1] - hip[1]) * (knee[0] - hip[0])
    # Distance of knee from hip-ankle line, as a fraction of leg length.
    leg_len = dist2d(hip, knee) + dist2d(knee, ankle)
    line_len = dist2d(hip, ankle) or 1.0
    signed = (180.0 - ang) * (1 if cross >= 0 else -1)
    return {
        "angle_deg": round(ang, 2),
        "deviation_deg": round(abs(180.0 - ang), 2),
        # Frontal HKA is never reported to 0.1 deg precision as if
        # radiographic: a +/-3 deg (or greater) band always travels
        # with it. Ge 2025 (OpenPose, severe pre-surgical knee OA
        # only): mean absolute difference 1.44 deg, 95% limits about
        # +/-3 deg — narrow scope, NOT generalised to MediaPipe or
        # to mild/moderate OA.
        "uncertainty_band_deg": 3.0,
        "signed_deviation_deg": round(signed, 2),
        "direction": "offset_positive" if cross >= 0 else "offset_negative",
        "knee_offset_fraction": round(abs(cross) / line_len / (leg_len or 1.0), 4) if leg_len else 0.0,
    }


def leg_length(frame: dict, leg: str) -> float | None:
    hip = _pt(frame, f"{leg}_hip")
    knee = _pt(frame, f"{leg}_knee")
    ankle = _pt(frame, f"{leg}_ankle")
    if hip is None or knee is None or ankle is None:
        return None
    return dist2d(hip, knee) + dist2d(knee, ankle)


def midpoint(frame: dict, a: str, b: str):
    pa, pb = _pt(frame, a), _pt(frame, b)
    if pa is None or pb is None:
        return None
    return ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2,
            (pa[2] + pb[2]) / 2 if pa[2] is not None and pb[2] is not None else None)


def line_angle_deg(p, q) -> float | None:
    """Angle of segment p->q vs horizontal, degrees. 0 = level."""
    if p is None or q is None:
        return None
    return math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))


def vertical_angle_deg(p, q) -> float | None:
    """Angle of segment p->q vs vertical, degrees. 0 = upright."""
    if p is None or q is None:
        return None
    dx, dy = q[0] - p[0], q[1] - p[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))
