"""Synthetic landmark fixtures with KNOWN ground truth.

Every generator builds full 33-landmark frames (names per
app.pose.geometry.NAMES) so the demo, the API and the eval harness all
run on the same shapes a real on-device MediaPipe conversion would
produce (see docs/ARCHITECTURE.md for the y-flip note). Coordinates
use the product convention: x horizontal, y vertical (larger = higher),
in arbitrary units where a leg is ~0.95 long.

Ground truth is carried alongside the frames so tests can REPORT
measured angle errors, asymmetry errors and TUG errors honestly
without ever touching a real patient. Synthetic errors are NOT
clinical accuracy: published single-phone MediaPipe vs Qualisys mean
error is 7.16 deg (n=6, healthy, non-gait), and multi-camera OpenCap
is about 4.5 deg (see README).
"""

from __future__ import annotations

import math

from app.pose.geometry import NAMES

HIP_Y = 0.95
KNEE_Y = 0.50
ANKLE_GROUND = 0.05


def _blank() -> dict:
    return {n: {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 1.0} for n in NAMES}


def _frame(t: float, lm: dict) -> dict:
    return {"t": round(t, 3), "landmarks": lm}


def _leg_frame(hip_x: float, dev_deg: float, knee_y: float = KNEE_Y,
               hip_y: float = HIP_Y) -> tuple[dict, dict]:
    """One leg's hip/knee/ankle with a known frontal deviation (deg).

    Hip directly above knee; ankle placed so the hip-knee-ankle angle
    is exactly 180 - dev_deg.
    """
    knee = {"x": hip_x, "y": knee_y, "z": 0.0, "visibility": 1.0}
    hip = {"x": hip_x, "y": hip_y, "z": 0.0, "visibility": 1.0}
    r = math.radians(dev_deg)
    ankle = {"x": hip_x + 0.45 * math.sin(r), "y": knee_y - 0.45 * math.cos(r),
             "z": 0.0, "visibility": 1.0}
    return hip, knee, ankle


def standing_frame(hka_left_deg: float = 0.0, hka_right_deg: float = 0.0,
                   pelvic_obliquity_deg: float = 0.0,
                   trunk_lean_deg: float = 0.0) -> dict:
    """Front-view standing hold with known alignment.

    Pelvic obliquity: right hip lower by hip_separation * tan(obliquity).
    Trunk lean: shoulder midpoint shifted sideways by shoulder height
    above hips * tan(lean).
    """
    lm = _blank()
    sep = 0.30
    lx, rx = -sep / 2, sep / 2
    drop = sep * math.tan(math.radians(pelvic_obliquity_deg))
    lh, lk, la = _leg_frame(lx, hka_left_deg, hip_y=HIP_Y)
    rh, rk, ra = _leg_frame(rx, hka_right_deg, hip_y=HIP_Y - drop)
    # Recompute right knee/ankle from the dropped hip so HKA stays exact.
    rh, rk, ra = _leg_frame(rx, hka_right_deg, hip_y=HIP_Y - drop)
    # _leg_frame puts knee at KNEE_Y regardless of hip_y; keep that, so
    # the hip-knee-ankle angle stays 180-dev by construction.
    lm["left_hip"], lm["left_knee"], lm["left_ankle"] = lh, lk, la
    lm["right_hip"], lm["right_knee"], lm["right_ankle"] = rh, rk, ra
    lm["left_heel"] = dict(la); lm["right_heel"] = dict(ra)
    lm["left_foot_index"] = {"x": la["x"] + 0.05, "y": la["y"], "z": 0.0, "visibility": 1.0}
    lm["right_foot_index"] = {"x": ra["x"] + 0.05, "y": ra["y"], "z": 0.0, "visibility": 1.0}
    hip_mid_x = (lx + rx) / 2
    hip_mid_y = (HIP_Y + HIP_Y - drop) / 2
    lean_shift = 0.50 * math.tan(math.radians(trunk_lean_deg))
    lm["left_shoulder"] = {"x": hip_mid_x - 0.20 + lean_shift, "y": 1.45, "z": 0.0, "visibility": 1.0}
    lm["right_shoulder"] = {"x": hip_mid_x + 0.20 + lean_shift, "y": 1.45, "z": 0.0, "visibility": 1.0}
    lm["nose"] = {"x": hip_mid_x + lean_shift, "y": 1.70, "z": 0.0, "visibility": 1.0}
    return lm


def standing_sequence(hka_left_deg=0.0, hka_right_deg=0.0,
                      pelvic_obliquity_deg=0.0, trunk_lean_deg=0.0,
                      n: int = 10, dt: float = 0.1) -> list[dict]:
    lm = standing_frame(hka_left_deg, hka_right_deg, pelvic_obliquity_deg, trunk_lean_deg)
    return [_frame(i * dt, {k: dict(v) for k, v in lm.items()}) for i in range(n)]


def side_knee_sequence(flex_min_deg: float, flex_max_deg: float,
                       n: int = 40, duration: float = 2.0) -> list[dict]:
    """Side pass where knee flexion sweeps flex_min -> flex_max -> flex_min.

    Knee fixed; ankle placed so hip-knee-ankle angle = 180 - flexion.
    Ground truth ROM = flex_max - flex_min.
    """
    out = []
    for i in range(n):
        t = duration * i / max(1, n - 1)
        phase = 0.5 - 0.5 * math.cos(2 * math.pi * t / duration)
        flex = flex_min_deg + (flex_max_deg - flex_min_deg) * phase
        ang = math.radians(180.0 - flex)
        lm = _blank()
        lm["left_hip"] = {"x": 0.0, "y": HIP_Y, "z": 0.0, "visibility": 1.0}
        lm["right_hip"] = {"x": 0.0, "y": HIP_Y, "z": 0.0, "visibility": 1.0}
        lm["left_knee"] = {"x": 0.0, "y": KNEE_Y, "z": 0.0, "visibility": 1.0}
        lm["right_knee"] = {"x": 0.0, "y": KNEE_Y, "z": 0.0, "visibility": 1.0}
        lm["left_ankle"] = {"x": 0.45 * math.sin(ang), "y": KNEE_Y + 0.45 * math.cos(ang),
                            "z": 0.0, "visibility": 1.0}
        lm["right_ankle"] = dict(lm["left_ankle"])
        lm["left_shoulder"] = {"x": 0.0, "y": 1.45, "z": 0.0, "visibility": 1.0}
        lm["right_shoulder"] = {"x": 0.0, "y": 1.45, "z": 0.0, "visibility": 1.0}
        out.append(_frame(t, lm))
    return out


def side_hip_sequence(flex_min_deg: float, flex_max_deg: float,
                      n: int = 40, duration: float = 2.0) -> list[dict]:
    """Side pass where hip flexion sweeps flex_min -> flex_max -> flex_min.

    Shoulder directly above hip; knee placed so shoulder-hip-knee
    angle = 180 - flexion. Ground truth hip ROM = flex_max - flex_min.
    """
    out = []
    for i in range(n):
        t = duration * i / max(1, n - 1)
        phase = 0.5 - 0.5 * math.cos(2 * math.pi * t / duration)
        flex = flex_min_deg + (flex_max_deg - flex_min_deg) * phase
        r = math.radians(flex)
        lm = _blank()
        lm["left_shoulder"] = {"x": 0.0, "y": 1.45, "z": 0.0, "visibility": 1.0}
        lm["right_shoulder"] = {"x": 0.0, "y": 1.45, "z": 0.0, "visibility": 1.0}
        lm["left_hip"] = {"x": 0.0, "y": HIP_Y, "z": 0.0, "visibility": 1.0}
        lm["right_hip"] = {"x": 0.0, "y": HIP_Y, "z": 0.0, "visibility": 1.0}
        lm["left_knee"] = {"x": 0.50 * math.sin(r), "y": HIP_Y - 0.50 * math.cos(r),
                           "z": 0.0, "visibility": 1.0}
        lm["right_knee"] = dict(lm["left_knee"])
        # Ankle below knee (straight shank) so knee angle does not confuse.
        lm["left_ankle"] = {"x": lm["left_knee"]["x"], "y": lm["left_knee"]["y"] - 0.45,
                            "z": 0.0, "visibility": 1.0}
        lm["right_ankle"] = dict(lm["left_ankle"])
        out.append(_frame(t, lm))
    return out


def _ankle_y_in_cycle(phase: float, stance_frac: float) -> float:
    if phase < stance_frac:
        return ANKLE_GROUND
    sw = (phase - stance_frac) / max(1e-6, 1.0 - stance_frac)
    return ANKLE_GROUND + 0.20 * math.sin(math.pi * sw)


def front_gait_sequence(stance_left_s: float = 0.60, stance_right_s: float = 0.60,
                        cycle_s: float = 1.0, duration: float = 4.0, dt: float = 0.05,
                        trendelenburg_deg: float = 0.0,
                        varus_thrust_range: tuple[float, float] | None = None) -> list[dict]:
    """Front pass with known per-leg stance durations.

    Ground truth stance asymmetry = |L-R| / mean * 100. Optional
    Trendelenburg: pelvis tilts by trendelenburg_deg whenever the
    opposite foot is in swing. Optional varus thrust: frontal knee
    deviation during stance sweeps the given (min, max) range.
    """
    out = []
    n = int(duration / dt) + 1
    for i in range(n):
        t = i * dt
        lm = _blank()
        # Which leg is in swing determines pelvic tilt.
        ph_l = (t / cycle_s) % 1.0
        ph_r = ((t / cycle_s) + 0.5) % 1.0
        swing_other = None
        if trendelenburg_deg:
            # Single support on left = right in swing.
            if ph_r >= stance_right_s / cycle_s:
                swing_other = "right"
            elif ph_l >= stance_left_s / cycle_s:
                swing_other = "left"
        drop = 0.0
        if swing_other == "right":
            drop = 0.30 * math.tan(math.radians(trendelenburg_deg))
        elif swing_other == "left":
            drop = -0.30 * math.tan(math.radians(trendelenburg_deg))
        lh_y = HIP_Y
        rh_y = HIP_Y - drop
        for leg, ph, stance_s, hip_x, hip_y in (
                ("left", ph_l, stance_left_s, -0.15, lh_y),
                ("right", ph_r, stance_right_s, 0.15, rh_y)):
            y = _ankle_y_in_cycle(ph, stance_s / cycle_s)
            dev = 0.0
            if varus_thrust_range:
                # Deviation sweeps min->max across the stance interval.
                # The toe-off frame (ph == stance fraction) keeps the
                # end-of-stance value: at toe-off the foot is still on
                # the ground, so its alignment is stance alignment.
                if ph <= stance_s / cycle_s + 1e-9:
                    frac = min(1.0, ph / max(1e-6, stance_s / cycle_s))
                    dev = varus_thrust_range[0] + (varus_thrust_range[1] - varus_thrust_range[0]) * frac
            hip, knee, ankle = _leg_frame(hip_x, dev, hip_y=hip_y)
            # Override ankle height with the gait trajectory, keeping x.
            ankle = {"x": ankle["x"], "y": y, "z": 0.0, "visibility": 1.0}
            knee = {"x": hip_x, "y": KNEE_Y, "z": 0.0, "visibility": 1.0}
            lm[f"{leg}_hip"] = hip
            lm[f"{leg}_knee"] = knee
            lm[f"{leg}_ankle"] = ankle
            lm[f"{leg}_heel"] = dict(ankle)
            lm[f"{leg}_foot_index"] = {"x": ankle["x"] + 0.05, "y": y, "z": 0.0, "visibility": 1.0}
        hip_mid_x = 0.0
        lm["left_shoulder"] = {"x": hip_mid_x - 0.20, "y": 1.45, "z": 0.0, "visibility": 1.0}
        lm["right_shoulder"] = {"x": hip_mid_x + 0.20, "y": 1.45, "z": 0.0, "visibility": 1.0}
        lm["nose"] = {"x": 0.0, "y": 1.70, "z": 0.0, "visibility": 1.0}
        out.append(_frame(t, lm))
    return out


def tug_sequence(stand_t: float = 1.2, turn_start_t: float = 5.0,
                 turn_end_t: float = 7.0, sit_t: float = 10.4,
                 dt: float = 0.1) -> tuple[list[dict], dict]:
    """Full TUG with known phase boundaries (markers ground truth).

    Hip height: sitting 0.55 -> standing 0.95 over sit-to-stand, back
    down over the final 0.8 s. Shoulder width: 0.40 frontal, 0.22
    during the turn. Ankles walk (stance/swing + advancing z) during
    walk phases, grounded otherwise.
    """
    markers = {"start": 0.0, "stand": stand_t, "turn_start": turn_start_t,
               "turn_end": turn_end_t, "sit": sit_t}
    out = []
    n = int(sit_t / dt) + 1
    for i in range(n):
        t = round(i * dt, 3)
        if t < stand_t:
            frac = t / stand_t if stand_t else 1.0
            hip_h = 0.55 + 0.40 * frac
        elif t < sit_t - 0.8:
            hip_h = 0.95
        elif t <= sit_t:
            frac = (t - (sit_t - 0.8)) / 0.8
            hip_h = 0.95 - 0.40 * frac
        else:
            hip_h = 0.55
        if turn_start_t <= t < turn_end_t:
            half_w = 0.11
        else:
            half_w = 0.20
        walking = (stand_t <= t < turn_start_t) or (turn_end_t <= t <= sit_t)
        # Walk progress for z: out 0->1, back 1->0.
        if stand_t <= t < turn_start_t:
            prog = (t - stand_t) / max(1e-6, turn_start_t - stand_t)
        elif turn_end_t <= t <= sit_t:
            prog = 1.0 - (t - turn_end_t) / max(1e-6, sit_t - turn_end_t)
        else:
            prog = 0.0
        lm = _blank()
        for leg, hip_x, offset in (("left", -0.15, 0.0), ("right", 0.15, 0.5)):
            if walking:
                ph = ((t / 1.0) + offset) % 1.0
                y = _ankle_y_in_cycle(ph, 0.60)
            else:
                y = ANKLE_GROUND
            lm[f"{leg}_hip"] = {"x": hip_x, "y": hip_h, "z": 0.0, "visibility": 1.0}
            lm[f"{leg}_knee"] = {"x": hip_x, "y": hip_h - 0.45, "z": 0.0, "visibility": 1.0}
            lm[f"{leg}_ankle"] = {"x": hip_x, "y": y, "z": 2.0 * prog, "visibility": 1.0}
            lm[f"{leg}_heel"] = dict(lm[f"{leg}_ankle"])
        lm["left_shoulder"] = {"x": -half_w, "y": hip_h + 0.50, "z": 0.0, "visibility": 1.0}
        lm["right_shoulder"] = {"x": half_w, "y": hip_h + 0.50, "z": 0.0, "visibility": 1.0}
        lm["nose"] = {"x": 0.0, "y": hip_h + 0.75, "z": 0.0, "visibility": 1.0}
        out.append(_frame(t, lm))
    return out, markers


def side_combined_sequence(knee_flex_min: float = 5.0, knee_flex_max: float = 60.0,
                           hip_flex_min: float = 5.0, hip_flex_max: float = 25.0,
                           n: int = 40, duration: float = 2.0) -> list[dict]:
    """Side pass where knee AND hip flexion sweep together (phase-shifted).

    Hip at origin height; knee placed from hip flexion, ankle placed
    from knee flexion by rotating the knee->hip direction. Ground truth
    knee ROM = knee_flex_max - knee_flex_min, hip ROM likewise. This is
    the side pass the arthritis demo uses, so both ROM flags are real.
    """
    out = []
    for i in range(n):
        t = duration * i / max(1, n - 1)
        ph_k = 0.5 - 0.5 * math.cos(2 * math.pi * t / duration)
        ph_h = 0.5 - 0.5 * math.cos(2 * math.pi * t / duration + 0.6)
        kf = knee_flex_min + (knee_flex_max - knee_flex_min) * ph_k
        hf = hip_flex_min + (hip_flex_max - hip_flex_min) * ph_h
        lm = _blank()
        lm["left_shoulder"] = {"x": 0.0, "y": 1.45, "z": 0.0, "visibility": 1.0}
        lm["right_shoulder"] = {"x": 0.0, "y": 1.45, "z": 0.0, "visibility": 1.0}
        lm["left_hip"] = {"x": 0.0, "y": HIP_Y, "z": 0.0, "visibility": 1.0}
        lm["right_hip"] = {"x": 0.0, "y": HIP_Y, "z": 0.0, "visibility": 1.0}
        r = math.radians(hf)
        kx, ky = 0.50 * math.sin(r), HIP_Y - 0.50 * math.cos(r)
        lm["left_knee"] = {"x": kx, "y": ky, "z": 0.0, "visibility": 1.0}
        lm["right_knee"] = {"x": kx, "y": ky, "z": 0.0, "visibility": 1.0}
        theta = math.radians(kf + hf - 90.0)
        ax, ay = kx + 0.45 * math.cos(theta), ky + 0.45 * math.sin(theta)
        lm["left_ankle"] = {"x": ax, "y": ay, "z": 0.0, "visibility": 1.0}
        lm["right_ankle"] = {"x": ax, "y": ay, "z": 0.0, "visibility": 1.0}
        out.append(_frame(t, lm))
    return out
