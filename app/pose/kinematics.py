"""Deterministic kinematics: TUG (Pack 1) + arthritis screening (Pack 2).

Everything here is ordinary arithmetic over landmark JSON. No model
call, no video, no guessing: a metric is None when its landmarks are
missing, and a flag is only computed when its metric is present.

Thresholds: see THRESHOLDS. Every threshold except the STEADI TUG
screening flag is app-defined and unvalidated, and is labelled
``APP_DEFINED_UNVALIDATED`` in every output (``threshold_status``),
in README, SECURITY and the UI. The STEADI >=12 s value is a screening
flag only, never a risk label. No validated performance values exist
in this product — see the test that scans the docs for a percentage
attached to either word.
"""

from __future__ import annotations

import math
import statistics

from app.pose import geometry as G

# ---------------------------------------------------------------------------
# Screening thresholds — named constant, source note per threshold.
# ALL thresholds are UNVALIDATED in this product until tested on
# labelled clinical data. A flag is a prompt for clinician review,
# never a finding about a condition.
# ---------------------------------------------------------------------------
THRESHOLDS: dict[str, dict] = {
    # Status values: "STEADI_SCREENING_FLAG_ONLY" is the ONLY citable
    # reference point in this dict, and only as a screening flag.
    # Every other threshold is "APP_DEFINED_UNVALIDATED": an app-defined
    # screening choice with NO citable source as a screening cutoff.
    # No threshold here is a validated cutoff, and none is presented
    # as sourced from the literature.
    "tug_total_s": {
        "amber": None, "red": 12.0, "unit": "s",
        "source": "CDC STEADI uses >=12 s only as a screening flag for "
                  "further assessment, never as a risk label. Barry 2014 "
                  "found sensitivity 0.31 at 13.5 s, and Schoene 2013 "
                  "(53 studies) concluded no cut-point can be recommended. "
                  "This product uses >=12 s only with the label "
                  "'STEADI screening flag \u2014 discuss with clinician'.",
        "status": "STEADI_SCREENING_FLAG_ONLY",
    },
    "hka_deviation_deg": {
        "amber": 3.0, "red": 5.0, "unit": "deg",
        "source": "Varus alignment is associated with OA progression in "
                  "the literature, but no degree threshold is cited as a "
                  "screening cutoff. The 3 deg amber / 5 deg red split "
                  "is app-defined.",
        "status": "APP_DEFINED_UNVALIDATED",
    },
    "pelvic_obliquity_deg": {
        "amber": 3.0, "red": 5.0, "unit": "deg",
        "source": "App-defined screening choice (provisional); no citable "
                  "source as a screening cutoff.",
        "status": "APP_DEFINED_UNVALIDATED",
    },
    "lateral_trunk_lean_deg": {
        "amber": 3.0, "red": 5.0, "unit": "deg",
        "source": "App-defined screening choice (provisional); no citable "
                  "source as a screening cutoff.",
        "status": "APP_DEFINED_UNVALIDATED",
    },
    "knee_flexion_rom_deg": {
        "amber": 50.0, "red": 40.0, "unit": "deg", "low_is_bad": True,
        "source": "App-defined screening choice (provisional); no citable "
                  "source as a screening cutoff.",
        "status": "APP_DEFINED_UNVALIDATED",
    },
    "hip_rom_deg": {
        "amber": 30.0, "red": 20.0, "unit": "deg", "low_is_bad": True,
        "source": "App-defined screening choice (provisional); no citable "
                  "source as a screening cutoff.",
        "status": "APP_DEFINED_UNVALIDATED",
    },
    "varus_thrust_deg": {
        "amber": 3.0, "red": 5.0, "unit": "deg",
        "source": "App-defined screening choice (provisional); no citable "
                  "source as a screening cutoff for varus-thrust degrees.",
        "status": "APP_DEFINED_UNVALIDATED",
    },
    "trendelenburg_drop_deg": {
        "amber": 4.0, "red": 8.0, "unit": "deg",
        "source": "Static-test Trendelenburg range 2-8 deg is contested "
                  "and not gait-validated; a >=4 deg flag here is "
                  "borrowed/app-defined, not a validated gait cutoff.",
        "status": "APP_DEFINED_UNVALIDATED",
    },
    "stance_time_asymmetry_pct": {
        "amber": 10.0, "red": 20.0, "unit": "%",
        "source": "App-defined screening choice (provisional); no citable "
                  "source as a screening cutoff for asymmetry percent.",
        "status": "APP_DEFINED_UNVALIDATED",
    },
    "stride_length_asymmetry_pct": {
        "amber": 10.0, "red": 20.0, "unit": "%",
        "source": "App-defined screening choice (provisional); no citable "
                  "source as a screening cutoff for asymmetry percent.",
        "status": "APP_DEFINED_UNVALIDATED",
    },
    "cadence_steps_per_min": {
        "amber": 100.0, "red": 80.0, "unit": "steps/min", "low_is_bad": True,
        "source": "App-defined screening choice (provisional); no citable "
                  "source as a screening cutoff.",
        "status": "APP_DEFINED_UNVALIDATED",
    },
    "step_width_norm": {
        "amber": 0.25, "red": 0.40, "unit": "leg_lengths",
        "source": "App-defined screening choice (provisional); no citable "
                  "source as a screening cutoff for step width.",
        "status": "APP_DEFINED_UNVALIDATED",
    },
}


def _traffic(metric_key: str, value: float | None) -> str | None:
    if value is None:
        return None
    spec = THRESHOLDS[metric_key]
    if spec.get("low_is_bad"):
        if spec["red"] is not None and value < spec["red"]:
            return "red"
        if spec["amber"] is not None and value < spec["amber"]:
            return "amber"
        return "green"
    if spec["red"] is not None and value > spec["red"]:
        return "red"
    if spec["amber"] is not None and value > spec["amber"]:
        return "amber"
    return "green"


def _asymmetry_pct(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    mean = (abs(left) + abs(right)) / 2
    if mean == 0:
        return 0.0
    return round(abs(left - right) / mean * 100.0, 2)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _lm(frame: dict) -> dict:
    """Unwrap a {"t", "landmarks"} frame to its landmarks dict."""
    if isinstance(frame, dict) and "landmarks" in frame:
        return frame["landmarks"]
    return frame


def _hip_height(frame: dict) -> float | None:
    mid = G.midpoint(_lm(frame), "left_hip", "right_hip")
    return mid[1] if mid else None


def _shoulder_width(frame: dict) -> float | None:
    lm = _lm(frame)
    ls, rs = G.get(lm, "left_shoulder"), G.get(lm, "right_shoulder")
    if ls is None or rs is None:
        return None
    return G.dist2d(ls, rs)


def _ankle(frame: dict, leg: str):
    return G.get(_lm(frame), f"{leg}_ankle")


# ---------------------------------------------------------------------------
# Step / contact detection from ankle trajectories (code, not a model)
# ---------------------------------------------------------------------------

def _contacts(frames: list[dict], leg: str) -> list[int]:
    """Indices of foot-contact frames: the START of each contiguous run
    of frames where ankle height sits near the ground (bottom quartile
    of that ankle's range). Run-based, so a stance plateau counts as
    one contact, not one per frame. Returns [] when flat/missing."""
    ys = []
    for f in frames:
        a = _ankle(f, leg)
        ys.append(a[1] if a else None)
    if any(y is None for y in ys) or len(ys) < 5:
        return []
    lo, hi = min(ys), max(ys)
    if hi - lo < 1e-6:
        return []
    thresh = lo + 0.25 * (hi - lo)
    out = []
    in_run = False
    for i, y in enumerate(ys):
        if y <= thresh and not in_run:
            out.append(i)
            in_run = True
        elif y > thresh:
            in_run = False
    return out


def _toe_off(frames: list[dict], leg: str, contact_idx: int) -> int | None:
    ys = []
    for f in frames:
        a = _ankle(f, leg)
        ys.append(a[1] if a else None)
    if any(y is None for y in ys):
        return None
    lo, hi = min(ys), max(ys)
    thresh = lo + 0.35 * (hi - lo)
    for i in range(contact_idx + 1, len(ys)):
        if ys[i] > thresh:
            return i
    return None


def _stance_stats(frames: list[dict], leg: str):
    """Mean stance duration (s) and mean stride length (x/z-units) per
    leg, plus step count, from contacts. Stride uses the x-z plane so
    it works for a side pass (forward = x) and a front pass walking
    toward the camera (forward = z). Stance duration is measured by
    interpolating the ankle-height crossings of a TIGHT threshold
    (ground + 5% of range) — a loose threshold would add a slice of
    swing whose length differs per leg and corrupt asymmetry.
    The first contact's duration is skipped: a capture may start
    mid-stance, so it is the one duration that can be truncated.
    None-safe."""
    contacts = _contacts(frames, leg)
    if len(contacts) < 1:
        return {"stance_s": None, "stride_x": None, "steps": 0, "contacts": contacts}
    ys = []
    for f in frames:
        a = _ankle(f, leg)
        ys.append(a[1] if a else None)
    lo, hi = min(ys), max(ys)
    tight = lo + 0.05 * (hi - lo)
    diffs = sorted(frames[i + 1]["t"] - frames[i]["t"]
                   for i in range(len(frames) - 1))
    med_dt = diffs[len(diffs) // 2] if diffs else 0.0

    def _cross_time(i_from: int, i_to: int) -> float:
        """Interpolated time when y crosses ``tight`` between frames."""
        y0, y1 = ys[i_from], ys[i_to]
        if y1 == y0:
            return frames[i_to]["t"]
        frac = (tight - y0) / (y1 - y0)
        frac = max(0.0, min(1.0, frac))
        return frames[i_from]["t"] + frac * (frames[i_to]["t"] - frames[i_from]["t"])

    durations = []
    for c in contacts:
        if c == 0:
            continue
        # Contact time: crossing down through tight before frame c.
        if ys[c - 1] > tight:
            t_contact = _cross_time(c - 1, c)
        else:
            t_contact = frames[c]["t"]
        off = None
        for i in range(c + 1, len(ys)):
            if ys[i] > tight:
                off = i
                break
        if off is None:
            continue
        t_off = _cross_time(off - 1, off)
        dur = t_off - t_contact
        # Skip a stance whose toe-off crosses a time gap (e.g. the
        # TUG turn between two walk intervals).
        if med_dt and dur > (off - c + 1) * med_dt * 1.5:
            continue
        durations.append(dur)
    strides = []
    for a, b in zip(contacts, contacts[1:]):
        pa, pb = _ankle(frames[a], leg), _ankle(frames[b], leg)
        if pa and pb:
            dx = pb[0] - pa[0]
            dz = (pb[2] or 0.0) - (pa[2] or 0.0)
            strides.append(math.hypot(dx, dz))
    return {
        "stance_s": round(statistics.mean(durations), 4) if durations else None,
        "stride_x": round(statistics.mean(strides), 4) if strides else None,
        "steps": len(contacts),
        "contacts": contacts,
    }


def _segmented_stance(frames: list[dict], leg: str) -> dict:
    """Stance stats over contiguous time segments (a TUG walk is two
    segments separated by the turn). Combining raw frames across the
    turn gap would invent stance durations spanning the turn."""
    if not frames:
        return {"stance_s": None, "stride_x": None, "steps": 0}
    segments, cur = [], [frames[0]]
    for prev, f in zip(frames, frames[1:]):
        if f["t"] - prev["t"] > 0.3:
            segments.append(cur)
            cur = [f]
        else:
            cur.append(f)
    segments.append(cur)
    stances, strides, steps = [], [], 0
    for seg in segments:
        st = _stance_stats(seg, leg)
        if st["stance_s"] is not None:
            stances.append(st["stance_s"])
        if st["stride_x"] is not None:
            strides.append(st["stride_x"])
        steps += st["steps"]
    return {
        "stance_s": round(statistics.mean(stances), 4) if stances else None,
        "stride_x": round(statistics.mean(strides), 4) if strides else None,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Pack 1 — TUG
# ---------------------------------------------------------------------------

def infer_tug_boundaries(frames: list[dict]) -> dict:
    """Heuristic phase inference (documented heuristic, markers preferred).

    Signatures used, in order:
    * sit-to-stand end: hip height first reaches 95% of the standing
      plateau (median of the top-decile hip heights) and stays there.
    * turn start: shoulder width first drops below 85% of the frontal
      median (body rotating away from the camera).
    * turn end: shoulder width first recovers above 90% of frontal median.
    * sit: hip height first drops below 85% of the standing plateau
      after the turn and stays low.
    Returns boundary times; missing signatures fall back to quartiles
    of the capture so the function never crashes — callers record
    ``method: inferred`` and the README states markers are preferred.
    """
    ts = [f["t"] for f in frames]
    heights = [_hip_height(f) for f in frames]
    widths = [_shoulder_width(f) for f in frames]
    valid_h = [h for h in heights if h is not None]
    valid_w = [w for w in widths if w is not None]
    if not valid_h or len(frames) < 5:
        return {"start": ts[0], "stand": ts[0], "turn_start": ts[0],
                "turn_end": ts[-1], "sit": ts[-1], "method": "inferred_fallback"}
    standing = statistics.median(sorted(valid_h)[-max(1, len(valid_h) // 10):])
    sitting_floor = min(valid_h)
    stand = None
    for i, h in enumerate(heights):
        if h is not None and h >= standing - 0.05 * (standing - sitting_floor):
            stand = ts[i]
            break
    stand = stand if stand is not None else ts[len(ts) // 4]
    frontal_w = _median(valid_w) or 0.0
    turn_start = None
    for i, w in enumerate(widths):
        if ts[i] > stand and w is not None and frontal_w and w < 0.85 * frontal_w:
            turn_start = ts[i]
            break
    turn_start = turn_start if turn_start is not None else ts[len(ts) // 2]
    turn_end = None
    for i, w in enumerate(widths):
        if ts[i] > turn_start and w is not None and frontal_w and w > 0.90 * frontal_w:
            turn_end = ts[i]
            break
    turn_end = turn_end if turn_end is not None else ts[(3 * len(ts)) // 4]
    # Sit = when hip height ARRIVES near the sitting floor (not when
    # the sit motion starts), so inferred total time matches the
    # stopwatch definition: seated at the end of the test.
    sit = None
    for i, h in enumerate(heights):
        if ts[i] > turn_end and h is not None and h <= sitting_floor + 0.05 * (standing - sitting_floor):
            sit = ts[i]
            break
    sit = sit if sit is not None else ts[-1]
    return {"start": ts[0], "stand": stand, "turn_start": turn_start,
            "turn_end": turn_end, "sit": sit, "method": "inferred"}


def analyse_tug(frames: list[dict], markers: dict | None = None) -> dict:
    """TUG metrics from a landmark sequence.

    ``markers`` (preferred): {"start","stand","turn_start","turn_end",
    "sit"} as absolute times matching frame timestamps. When omitted,
    phases are inferred by ``infer_tug_boundaries`` (heuristic).
    """
    frames = sorted(frames, key=lambda f: f["t"])
    if not frames:
        raise ValueError("empty frame sequence")
    if markers:
        b = {"start": markers["start"], "stand": markers["stand"],
             "turn_start": markers["turn_start"], "turn_end": markers["turn_end"],
             "sit": markers["sit"], "method": "markers"}
    else:
        b = infer_tug_boundaries(frames)
    total = b["sit"] - b["start"]
    sit_to_stand = b["stand"] - b["start"]
    walk = (b["turn_start"] - b["stand"]) + (b["sit"] - b["turn_end"])
    turn = b["turn_end"] - b["turn_start"]
    # Walk-phase frames for gait proxies.
    walk_frames = [f for f in frames
                   if (b["stand"] <= f["t"] < b["turn_start"])
                   or (b["turn_end"] <= f["t"] <= b["sit"])]
    leg_lengths = []
    for f in frames:
        for leg in ("left", "right"):
            ll = G.leg_length(_lm(f), leg)
            if ll:
                leg_lengths.append(ll)
    leg_len = _median(leg_lengths) or 1.0
    stats = {leg: _segmented_stance(walk_frames, leg) for leg in ("left", "right")}
    steps = stats["left"]["steps"] + stats["right"]["steps"]
    stride_norm = None
    strides = [s["stride_x"] for s in stats.values() if s["stride_x"]]
    if strides:
        stride_norm = round(statistics.mean(strides) / leg_len, 4)
    walk_time = walk if walk > 0 else None
    cadence = round(steps / walk_time * 60.0, 2) if walk_time and steps else None
    # Gait speed proxy: stride length (normalised) per mean step time.
    gait_speed = None
    if stride_norm is not None and cadence:
        gait_speed = round(stride_norm * cadence / 60.0, 4)
    stance_asym = _asymmetry_pct(stats["left"]["stance_s"], stats["right"]["stance_s"])
    tug_flag = _traffic("tug_total_s", total)
    return {
        "tug_total_s": round(total, 3),
        "sit_to_stand_s": round(sit_to_stand, 3),
        "walk_s": round(walk, 3),
        "turn_s": round(turn, 3),
        "gait_speed_proxy_leg_lengths_per_s": gait_speed,
        "stride_length_norm_leg_lengths": stride_norm,
        "stance_time_asymmetry_pct": stance_asym,
        "step_count_estimate": steps,
        "cadence_steps_per_min": cadence,
        "phase_method": b["method"],
        "boundaries": {k: round(v, 3) if isinstance(v, float) else v
                       for k, v in b.items() if k != "method"},
        "tug_flag": "steadi_screening_flag" if tug_flag == "red" else "below_steadi_flag",
        "tug_flag_label": ("STEADI screening flag \u2014 discuss with clinician"
                           if tug_flag == "red" else
                           "Below the STEADI screening flag"),
        "tug_caveat": ("CDC STEADI uses >=12 s only as a screening flag for "
                       "further assessment. Barry 2014 found sensitivity 0.31 "
                       "at 13.5 s; Schoene 2013 (53 studies) concluded no "
                       "cut-point can be recommended. Flag for clinician "
                       "review only, with the measured time and its uncertainty."),
        "tug_threshold": THRESHOLDS["tug_total_s"],
    }


# ---------------------------------------------------------------------------
# Pack 2 — Arthritis kinematics screening
# ---------------------------------------------------------------------------

def _average_frame(frames: list[dict]) -> dict:
    """Mean landmark position across frames (visibility-weighted by
    dropping low-visibility landmarks per frame)."""
    names: set[str] = set()
    for f in frames:
        names.update(f.get("landmarks", {}).keys())
    out: dict[str, dict] = {}
    for name in names:
        xs, ys, zs, vs = [], [], [], []
        for f in frames:
            lm = f.get("landmarks", {}).get(name)
            if lm and lm.get("visibility", 1.0) >= 0.3:
                xs.append(float(lm["x"])); ys.append(float(lm["y"]))
                if lm.get("z") is not None:
                    zs.append(float(lm["z"]))
                vs.append(float(lm.get("visibility", 1.0)))
        if xs:
            out[name] = {"x": statistics.mean(xs), "y": statistics.mean(ys),
                         "z": statistics.mean(zs) if zs else None,
                         "visibility": statistics.mean(vs)}
    return out


def _evidence_summary(frame: dict, names: list[str]) -> dict:
    out = {}
    for n in names:
        lm = G.get(frame, n)
        out[n] = ({"x": round(lm[0], 3), "y": round(lm[1], 3)}
                  if lm else None)
    return out


def analyse_arthritis(standing_frames: list[dict],
                      gait_passes: list[dict]) -> dict:
    """standing_frames: front-view hold frames.
    gait_passes: [{"view": "front"|"side", "frames": [...]}].
    A metric is None when its pass/view is absent — never guessed."""
    standing = _average_frame(standing_frames) if standing_frames else {}
    result: dict = {"hka": {}, "flags": []}
    leg_len = None
    for leg in ("left", "right"):
        hka = G.signed_knee_deviation(standing, leg) if standing else None
        ll = G.leg_length(standing, leg) if standing else None
        if ll:
            leg_len = ll if leg_len is None else max(leg_len, ll)
        result["hka"][leg] = hka
    # Pelvic obliquity: hip line vs horizontal (front standing hold).
    lh, rh = G.get(standing, "left_hip"), G.get(standing, "right_hip")
    obliq = None
    if lh and rh:
        obliq = abs(G.line_angle_deg(lh, rh) or 0.0)
    result["pelvic_obliquity_deg"] = round(obliq, 2) if obliq is not None else None
    # Lateral trunk lean: shoulder-mid -> hip-mid vs vertical.
    smid = G.midpoint(standing, "left_shoulder", "right_shoulder")
    hmid = G.midpoint(standing, "left_hip", "right_hip")
    lean = None
    if smid and hmid:
        lean = G.vertical_angle_deg(hmid, smid)
    result["lateral_trunk_lean_deg"] = round(lean, 2) if lean is not None else None
    # Mid-hip offset vs base-of-support midpoint, normalised.
    offset_norm = None
    if hmid:
        la, ra = G.get(standing, "left_ankle"), G.get(standing, "right_ankle")
        if la and ra and leg_len:
            base_mid = (la[0] + ra[0]) / 2
            offset_norm = round(abs(hmid[0] - base_mid) / leg_len, 4)
    result["mid_hip_offset_norm"] = offset_norm

    front_frames: list[dict] = []
    side_frames: list[dict] = []
    for p in gait_passes or []:
        if p.get("view") == "front":
            front_frames.extend(p.get("frames", []))
        elif p.get("view") == "side":
            side_frames.extend(p.get("frames", []))
    front_frames.sort(key=lambda f: f["t"])
    side_frames.sort(key=lambda f: f["t"])

    # Sagittal ROM from side pass.
    knee_rom = hip_rom = None
    if side_frames:
        knee_flex, hip_flex = [], []
        for f in side_frames:
            lm = f.get("landmarks", {})
            ka = G.angle_at(G.get(lm, "left_hip"), G.get(lm, "left_knee"),
                            G.get(lm, "left_ankle"))
            # Use the leg with the clearest data per frame (mean of both).
            kas = [G.angle_at(G.get(lm, f"{leg}_hip"), G.get(lm, f"{leg}_knee"),
                              G.get(lm, f"{leg}_ankle")) for leg in ("left", "right")]
            kas = [a for a in kas if a is not None]
            if kas:
                knee_flex.append(180.0 - statistics.mean(kas))
            has = [G.angle_at(G.get(lm, f"{leg}_shoulder"), G.get(lm, f"{leg}_hip"),
                              G.get(lm, f"{leg}_knee")) for leg in ("left", "right")]
            has = [a for a in has if a is not None]
            if has:
                hip_flex.append(180.0 - statistics.mean(has))
        if knee_flex:
            knee_rom = round(max(knee_flex) - min(knee_flex), 2)
        if hip_flex:
            hip_rom = round(max(hip_flex) - min(hip_flex), 2)
    result["knee_flexion_rom_deg"] = knee_rom
    result["hip_flexion_extension_rom_deg"] = hip_rom

    # Frontal gait metrics from front pass.
    varus_thrust = {}
    trendelenburg = {}
    stance, strides, steps = {}, {}, {}
    if front_frames:
        for leg in ("left", "right"):
            st = _stance_stats(front_frames, leg)
            stance[leg] = st["stance_s"]
            strides[leg] = st["stride_x"]
            steps[leg] = st["steps"]
            # Varus thrust: HKA deviation range during that leg's stance.
            contacts = st["contacts"]
            devs = []
            all_y = [G.get(x.get("landmarks", {}), f"{leg}_ankle")[1]
                     for x in front_frames
                     if G.get(x.get("landmarks", {}), f"{leg}_ankle")]
            ground_thresh = (min(all_y) + 0.10 * (max(all_y) - min(all_y))
                             if all_y else None)
            for c in contacts:
                off = _toe_off(front_frames, leg, c)
                end = off if off is not None else min(c + 5, len(front_frames) - 1)
                for f in front_frames[c:end + 1]:
                    lm_f = f.get("landmarks", {})
                    ank = G.get(lm_f, f"{leg}_ankle")
                    if ground_thresh is not None and ank and ank[1] > ground_thresh:
                        continue
                    d = G.signed_knee_deviation(lm_f, leg)
                    if d:
                        devs.append(d["deviation_deg"])
            varus_thrust[leg] = round(max(devs) - min(devs), 2) if len(devs) >= 2 else None
            # Trendelenburg: pelvic obliquity while the OTHER leg is lifted
            # (single-leg support on this leg) vs the standing baseline.
            other = "right" if leg == "left" else "left"
            drops = []
            for f in front_frames:
                lm = f.get("landmarks", {})
                oa = G.get(lm, f"{other}_ankle")
                lh2, rh2 = G.get(lm, "left_hip"), G.get(lm, "right_hip")
                if oa and lh2 and rh2:
                    # Other ankle clearly lifted = single support.
                    all_y = [G.get(x.get("landmarks", {}), f"{other}_ankle")[1]
                             for x in front_frames
                             if G.get(x.get("landmarks", {}), f"{other}_ankle")]
                    if all_y and oa[1] > min(all_y) + 0.3 * (max(all_y) - min(all_y)):
                        drops.append(abs(G.line_angle_deg(lh2, rh2) or 0.0))
            # drops already hold absolute pelvic angles during single
            # support; the mean IS the drop measure (not a delta from
            # the standing baseline, which would hide a pre-existing tilt).
            trendelenburg[leg] = round(statistics.mean(drops), 2) if drops else None
    result["varus_thrust_proxy_deg"] = varus_thrust
    result["trendelenburg_drop_deg"] = trendelenburg
    result["stance_time_asymmetry_pct"] = _asymmetry_pct(stance.get("left"), stance.get("right"))
    # Stride asymmetry: normalise by leg length first.
    sl = {}
    if leg_len:
        for leg in ("left", "right"):
            sl[leg] = round(strides[leg] / leg_len, 4) if strides.get(leg) else None
    result["stride_length_norm_leg_lengths"] = sl
    result["stride_length_asymmetry_pct"] = _asymmetry_pct(sl.get("left"), sl.get("right"))
    total_steps = sum(v for v in steps.values())
    duration = None
    if front_frames:
        duration = front_frames[-1]["t"] - front_frames[0]["t"]
    result["cadence_steps_per_min"] = (
        round(total_steps / duration * 60.0, 2) if duration and total_steps else None)
    result["step_count_estimate"] = total_steps
    # Step width at double support: frames where both ankles are near ground.
    step_width = None
    if front_frames and leg_len:
        widths = []
        for f in front_frames:
            lm = f.get("landmarks", {})
            la, ra = G.get(lm, "left_ankle"), G.get(lm, "right_ankle")
            if la and ra:
                widths.append(abs(la[0] - ra[0]))
        if widths:
            step_width = round(statistics.median(widths) / leg_len, 4)
    result["step_width_norm_leg_lengths"] = step_width

    # Flags: traffic light per joint per metric, UNVALIDATED everywhere.
    def add(metric_key, value, leg=None, evidence=None):
        status = _traffic(metric_key, value)
        if status is None or value is None:
            return
        spec = THRESHOLDS[metric_key]
        flag = {
            "metric": metric_key, "leg": leg, "value": value,
            "unit": spec["unit"], "status": status,
            "threshold_amber": spec["amber"], "threshold_red": spec["red"],
            "threshold_source": spec["source"],
            "threshold_status": spec["status"],
            "caveat": ("App-defined threshold; not a validated cutoff. "
                       "Flag for clinician review only, with the measured "
                       "value and its uncertainty."),
            "evidence": evidence or {},
        }
        if metric_key == "hka_deviation_deg":
            # Frontal HKA is never reported as if radiographic: a
            # +/-3 deg (or greater) uncertainty band always travels
            # with it (see README / Ge 2025 narrow-scope note).
            flag["uncertainty_band_deg"] = 3.0
        result["flags"].append(flag)

    for leg in ("left", "right"):
        hka = result["hka"].get(leg)
        if hka:
            add("hka_deviation_deg", hka["deviation_deg"], leg=leg,
                evidence={"skeleton": _evidence_summary(
                    standing, [f"{leg}_hip", f"{leg}_knee", f"{leg}_ankle"])})
        add("varus_thrust_deg", varus_thrust.get(leg), leg=leg)
        add("trendelenburg_drop_deg", trendelenburg.get(leg), leg=leg)
    add("pelvic_obliquity_deg", result["pelvic_obliquity_deg"])
    add("lateral_trunk_lean_deg", result["lateral_trunk_lean_deg"])
    add("knee_flexion_rom_deg", knee_rom)
    add("hip_rom_deg", hip_rom)
    add("stance_time_asymmetry_pct", result["stance_time_asymmetry_pct"])
    add("stride_length_asymmetry_pct", result["stride_length_asymmetry_pct"])
    add("cadence_steps_per_min", result["cadence_steps_per_min"])
    add("step_width_norm", step_width)
    reds = [f for f in result["flags"] if f["status"] == "red"]
    ambers = [f for f in result["flags"] if f["status"] == "amber"]
    result["screening_flag"] = ("red" if reds else "amber" if ambers else "green")
    result["screening_note"] = (
        "Screening pattern for clinician review only. This is not a "
        "finding about any condition, and every threshold used is "
        "app-defined and unvalidated in this product (the STEADI TUG "
        "value, where used, is a screening flag only).")
    return result
