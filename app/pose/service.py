"""Product service: captures, summaries, trend — all via the gateway.

Deterministic maths (app.pose.kinematics) computes every number.
The gateway LLM only phrases metrics JSON into a short summary.
The full tier adds longitudinal context to the prompt and a
second-pass verifier check that the summary's numbers match the
metrics JSON. Vision / landmark extraction is identical in both
tiers by design.
"""

from __future__ import annotations

import json
import time
import uuid

from app import gateway, safety
from app.pose import kinematics

CAPTURES: dict[str, list[dict]] = {}


def _verifier_expected(metrics: dict, kind: str) -> dict:
    """The numbers a summary is expected to carry. The verifier checks
    only these — thresholds, boundaries and ids are not summary content
    and must not fail a check for being absent."""
    if kind == "tug":
        return {k: metrics[k] for k in (
            "tug_total_s", "sit_to_stand_s", "walk_s", "turn_s",
            "gait_speed_proxy_leg_lengths_per_s", "stance_time_asymmetry_pct",
            "cadence_steps_per_min") if metrics.get(k) is not None}
    out = {}
    for k in ("pelvic_obliquity_deg", "lateral_trunk_lean_deg",
              "stance_time_asymmetry_pct", "cadence_steps_per_min"):
        if metrics.get(k) is not None:
            out[k] = metrics[k]
    for leg in ("left", "right"):
        hka = (metrics.get("hka") or {}).get(leg) or {}
        if hka.get("deviation_deg") is not None:
            out[f"hka_{leg}_deviation_deg"] = hka["deviation_deg"]
    return out


def _summarise(task: str, metrics: dict, tier: str,
               previous: list[dict] | None = None, kind: str = "tug") -> dict:
    """Phrase metrics via the gateway; scan the result for banned
    language before returning it. Frugal summaries are capped at the
    tier's token budget (~220 tokens ~= 880 chars)."""
    profile = gateway.tier_profile(tier)
    prompt_parts = [f"[TASK: {task}]", "[METRICS]",
                    json.dumps(metrics, sort_keys=True), "[/METRICS]"]
    if tier == "full" and previous:
        prompt_parts += ["[PREVIOUS_CAPTURES]",
                         json.dumps(previous[-3:], sort_keys=True)[:2000],
                         "[/PREVIOUS_CAPTURES]"]
    resp = gateway.dispatch_chat(
        [{"role": "user", "content": "\n".join(prompt_parts)}],
        model=profile["model_alias"], tier=tier,
        max_tokens=profile["summary_max_tokens"])
    summary = resp["choices"][0]["message"]["content"]
    if tier == "frugal":
        # ~220 tokens at ~4 chars/token.
        summary = summary[:880]
    banned = safety.find_banned_language(summary)
    verifier = None
    if profile["verifier"]:
        vresp = gateway.dispatch_chat(
            [{"role": "user", "content": "\n".join([
                "[TASK: VERIFY_REPORT]", "[EXPECTED]",
                json.dumps(_verifier_expected(metrics, kind), sort_keys=True),
                "[/EXPECTED]",
                "[SUMMARY]", summary, "[/SUMMARY]"])}],
            model=profile["model_alias"], tier=tier)
        try:
            verifier = json.loads(vresp["choices"][0]["message"]["content"])
        except Exception:
            verifier = {"ok": False, "problems": ["verifier output unreadable"]}
    return {
        "summary": summary,
        "banned_language_found": banned,
        "verifier": verifier,
        "gateway_usage": resp["usage"],
        "tier": tier,
    }


def analyse_tug_capture(patient_id: str, frames: list[dict],
                        markers: dict | None = None, tier: str = "frugal") -> dict:
    metrics = kinematics.analyse_tug(frames, markers=markers)
    previous = CAPTURES.get(patient_id, [])
    summary = _summarise("SUMMARISE_TUG", metrics, tier, previous, kind="tug")
    emb = gateway.dispatch_embeddings(json.dumps(metrics, sort_keys=True),
                                      model=gateway.tier_profile(tier)["model_alias"])
    capture = {
        "id": uuid.uuid4().hex[:12], "patient_id": patient_id,
        "kind": "tug", "created_at": int(time.time()),
        "metrics": metrics, "summary": summary,
        "metrics_embedding": emb["data"][0]["embedding"][:8],
        "phase_method": metrics["phase_method"],
    }
    CAPTURES.setdefault(patient_id, []).append(capture)
    return capture


def analyse_arthritis_capture(patient_id: str, standing_frames: list[dict],
                              gait_passes: list[dict], tier: str = "frugal") -> dict:
    metrics = kinematics.analyse_arthritis(standing_frames, gait_passes)
    previous = CAPTURES.get(patient_id, [])
    summary = _summarise("SUMMARISE_ARTHRITIS", metrics, tier, previous, kind="arthritis")
    emb = gateway.dispatch_embeddings(json.dumps(metrics, sort_keys=True),
                                      model=gateway.tier_profile(tier)["model_alias"])
    capture = {
        "id": uuid.uuid4().hex[:12], "patient_id": patient_id,
        "kind": "arthritis", "created_at": int(time.time()),
        "metrics": metrics, "summary": summary,
        "metrics_embedding": emb["data"][0]["embedding"][:8],
    }
    CAPTURES.setdefault(patient_id, []).append(capture)
    return capture


def trend_for(patient_id: str) -> dict:
    """Captures for one patient, each with delta vs the previous capture
    of the same kind (None for the first)."""
    caps = CAPTURES.get(patient_id, [])
    out = []
    prev_by_kind: dict[str, dict] = {}
    for c in caps:
        prev = prev_by_kind.get(c["kind"])
        delta = None
        if prev:
            delta = {}
            pm, m = prev["metrics"], c["metrics"]
            for key in ("tug_total_s", "gait_speed_proxy_leg_lengths_per_s",
                        "stance_time_asymmetry_pct", "cadence_steps_per_min",
                        "pelvic_obliquity_deg", "lateral_trunk_lean_deg",
                        "knee_flexion_rom_deg", "hip_flexion_extension_rom_deg"):
                if isinstance(m.get(key), (int, float)) and isinstance(pm.get(key), (int, float)):
                    delta[key] = round(m[key] - pm[key], 3)
            for leg in ("left", "right"):
                for src, dst in (("hka", "hka_deviation_deg"),):
                    a = m.get(src, {}).get(leg, {}) if isinstance(m.get(src), dict) else None
                    b = pm.get(src, {}).get(leg, {}) if isinstance(pm.get(src), dict) else None
                    if a and b and a.get("deviation_deg") is not None and b.get("deviation_deg") is not None:
                        delta[f"hka_{leg}_deviation_deg"] = round(
                            a["deviation_deg"] - b["deviation_deg"], 3)
        out.append({**c, "delta_vs_previous": delta})
        prev_by_kind[c["kind"]] = c
    return {"patient_id": patient_id, "captures": out}
