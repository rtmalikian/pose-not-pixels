"""Eval harness + API/gateway/docs tests for Pose-Not-Pixels.

Synthetic fixtures carry KNOWN ground truth, so every headline
number below is a REPORTED measured error, not a vibe.
Pre-specified bars to be demonstrated on real captures (NOT claimed
as results): angle error < 5 deg (clinical-acceptability bar);
asymmetry error < 3 points; TUG error < 0.5 s as a target to be
demonstrated for pose-video TUG (the 0.42 s figure in the literature
was inertial-sensor, not pose video). Published single-phone
MediaPipe vs Qualisys mean error is 7.16 deg (n=6, healthy,
non-gait); multi-camera OpenCap is about 4.5 deg. Synthetic errors
do NOT demonstrate clinical accuracy.
Run:  python -m pytest -q
"""

import copy
import json
import math
import os
import random
import re

import pytest
from fastapi.testclient import TestClient

from app import gateway, safety
from app.pose import geometry as G, kinematics as K, synth
from app.pose.service import CAPTURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _clean():
    gateway.reset_ledger()
    CAPTURES.clear()
    yield
    gateway.reset_ledger()


@pytest.fixture()
def client():
    from app.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _pt(x, y):
    return (x, y, None)


def test_angle_straight_is_180():
    assert G.angle_at(_pt(0, 1), _pt(0, 0), _pt(0, -1)) == pytest.approx(180.0)


def test_angle_known_120():
    # Vectors (0,1) and (sin120, cos120) meet at 120 deg.
    a = math.radians(120)
    assert G.angle_at(_pt(0, 1), _pt(0, 0), _pt(math.sin(a), math.cos(a))) == pytest.approx(120.0, abs=0.01)


def test_hka_known_deviations_reported():
    for dev in (0.0, 3.0, 5.0, 8.0, 12.0):
        lm = synth.standing_frame(hka_left_deg=dev, hka_right_deg=dev)
        got = G.signed_knee_deviation(lm, "left")["deviation_deg"]
        assert abs(got - dev) < 0.01, (dev, got)


def test_leg_length():
    lm = synth.standing_frame()
    assert G.leg_length(lm, "left") == pytest.approx(0.90, abs=0.01)


def test_pelvic_obliquity_known():
    for ob in (0.0, 4.0, 6.0):
        frames = synth.standing_sequence(pelvic_obliquity_deg=ob)
        a = K.analyse_arthritis(frames, [])
        assert abs(a["pelvic_obliquity_deg"] - ob) < 0.5, (ob, a["pelvic_obliquity_deg"])


def test_trunk_lean_known():
    for lean in (0.0, 6.0):
        frames = synth.standing_sequence(trunk_lean_deg=lean)
        a = K.analyse_arthritis(frames, [])
        assert abs(a["lateral_trunk_lean_deg"] - lean) < 0.5, (lean, a["lateral_trunk_lean_deg"])


# ---------------------------------------------------------------------------
# Pack 1 — TUG
# ---------------------------------------------------------------------------

def test_tug_markers_exact():
    frames, markers = synth.tug_sequence()
    m = K.analyse_tug(frames, markers)
    assert m["tug_total_s"] == pytest.approx(10.4, abs=0.001)
    assert m["sit_to_stand_s"] == pytest.approx(1.2, abs=0.001)
    assert m["walk_s"] == pytest.approx(7.2, abs=0.001)
    assert m["turn_s"] == pytest.approx(2.0, abs=0.001)
    assert m["phase_method"] == "markers"


def test_tug_inferred_mae_under_half_second():
    frames, markers = synth.tug_sequence()
    m = K.analyse_tug(frames)  # no markers: heuristic
    assert m["phase_method"] == "inferred"
    assert abs(m["tug_total_s"] - 10.4) < 0.5
    assert abs(m["sit_to_stand_s"] - 1.2) < 0.5
    assert abs(m["walk_s"] - 7.2) < 0.5
    assert abs(m["turn_s"] - 2.0) < 0.5


def test_tug_slow_steadi_screening_flag():
    frames, markers = synth.tug_sequence(stand_t=2.0, turn_start_t=7.0,
                                         turn_end_t=10.0, sit_t=15.2)
    m = K.analyse_tug(frames, markers)
    assert m["tug_total_s"] == pytest.approx(15.2, abs=0.01)
    assert m["tug_flag"] == "steadi_screening_flag"
    assert m["tug_flag_label"] == "STEADI screening flag \u2014 discuss with clinician"
    assert m["tug_threshold"]["red"] == 12.0


def test_tug_standard_below_steadi_flag():
    frames, markers = synth.tug_sequence()
    assert K.analyse_tug(frames, markers)["tug_flag"] == "below_steadi_flag"


def test_tug_gait_proxies_positive():
    frames, markers = synth.tug_sequence()
    m = K.analyse_tug(frames, markers)
    assert m["gait_speed_proxy_leg_lengths_per_s"] and m["gait_speed_proxy_leg_lengths_per_s"] > 0
    assert m["step_count_estimate"] >= 10
    assert m["cadence_steps_per_min"] and 60 < m["cadence_steps_per_min"] < 200


def test_tug_equal_stance_asymmetry_near_zero():
    frames, markers = synth.tug_sequence()
    m = K.analyse_tug(frames, markers)
    assert m["stance_time_asymmetry_pct"] is not None
    assert m["stance_time_asymmetry_pct"] < 3.0


# ---------------------------------------------------------------------------
# Pack 2 — Arthritis kinematics
# ---------------------------------------------------------------------------

def _full_arthritis(**front_kw):
    standing = synth.standing_sequence(hka_left_deg=8.0, hka_right_deg=1.0)
    front = synth.front_gait_sequence(**front_kw)
    side = synth.side_combined_sequence(5.0, 60.0, 5.0, 25.0)
    return K.analyse_arthritis(standing, [{"view": "front", "frames": front},
                                          {"view": "side", "frames": side}])


def test_hka_per_leg_within_5deg_bar():
    # Pre-specified clinical-acceptability bar (<5 deg) applied to
    # SYNTHETIC fixtures only; it is not a demonstrated clinical result.
    a = _full_arthritis()
    assert abs(a["hka"]["left"]["deviation_deg"] - 8.0) < 5.0
    assert abs(a["hka"]["right"]["deviation_deg"] - 1.0) < 5.0
    # Frontal HKA always carries a +/-3 deg (or greater) band.
    assert a["hka"]["left"]["uncertainty_band_deg"] >= 3.0


def test_knee_rom_within_5deg_bar():
    side = synth.side_knee_sequence(5.0, 60.0)
    a = K.analyse_arthritis(synth.standing_sequence(), [{"view": "side", "frames": side}])
    assert abs(a["knee_flexion_rom_deg"] - 55.0) < 5.0


def test_hip_rom_within_5deg_bar():
    side = synth.side_hip_sequence(5.0, 25.0)
    a = K.analyse_arthritis(synth.standing_sequence(), [{"view": "side", "frames": side}])
    assert abs(a["hip_flexion_extension_rom_deg"] - 20.0) < 5.0


def test_combined_rom_within_5deg_bar():
    a = _full_arthritis()
    assert abs(a["knee_flexion_rom_deg"] - 55.0) < 5.0
    assert abs(a["hip_flexion_extension_rom_deg"] - 20.0) < 5.0


def test_stance_asymmetry_error_under_3pts():
    a = _full_arthritis(stance_left_s=0.70, stance_right_s=0.50)
    expected = abs(0.70 - 0.50) / 0.60 * 100  # 33.33
    assert abs(a["stance_time_asymmetry_pct"] - expected) < 3.0


def test_varus_thrust_proxy():
    a = _full_arthritis(varus_thrust_range=(2.0, 9.0))
    assert abs(a["varus_thrust_proxy_deg"]["left"] - 7.0) < 5.0


def test_trendelenburg_proxy():
    a = _full_arthritis(trendelenburg_deg=6.0)
    assert abs(a["trendelenburg_drop_deg"]["left"] - 6.0) < 5.0
    assert abs(a["trendelenburg_drop_deg"]["right"] - 6.0) < 5.0


def test_step_width_norm():
    a = _full_arthritis()
    assert a["step_width_norm_leg_lengths"] is not None
    assert abs(a["step_width_norm_leg_lengths"] - 0.30 / 0.95) < 0.08


def test_flags_unvalidated_everywhere():
    a = _full_arthritis(stance_left_s=0.70, stance_right_s=0.50,
                        trendelenburg_deg=6.0, varus_thrust_range=(2.0, 9.0))
    assert a["flags"], "expected at least one flag"
    for f in a["flags"]:
        assert f["threshold_status"] == "APP_DEFINED_UNVALIDATED"
        assert f["threshold_source"], f["metric"]
        assert "caveat" in f


def test_flags_traffic_light():
    a = _full_arthritis()
    by_key = {(f["metric"], f["leg"]): f["status"] for f in a["flags"]}
    assert by_key[("hka_deviation_deg", "left")] == "red"   # 8 deg > 5
    assert by_key[("hka_deviation_deg", "right")] == "green"  # 1 deg
    assert a["screening_flag"] == "red"


def test_straight_capture_green():
    standing = synth.standing_sequence()
    front = synth.front_gait_sequence()
    side = synth.side_combined_sequence()
    a = K.analyse_arthritis(standing, [{"view": "front", "frames": front},
                                       {"view": "side", "frames": side}])
    # Straight standing + symmetric gait: no red flags on HKA.
    by_key = {(f["metric"], f["leg"]): f["status"] for f in a["flags"]}
    assert by_key[("hka_deviation_deg", "left")] == "green"
    assert by_key[("hka_deviation_deg", "right")] == "green"


def test_missing_side_pass_gives_none_not_crash():
    standing = synth.standing_sequence(hka_left_deg=4.0)
    front = synth.front_gait_sequence()
    a = K.analyse_arthritis(standing, [{"view": "front", "frames": front}])
    assert a["knee_flexion_rom_deg"] is None
    assert a["hip_flexion_extension_rom_deg"] is None
    assert a["hka"]["left"]["deviation_deg"] == pytest.approx(4.0, abs=0.5)


def test_missing_landmarks_degrade_gracefully():
    frames = synth.standing_sequence(hka_left_deg=6.0)
    for f in frames:
        f["landmarks"]["left_knee"]["visibility"] = 0.0
    a = K.analyse_arthritis(frames, [])
    assert a["hka"]["left"] is None
    assert a["hka"]["right"] is not None


# ---------------------------------------------------------------------------
# Robustness: jitter, dropped frames, occlusion
# ---------------------------------------------------------------------------

def _jitter(frames, seed=7, amp=0.01):
    rng = random.Random(seed)
    out = copy.deepcopy(frames)
    for f in out:
        for lm in f["landmarks"].values():
            lm["x"] += rng.uniform(-amp, amp)
            lm["y"] += rng.uniform(-amp, amp)
    return out


def test_jitter_hka_within_5deg_bar():
    frames = synth.standing_sequence(hka_left_deg=8.0, hka_right_deg=2.0)
    a = K.analyse_arthritis(_jitter(frames), [])
    assert abs(a["hka"]["left"]["deviation_deg"] - 8.0) < 5.0
    assert abs(a["hka"]["right"]["deviation_deg"] - 2.0) < 5.0


def test_dropped_frames_tug_still_exact_with_markers():
    frames, markers = synth.tug_sequence()
    kept = [f for i, f in enumerate(frames) if i % 3 != 2]
    m = K.analyse_tug(kept, markers)
    assert m["tug_total_s"] == pytest.approx(10.4, abs=0.001)


def test_dropped_frames_gait_no_crash():
    front = synth.front_gait_sequence(stance_left_s=0.70, stance_right_s=0.50)
    kept = [f for i, f in enumerate(front) if i % 4 != 3]
    a = K.analyse_arthritis(synth.standing_sequence(),
                            [{"view": "front", "frames": kept}])
    assert a["stance_time_asymmetry_pct"] is not None
    assert abs(a["stance_time_asymmetry_pct"] - 33.33) < 8.0


def test_occluded_frames_no_crash():
    frames, markers = synth.tug_sequence()
    for i, f in enumerate(frames):
        if i % 5 == 0:
            f["landmarks"]["left_ankle"]["visibility"] = 0.0
            f["landmarks"]["right_ankle"]["visibility"] = 0.0
    m = K.analyse_tug(frames, markers)
    assert m["tug_total_s"] == pytest.approx(10.4, abs=0.001)


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

def test_models_lists_both_tiers():
    data = gateway.list_models()
    ids = [m["id"] for m in data["data"]]
    assert "pose-not-pixels-frugal" in ids
    assert "pose-not-pixels-full" in ids


def test_gateway_tier_resolution():
    assert gateway.resolve_tier("pose-not-pixels-full") == "full"
    assert gateway.resolve_tier("pose-not-pixels-frugal") == "frugal"
    assert gateway.resolve_tier(None, "full") == "full"
    assert gateway.resolve_tier(None, None) == "frugal"


def test_gateway_chat_metrics_only():
    frames, markers = synth.tug_sequence()
    m = K.analyse_tug(frames, markers)
    resp = gateway.dispatch_chat(
        [{"role": "user", "content": "[TASK: SUMMARISE_TUG]\n[METRICS]\n"
          + json.dumps(m) + "\n[/METRICS]"}],
        model="pose-not-pixels-frugal", tier="frugal")
    text = resp["choices"][0]["message"]["content"]
    assert "10.40" in text
    assert safety.find_banned_language(text) == []


def test_gateway_rejects_image_part():
    with pytest.raises(Exception):
        gateway.dispatch_chat(
            [{"role": "user", "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}])


def test_gateway_rejects_video_payload():
    with pytest.raises(Exception):
        gateway.dispatch_chat(
            [{"role": "user", "content": 'metrics {"video": "base64..."}'}])


def test_gateway_embeddings_deterministic():
    a = gateway.dispatch_embeddings("{\"tug_total_s\": 10.4}")
    b = gateway.dispatch_embeddings("{\"tug_total_s\": 10.4}")
    assert a["data"][0]["embedding"] == b["data"][0]["embedding"]
    assert len(a["data"][0]["embedding"]) == 32


def test_gateway_ledger_has_zero_video_payloads():
    frames, markers = synth.tug_sequence()
    m = K.analyse_tug(frames, markers)
    for _ in range(3):
        gateway.dispatch_chat(
            [{"role": "user", "content": "[TASK: SUMMARISE_TUG]\n[METRICS]\n"
              + json.dumps(m) + "\n[/METRICS]"}],
            model="pose-not-pixels-frugal", tier="frugal")
        gateway.dispatch_embeddings(json.dumps(m))
    # Scan LEDGER ENTRIES (not substrings of the product name): no
    # entry may carry an image/video/frame key or value.
    forbidden = {"image_url", "video_url", "video", "image", "frames",
                 "frame_data", "pixels", "base64"}
    for entry in gateway.CALL_LOG:
        assert forbidden.isdisjoint(entry.keys()), entry
        for v in entry.values():
            assert not (isinstance(v, str) and any(k in v for k in ("image_url", "video_url"))), entry


def test_full_tier_verifier_ok():
    from app.pose.service import _summarise
    frames, markers = synth.tug_sequence()
    m = K.analyse_tug(frames, markers)
    out = _summarise("SUMMARISE_TUG", m, "full", kind="tug")
    assert out["verifier"] is not None
    assert out["verifier"]["ok"] is True, out["verifier"]


def test_frugal_summary_token_cap():
    from app.pose.service import _summarise
    frames, markers = synth.tug_sequence()
    m = K.analyse_tug(frames, markers)
    out = _summarise("SUMMARISE_TUG", m, "frugal", kind="tug")
    assert len(out["summary"]) <= 880  # ~220 tokens at 4 chars/token


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def test_phi_redaction():
    redacted, report = safety.redact_phi("Patient: John Smith, john@example.com, 555-123-4567")
    assert "john@example.com" not in redacted
    assert "555-123-4567" not in redacted
    assert report, "expected at least one PHI kind redacted"


def test_banned_language_detection():
    assert "diagnosed" in str(safety.find_banned_language("you were diagnosed"))
    assert safety.find_banned_language("pattern for clinician review") == []


def test_service_summaries_have_no_banned_language():
    from app.pose import service
    frames, markers = synth.tug_sequence()
    cap = service.analyse_tug_capture("p1", frames, markers=markers, tier="frugal")
    assert cap["summary"]["banned_language_found"] == []
    standing = synth.standing_sequence(hka_left_deg=8.0)
    front = synth.front_gait_sequence()
    side = synth.side_combined_sequence()
    cap2 = service.analyse_arthritis_capture(
        "p1", standing, [{"view": "front", "frames": front},
                         {"view": "side", "frames": side}], tier="frugal")
    assert cap2["summary"]["banned_language_found"] == []
    assert "diagnosed" not in cap2["summary"]["summary"].lower()
    assert "you have osteoarthritis" not in cap2["summary"]["summary"].lower()
    assert "you have arthritis" not in cap2["summary"]["summary"].lower()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _tug_payload():
    frames, markers = synth.tug_sequence()
    return {"patient_id": "demo-patient", "frames": frames, "markers": markers}


def test_api_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_tug(client):
    r = client.post("/api/tug", json=_tug_payload())
    assert r.status_code == 200, r.text
    assert r.json()["metrics"]["tug_total_s"] == pytest.approx(10.4, abs=0.01)


def test_api_tug_rejects_video_payload(client):
    payload = _tug_payload()
    payload["video"] = "base64-video-bytes"
    r = client.post("/api/tug", json=payload)
    assert r.status_code in (400, 422)


def test_api_tug_rejects_frame_payload(client):
    payload = _tug_payload()
    payload["frames_base64"] = ["AAA", "BBB"]
    r = client.post("/api/tug", json=payload)
    assert r.status_code in (400, 422)


def test_api_arthritis(client):
    standing = synth.standing_sequence(hka_left_deg=8.0)
    front = synth.front_gait_sequence()
    side = synth.side_combined_sequence()
    r = client.post("/api/arthritis", json={
        "patient_id": "demo-patient", "standing_frames": standing,
        "gait_passes": [{"view": "front", "frames": front},
                        {"view": "side", "frames": side}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metrics"]["hka"]["left"]["deviation_deg"] == pytest.approx(8.0, abs=0.5)
    assert all(f["threshold_status"] == "APP_DEFINED_UNVALIDATED" for f in body["metrics"]["flags"])


def test_api_trend_delta(client):
    client.post("/api/tug", json=_tug_payload())
    payload2 = _tug_payload()
    payload2["markers"] = {"start": 0.0, "stand": 2.0, "turn_start": 7.0,
                           "turn_end": 10.0, "sit": 15.2}
    client.post("/api/tug", json=payload2)
    r = client.get("/api/patients/demo-patient/trend")
    assert r.status_code == 200
    caps = r.json()["captures"]
    assert len(caps) == 2
    assert caps[0]["delta_vs_previous"] is None
    assert caps[1]["delta_vs_previous"]["tug_total_s"] == pytest.approx(4.8, abs=0.01)


def test_api_thresholds(client):
    r = client.get("/api/thresholds")
    assert r.status_code == 200
    th = r.json()
    assert th["tug_total_s"]["red"] == 12.0
    assert th["tug_total_s"]["status"] == "STEADI_SCREENING_FLAG_ONLY"
    assert "screening flag" in th["tug_total_s"]["source"]
    assert all(v.get("source") and v.get("status") in
               ("APP_DEFINED_UNVALIDATED", "STEADI_SCREENING_FLAG_ONLY")
               for v in th.values())
    assert all(v.get("status") == "APP_DEFINED_UNVALIDATED"
               for k, v in th.items() if k != "tug_total_s")


def test_api_demo_endpoints(client):
    r = client.post("/api/demo/tug?scenario=tug_standard")
    assert r.status_code == 200
    assert r.json()["metrics"]["tug_total_s"] == pytest.approx(10.4, abs=0.01)
    r = client.post("/api/demo/arthritis?scenario=arthritis_varus_left")
    assert r.status_code == 200
    assert r.json()["metrics"]["screening_flag"] == "red"
    r = client.post("/api/demo/tug", headers={"X-Tier": "full"})
    assert r.status_code == 200
    assert r.json()["summary"]["verifier"]["ok"] is True


def test_api_gateway_endpoints(client):
    assert client.get("/v1/models").status_code == 200
    r = client.post("/v1/chat/completions", json={
        "model": "pose-not-pixels-frugal",
        "messages": [{"role": "user", "content": "[TASK: SUMMARISE_TUG]\n[METRICS]\n{}\n[/METRICS]"}]})
    assert r.status_code == 200
    r = client.post("/v1/chat/completions", json={
        "model": "pose-not-pixels-frugal",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:x"}}]}]})
    assert r.status_code == 400
    r = client.post("/v1/embeddings", json={"input": "{\"tug_total_s\": 10.4}"})
    assert r.status_code == 200
    assert len(r.json()["data"][0]["embedding"]) == 32


# ---------------------------------------------------------------------------
# Docs / UI honesty gates
# ---------------------------------------------------------------------------

def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def test_readme_disclaimer_is_line_1():
    readme = _read("README.md")
    line1 = readme.splitlines()[0]
    assert "PROOF OF CONCEPT" in line1 and "NOT CLINICALLY TESTED" in line1


def test_no_sensitivity_specificity_percent_claims():
    pat = re.compile(r"(sensitivity|specificity)[^.\n]{0,80}\d+\s*%", re.IGNORECASE)
    for rel in ("README.md", "SECURITY.md", "static/index.html"):
        text = _read(rel)
        assert pat.search(text) is None, f"{rel}: {pat.search(text)}"


def test_ui_disclaimer_above_results():
    html = _read("static/index.html")
    assert "PROOF OF CONCEPT" in html
    assert html.index("PROOF OF CONCEPT") < html.index("Results")
    assert "skeleton" in html.lower()


def test_thresholds_constant_has_sources():
    for key, spec in K.THRESHOLDS.items():
        assert spec["source"], key
        assert spec["status"] in ("APP_DEFINED_UNVALIDATED",
                                  "STEADI_SCREENING_FLAG_ONLY"), key
        assert "literature-based" not in spec["source"].lower(), key
