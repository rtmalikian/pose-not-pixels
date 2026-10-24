"""Pose-Not-Pixels — FastAPI application.

Routes:
  /v1/*    Embedded OpenAI-compatible gateway (metrics JSON only).
           Tier by model alias (pose-not-pixels-frugal | full) or
           X-Tier header. Vision / landmark extraction is identical
           in both tiers by design.
  /api/*   Product API: TUG captures, arthritis screening captures,
           per-patient trend, synthetic demos.
  /        Static single-page demo UI.
  /healthz Health check.

Storage is in-memory in v0.1.0. Raw video is NEVER an input type:
request models forbid extra fields, so a video/frame payload shape
is rejected (422) before any processing, and the gateway rejects
image/video content parts (400). Landmarks are the only input.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import gateway, safety
from app.pose import kinematics, models, service, synth

logger = logging.getLogger("pose-not-pixels")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(
    title="Pose-Not-Pixels",
    description="Phone gait / Timed Up and Go trend and knee & hip "
                "arthritis kinematics screening from skeleton landmarks "
                "only. Video never leaves the device and never reaches "
                "a model: deterministic maths computes the numbers, the "
                "gateway LLM only phrases metrics JSON.",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict] = Field(default_factory=list)
    temperature: float = 0.2
    max_tokens: int = 2000
    response_format: dict | None = None


class EmbeddingsRequest(BaseModel):
    model: str | None = None
    input: str | list[str] = ""


@app.get("/v1/models")
def v1_models():
    return gateway.list_models()


@app.post("/v1/chat/completions")
def v1_chat_completions(req: ChatRequest,
                        x_tier: str | None = Header(default=None)):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    tier = gateway.resolve_tier(req.model, x_tier)
    return gateway.dispatch_chat(
        req.messages, model=req.model, temperature=req.temperature,
        max_tokens=req.max_tokens, response_format=req.response_format,
        tier=tier)


@app.post("/v1/embeddings")
def v1_embeddings(req: EmbeddingsRequest):
    return gateway.dispatch_embeddings(req.input, model=req.model)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "backend": gateway.get_backend(),
            "tier_default": gateway.resolve_tier(),
            "patients": len(service.CAPTURES)}


def _frames(req_frames):
    return [{"t": f.t, "landmarks": {k: v.model_dump() for k, v in f.landmarks.items()}}
            for f in req_frames]


@app.post("/api/tug")
def api_tug(req: models.TugRequest, x_tier: str | None = Header(default=None)):
    patient_id, _ = safety.redact_phi(req.patient_id)
    tier = gateway.resolve_tier(None, x_tier)
    markers = req.markers.model_dump() if req.markers else None
    try:
        return service.analyse_tug_capture(patient_id, _frames(req.frames),
                                           markers=markers, tier=tier)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/arthritis")
def api_arthritis(req: models.ArthritisRequest,
                  x_tier: str | None = Header(default=None)):
    patient_id, _ = safety.redact_phi(req.patient_id)
    tier = gateway.resolve_tier(None, x_tier)
    passes = [{"view": p.view, "frames": _frames(p.frames)} for p in req.gait_passes]
    return service.analyse_arthritis_capture(
        patient_id, _frames(req.standing_frames), passes, tier=tier)


@app.get("/api/patients/{patient_id}/trend")
def api_trend(patient_id: str):
    return service.trend_for(patient_id)


@app.get("/api/thresholds")
def api_thresholds():
    return kinematics.THRESHOLDS


# ---------------------------------------------------------------------------
# Synthetic demos — the same fixtures CI measures
# ---------------------------------------------------------------------------

@app.get("/api/demo/scenarios")
def api_demo_scenarios():
    return {
        "tug": ["tug_standard", "tug_slow"],
        "arthritis": ["arthritis_straight", "arthritis_varus_left",
                      "arthritis_asymmetric"],
    }


@app.post("/api/demo/tug")
def api_demo_tug(scenario: str = "tug_standard",
                 x_tier: str | None = Header(default=None)):
    tier = gateway.resolve_tier(None, x_tier)
    if scenario == "tug_slow":
        frames, markers = synth.tug_sequence(stand_t=2.0, turn_start_t=7.0,
                                             turn_end_t=10.0, sit_t=15.2)
    elif scenario == "tug_standard":
        frames, markers = synth.tug_sequence()
    else:
        raise HTTPException(status_code=400, detail="unknown tug scenario")
    cap = service.analyse_tug_capture("demo", frames, markers=markers, tier=tier)
    return {"scenario": scenario, "truth_markers": markers, "tier": tier, **cap}


@app.post("/api/demo/arthritis")
def api_demo_arthritis(scenario: str = "arthritis_straight",
                       x_tier: str | None = Header(default=None)):
    tier = gateway.resolve_tier(None, x_tier)
    if scenario == "arthritis_straight":
        standing = synth.standing_sequence()
        front = synth.front_gait_sequence()
    elif scenario == "arthritis_varus_left":
        standing = synth.standing_sequence(hka_left_deg=8.0, hka_right_deg=1.0)
        front = synth.front_gait_sequence(varus_thrust_range=(2.0, 9.0))
    elif scenario == "arthritis_asymmetric":
        standing = synth.standing_sequence(hka_left_deg=4.0, pelvic_obliquity_deg=4.0,
                                           trunk_lean_deg=6.0)
        front = synth.front_gait_sequence(stance_left_s=0.70, stance_right_s=0.50,
                                          trendelenburg_deg=6.0)
    else:
        raise HTTPException(status_code=400, detail="unknown arthritis scenario")
    side = synth.side_combined_sequence(5.0, 60.0, 5.0, 25.0)
    cap = service.analyse_arthritis_capture(
        "demo", standing,
        [{"view": "front", "frames": front}, {"view": "side", "frames": side}],
        tier=tier)
    return {"scenario": scenario, "tier": tier, **cap}


_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
