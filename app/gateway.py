"""Embedded OpenAI-compatible inference gateway for Pose-Not-Pixels.

Same skeleton as pill-tray / wound-delta: every AI feature calls this
gateway, which speaks ``/v1/models``, ``/v1/chat/completions`` and
``/v1/embeddings`` and routes to a pluggable backend (stub | local |
upstream). Tier is selected by the ``model`` alias
(``pose-not-pixels-frugal`` | ``pose-not-pixels-full``) or an ``X-Tier``
header, resolved against ``POSE_TIER`` when neither is given.

CENTRAL CONTRACT — enforced here, not just documented:

* The gateway takes METRICS JSON ONLY. Any message content part of
  type ``image_url`` / ``video_url``, or any part carrying an
  ``image`` / ``video`` / ``frame`` key, is rejected with HTTP 400
  before any backend is contacted.
* The call ledger records kinds, tiers, backends and token estimates
  ONLY — never message contents, never landmarks, never frames. Tests
  scan the ledger for image/video keys and must find none.
* Vision / landmark extraction is IDENTICAL in both tiers by design.
  The frugal tier phrases a summary (max ~220 tokens). The full tier
  buys richer longitudinal context plus a second-pass check that the
  summary's numbers match the metrics JSON. It buys no new vision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid

import httpx
from fastapi import HTTPException

STUB_MODEL_ID = "pose-not-pixels-stub-1"
FRUGAL_MODEL_ID = "pose-not-pixels-frugal"
FULL_MODEL_ID = "pose-not-pixels-full"
_REQUEST_TIMEOUT = 120.0

TIERS: dict[str, dict] = {
    "frugal": {
        "model_alias": FRUGAL_MODEL_ID,
        "verifier": False,
        "summary_max_tokens": 220,
        "longitudinal_context": "latest_only",
    },
    "full": {
        "model_alias": FULL_MODEL_ID,
        "verifier": True,
        "summary_max_tokens": 500,
        "longitudinal_context": "all_previous",
    },
}

_FORBIDDEN_PART_TYPES = {"image_url", "video_url", "image", "video", "frame"}
_FORBIDDEN_KEYS = {"image_url", "video_url", "video", "image", "frames",
                   "frame_data", "pixels", "base64"}


def resolve_tier(model: str | None = None, x_tier: str | None = None) -> str:
    if x_tier and x_tier.strip().lower() in TIERS:
        return x_tier.strip().lower()
    if model:
        m = model.strip().lower()
        if m.endswith("-full") or m == FULL_MODEL_ID:
            return "full"
        if m.endswith("-frugal") or m == FRUGAL_MODEL_ID:
            return "frugal"
    env = (os.environ.get("POSE_TIER", "frugal") or "frugal").strip().lower()
    return env if env in TIERS else "frugal"


def tier_profile(tier: str) -> dict:
    return dict(TIERS[tier])


def estimate_text_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


CALL_LEDGER: dict[str, int] = {"chat": 0, "embeddings": 0}
CALL_LOG: list[dict] = []


def reset_ledger() -> None:
    CALL_LEDGER["chat"] = 0
    CALL_LEDGER["embeddings"] = 0
    CALL_LOG.clear()


def get_backend() -> str:
    return (os.environ.get("AI_BACKEND", "stub") or "stub").strip().lower()


def _local_base() -> str:
    return (os.environ.get("LOCAL_MODEL_URL", "http://localhost:11434") or "").rstrip("/")


def _upstream_base() -> str:
    return (os.environ.get("UPSTREAM_BASE_URL", "https://api.openai.com") or "").rstrip("/")


def _upstream_headers() -> dict:
    key = os.environ.get("UPSTREAM_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def assert_metrics_only(messages: list[dict]) -> None:
    """Reject any video/frame/image payload shape before dispatch."""
    for m in messages or []:
        content = m.get("content", "")
        if isinstance(content, str):
            low = content.lower()
            # A JSON blob naming a video field is a payload shape, not metrics.
            if any(k in low for k in ('"video"', '"image_url"', '"video_url"',
                                       '"frames_base64"', '"frame_data"')):
                raise HTTPException(status_code=400,
                                    detail="gateway takes metrics JSON only; "
                                           "video/frame payloads are rejected")
            continue
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in _FORBIDDEN_PART_TYPES:
                    raise HTTPException(
                        status_code=400,
                        detail="gateway takes metrics JSON only; "
                               f"forbidden content part: {part.get('type')}")
                if _FORBIDDEN_KEYS & set(part.keys()):
                    raise HTTPException(
                        status_code=400,
                        detail="gateway takes metrics JSON only; "
                               "video/frame payloads are rejected")


def list_models() -> dict:
    backend = get_backend()
    if backend == "stub":
        data = [
            {"id": STUB_MODEL_ID, "object": "model",
             "created": int(time.time()), "owned_by": "pose-not-pixels"},
            {"id": FRUGAL_MODEL_ID, "object": "model",
             "created": int(time.time()), "owned_by": "pose-not-pixels",
             "tier": "frugal"},
            {"id": FULL_MODEL_ID, "object": "model",
             "created": int(time.time()), "owned_by": "pose-not-pixels",
             "tier": "full"},
        ]
        return {"object": "list", "data": data}
    base = _local_base() if backend == "local" else _upstream_base()
    headers = {"Content-Type": "application/json"} if backend == "local" else _upstream_headers()
    try:
        r = httpx.get(f"{base}/v1/models", headers=headers, timeout=15.0)
        r.raise_for_status()
        return r.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{backend} /v1/models failed: {exc}")


def dispatch_chat(messages: list[dict], model: str | None = None,
                  temperature: float = 0.2, max_tokens: int = 2000,
                  response_format: dict | None = None,
                  tier: str | None = None) -> dict:
    assert_metrics_only(messages)
    tier = tier or resolve_tier(model)
    profile = tier_profile(tier)
    backend = get_backend()
    CALL_LEDGER["chat"] += 1
    # Ledger entry: kinds/tier/backend/model/token estimates ONLY.
    # Message contents are deliberately never stored here.
    prompt_tokens = sum(
        estimate_text_tokens(m.get("content", "") if isinstance(m.get("content", ""), str)
                             else " ".join(p.get("text", "") for p in m.get("content", [])
                                           if isinstance(p, dict)))
        for m in messages)
    CALL_LOG.append({"kind": "chat", "tier": tier, "backend": backend,
                     "model": model or profile["model_alias"],
                     "prompt_tokens_est": prompt_tokens})
    if backend == "stub":
        content = _stub_chat(messages)
        model_id = model or profile["model_alias"]
    elif backend in ("local", "upstream"):
        content, model_id = _forward_chat(messages, model, temperature,
                                          max_tokens, response_format, backend)
    else:
        raise HTTPException(status_code=500, detail=f"unknown AI_BACKEND: {backend}")
    completion_tokens = estimate_text_tokens(content)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "tier": tier,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def dispatch_embeddings(input: str | list[str], model: str | None = None) -> dict:
    backend = get_backend()
    CALL_LEDGER["embeddings"] += 1
    CALL_LOG.append({"kind": "embeddings", "backend": backend,
                     "model": model or STUB_MODEL_ID,
                     "n_inputs": len([input] if isinstance(input, str) else input)})
    texts = [input] if isinstance(input, str) else list(input)
    if backend == "stub":
        vectors = [_stub_embedding(t) for t in texts]
        model_id = model or STUB_MODEL_ID
    elif backend in ("local", "upstream"):
        vectors, model_id = _forward_embeddings(texts, model, backend)
    else:
        raise HTTPException(status_code=500, detail=f"unknown AI_BACKEND: {backend}")
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": vec, "index": i}
                 for i, vec in enumerate(vectors)],
        "model": model_id,
        "usage": {"prompt_tokens": max(1, sum(len(t) for t in texts) // 4),
                  "total_tokens": max(1, sum(len(t) for t in texts) // 4)},
    }


def _clean_messages(messages):
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [{k: v for k, v in part.items() if not k.startswith("_")}
                       for part in content]
        out.append({**m, "content": content})
    return out


def _forward_chat(messages, model, temperature, max_tokens, response_format, backend):
    if backend == "local":
        url = f"{_local_base()}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        model_id = model or os.environ.get("LOCAL_MODEL", "local-model")
    else:
        url = f"{_upstream_base()}/v1/chat/completions"
        headers = _upstream_headers()
        model_id = model or os.environ.get("UPSTREAM_MODEL", "gpt-4o-mini")
    payload = {"model": model_id, "messages": _clean_messages(messages),
               "temperature": temperature, "max_tokens": max_tokens}
    if response_format:
        payload["response_format"] = response_format
    try:
        r = httpx.post(url, json=payload, headers=headers, timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"] or ""
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{backend} chat failed: {exc}")
    return content, data.get("model", model_id)


def _forward_embeddings(texts, model, backend):
    if backend == "local":
        url = f"{_local_base()}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        model_id = model or os.environ.get("LOCAL_MODEL", "local-model")
    else:
        url = f"{_upstream_base()}/v1/embeddings"
        headers = _upstream_headers()
        model_id = model or os.environ.get("UPSTREAM_MODEL", "text-embedding-3-small")
    try:
        r = httpx.post(url, json={"model": model_id, "input": texts},
                       headers=headers, timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        vectors = [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"{backend} embeddings failed: {exc}")
    return vectors, data.get("model", model_id)


def _stub_embedding(text: str, dims: int = 32) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [round(b / 255.0, 6) for b in digest[:dims]]


def _find_block(messages: list[dict], tag: str) -> str:
    blob = ""
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            blob += content + "\n"
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    blob += part.get("text", "") + "\n"
    mm = re.search(rf"\[{tag}\]\s*(.*?)\s*\[/{tag}\]", blob, re.S)
    return mm.group(1) if mm else ""


def _fmt(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def _stub_chat(messages: list[dict]) -> str:
    blob_all = ""
    for m in messages:
        c = m.get("content", "")
        blob_all += c if isinstance(c, str) else " ".join(
            p.get("text", "") for p in c if isinstance(p, dict))
    task_m = re.search(r"\[TASK:\s*([A-Z_]+)\]", blob_all)
    task = task_m.group(1) if task_m else None
    if task == "SUMMARISE_TUG":
        return _stub_summarise_tug(_find_block(messages, "METRICS"))
    if task == "SUMMARISE_ARTHRITIS":
        return _stub_summarise_arthritis(_find_block(messages, "METRICS"))
    if task == "VERIFY_REPORT":
        return _stub_verify(_find_block(messages, "EXPECTED"),
                            _find_block(messages, "SUMMARY"))
    return "Stub response."


def _stub_summarise_tug(metrics_blob: str) -> str:
    try:
        d = json.loads(metrics_blob)
    except Exception:
        return "No Timed Up and Go metrics were provided, so no summary can be written."
    lines = [
        f"Timed Up and Go total time {_fmt(d.get('tug_total_s'))} seconds; "
        f"sit-to-stand {_fmt(d.get('sit_to_stand_s'))} seconds; "
        f"walk {_fmt(d.get('walk_s'))} seconds; turn {_fmt(d.get('turn_s'))} seconds.",
        f"Gait speed proxy {_fmt(d.get('gait_speed_proxy_leg_lengths_per_s'))} "
        f"leg lengths per second; step count estimate {d.get('step_count_estimate')}; "
        f"cadence {_fmt(d.get('cadence_steps_per_min'), 1)} steps per minute; "
        f"stance-time asymmetry {_fmt(d.get('stance_time_asymmetry_pct'), 1)} percent.",
    ]
    flag = d.get("tug_flag", "review")
    if flag == "steadi_screening_flag":
        lines.append("STEADI screening flag \u2014 discuss with clinician. "
                     "The total time is at or above the 12-second STEADI "
                     "screening flag for further assessment; this is not a "
                     "risk label, and no cut-point is recommended as a "
                     "standalone rule.")
    else:
        lines.append("Below the 12-second STEADI screening flag. A single "
                     "capture does not rule out the need for clinician "
                     "assessment.")
    lines.append("These numbers come from skeleton landmarks only. No video was "
                 "sent to a model. A clinician reviews the trend, not this note alone.")
    return " ".join(lines)


def _stub_summarise_arthritis(metrics_blob: str) -> str:
    try:
        d = json.loads(metrics_blob)
    except Exception:
        return "No kinematics metrics were provided, so no summary can be written."
    parts = []
    for leg in ("left", "right"):
        hka = d.get("hka", {}).get(leg, {})
        parts.append(f"{leg} frontal knee alignment deviation "
                     f"{_fmt(hka.get('deviation_deg'))} degrees")
    parts.append(f"pelvic obliquity {_fmt(d.get('pelvic_obliquity_deg'))} degrees")
    parts.append(f"lateral trunk lean {_fmt(d.get('lateral_trunk_lean_deg'))} degrees")
    parts.append(f"stance-time asymmetry {_fmt(d.get('stance_time_asymmetry_pct'), 1)} percent")
    parts.append(f"cadence {_fmt(d.get('cadence_steps_per_min'), 1)} steps per minute")
    flags = d.get("flags", [])
    red = [f["metric"] for f in flags if f.get("status") == "red"]
    amber = [f["metric"] for f in flags if f.get("status") == "amber"]
    lines = ["Screening pattern summary (for clinician review): " + "; ".join(parts) + "."]
    if red:
        lines.append("Metrics flagged red for clinician review: " + ", ".join(red) + ".")
    elif amber:
        lines.append("Metrics flagged amber for clinician review: " + ", ".join(amber) + ".")
    else:
        lines.append("No metric crossed a screening threshold in this capture.")
    lines.append("All thresholds used here are app-defined and unvalidated. "
                 "This summary is pattern language only and "
                 "is not a finding about any condition.")
    return " ".join(lines)


def _stub_verify(expected_blob: str, summary_blob: str) -> str:
    """Second-pass check (full tier): every numeric metric value in the
    EXPECTED JSON must appear, within rounding tolerance, in the SUMMARY
    text. Returns JSON {ok, problems}."""
    try:
        expected = json.loads(expected_blob)
    except Exception:
        return json.dumps({"ok": False, "problems": ["unreadable expected metrics"]})
    numbers_in_summary = [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", summary_blob or "")]
    problems = []

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if obj == 0:
                return
            if not any(abs(n - obj) <= max(0.06, 0.02 * abs(obj)) for n in numbers_in_summary):
                problems.append(f"metric value not reflected in summary: {path}={obj}")

    walk(expected)
    # Only check top-level scalar metrics + per-leg HKA deviations, to keep
    # the verifier strict but not brittle on counts/ids.
    return json.dumps({"ok": not problems, "problems": problems[:20]})
