# Architecture

## The one idea

The product refuses to use a vision model at all. A 30-second TUG test
or a walking pass is video — ruinously expensive and PHI-heavy if
streamed to a vision model, and unnecessary: pose estimation on the
phone reduces the video to 33 skeleton landmarks, and ordinary code
computes the clinical numbers. The video never leaves the device; the
downstream model input is ~100 tokens of metrics JSON, which a small
model merely phrases. The skeleton replaces the pixels — that is the
entire architecture, and it is also the privacy story.

```
  phone (integration path, not shipped in v0.1.0)
    video ──► MediaPipe Pose (on-device) ──► 33 landmarks/frame
    video deleted by default
                        │  landmark JSON only
                        ▼
            ┌──────────────────────────────────────────────────────────┐
            │                      FastAPI app                          │
            │                                                          │
  HTTP ───► │  /api/*  tug · arthritis · trend · demo · thresholds     │
            │     │                                                    │
            │     ▼                                                    │
            │  app/pose/  domain module                                │
            │   geometry (joint angles, leg length, HKA deviation)     │
            │   kinematics (TUG phases, contacts, ROM, flags, THRESHOLDS)│
            │   service (captures, gateway summaries, trend deltas)    │
            │   synth (ground-truth fixtures) · models (reject video)  │
            │     │  phrasing only, numbers already computed           │
            │     ▼                                                    │
            │  app/gateway.py  embedded OpenAI-compatible gateway      │
            │   /v1/models · /v1/chat/completions · /v1/embeddings    │
            │   tiers: pose-not-pixels-frugal | pose-not-pixels-full  │
            │   frugal: phrase (≤ ~220 tokens)                        │
            │   full: phrase + longitudinal context + number verifier │
            │     │                                                    │
            │     ├─ stub     deterministic, no model                 │
            │     ├─ local    Ollama / llama.cpp / vLLM on-site       │
            │     └─ upstream any OpenAI-compatible provider          │
            │                                                          │
            │  app/safety.py  PHI redaction gate + banned-language    │
            │                 scan                                    │
            └──────────────────────────────────────────────────────────┘
```

## What v0.1.0 actually ships (and what it does not)

v0.1.0 accepts landmark JSON and computes everything from it. It does
NOT ship on-device MediaPipe extraction, a phone app, or a camera
pipeline, and it makes no claim of live camera extraction. The
integration path for a real deployment is:

1. Run MediaPipe Pose on the device (phone CPU, 30 FPS class hardware).
2. Convert each frame's 33 landmarks to this JSON: keys are the
   MediaPipe landmark names (`left_hip`, `right_knee`, … — full list in
   `app/pose/geometry.py: NAMES`, index order = MediaPipe order).
3. **Flip y**: MediaPipe image coordinates use y-down; this product's
   convention is y-up (larger = higher). The converter negates y (or
   uses `1 - y` in normalised coordinates) at conversion time. All
   angle maths is vector geometry, so the flip changes only sign
   conventions, not magnitudes — but it must be consistent, or pelvic
   obliquity and trunk lean signs flip.
4. Delete the video by default; keep the landmark sequence and the
   derived kinematics. Send only metrics JSON to the gateway — never
   frames, never video, never a photo.

The demo UI and the test suite use synthetic landmark JSON built by
`app/pose/synth.py` with known ground truth, in exactly this shape.

## The shared-skeleton answer, Week 5

Weeks 1–4 established the pattern: one platform skeleton (gateway +
redaction gate + verified generation + eval harness), a new domain
module per week. Week 5 reuses `gateway.py` and `safety.py` with three
changes forced by the pose domain:

1. **The gateway is metrics-only by construction.** `dispatch_chat`
   rejects any `image_url` / `video_url` content part or video-keyed
   payload before contacting a backend, and the ledger stores no
   message contents at all. PillTray's gateway counted VLM calls;
   this gateway's central assertion is that there are none to count.
2. **Tiers change phrasing, never vision.** Frugal phrases the metrics
   (≤ ~220 tokens, enforced by truncation in the service). Full adds
   previous captures to the prompt and runs a second-pass verifier:
   every number the summary is expected to carry must appear in it
   within rounding tolerance (`VERIFY_REPORT` in the stub; a real
   verifier model call on local/upstream backends). Landmark
   extraction is identical in both tiers because it happens on the
   device, outside the gateway entirely.
3. **Embeddings embed metrics, not images.** `/v1/embeddings` on a
   capture's metrics JSON gives a deterministic stub vector (SHA-256
   hash, 32 dims) used for longitudinal similarity; on a real backend
   it is a normal text embedding of the same JSON.

## The kinematics pipeline (app/pose/kinematics.py)

1. **TUG.** Markers preferred (`start, stand, turn_start, turn_end,
   sit`); without them, `infer_tug_boundaries` uses hip-height
   plateau/sitting-floor crossings and shoulder-width narrowing during
   the turn. Walk time = out-walk + back-walk. Walk frames are split
   into contiguous segments at the turn gap before stance statistics,
   so no stance duration ever spans the turn.
2. **Contacts / stance.** A foot contact is the start of a contiguous
   run of frames where ankle height sits near the ground. Stance
   duration is measured by interpolating ankle-height crossings of a
   tight threshold (ground + 5% of range) — a loose threshold would
   add a slice of swing whose length differs per leg and corrupt
   asymmetry. The first contact's duration is skipped (a capture may
   start mid-stance).
3. **Stride.** Successive same-leg contacts, distance in the x–z plane
   (forward = x on a side pass, z on a front pass toward the camera),
   normalised by leg length (hip–knee + knee–ankle segment sum).
4. **Standing panel.** HKA deviation per leg (180 − hip–knee–ankle
   angle), pelvic obliquity (hip line vs horizontal), lateral trunk
   lean (shoulder-mid → hip-mid vs vertical), mid-hip offset vs the
   ankle midpoint, normalised by leg length.
5. **Gait panel.** Side pass → knee/hip flexion ROM (max − min flexion
   over the pass, flexion = 180 − joint angle). Front pass → varus
   thrust (HKA deviation range over near-ground stance frames only),
   Trendelenburg drop (mean absolute pelvic angle while the other
   foot is lifted), step width (median ankle separation, normalised),
   stance/stride asymmetry, cadence.
6. **Flags.** `THRESHOLDS` (named constant, source note per threshold)
   maps each metric to green/amber/red. Every flag object carries
   `threshold_status: "APP_DEFINED_UNVALIDATED"` (STEADI flag excepted), its source note, its caveat, and a skeleton
   coordinate summary as evidence. A metric with missing landmarks or
   a missing view is `null`, never guessed.

## Verified generation, applied to summaries

Weeks 1–2 verified claims against sources; Week 3 verified numbers
against measurements; Week 4 verified reports against lists. Week 5's
version is the number verifier: the full tier's second pass checks
that every expected metric value appears in the summary within
rounding tolerance. A summary that invents or drops a number fails
the check in the response payload, where a human or a test can see it.
