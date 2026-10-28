> ## ⚠️ PROOF OF CONCEPT — NOT CLINICALLY TESTED
>
> **This tool has not been tested clinically and is a proof-of-concept for
> research and software demonstration only. It is not a medical device, it
> does not diagnose, treat, or rule out any condition, and its outputs may
> be wrong — including false positives and false negatives.**
>
> **Do not use it to make decisions about your health.**
> For any medical issue or concern, seek a licensed healthcare professional.
> **In an emergency, go to your nearest emergency room or dial 911 immediately.**

▶ **Try the live demo:** https://rtmalikian.github.io/pose-not-pixels/ — static preview with demo data, no backend connected.

# Pose-Not-Pixels — a 30-second TUG test and knee/hip gait screen, as numbers and a trend, with zero video sent to any model

**The measurable upgrade.** Today a Timed Up and Go (TUG) test is a stopwatch number on paper, and a gait screen is a clinician's eye — neither trends over time at home, and neither is comparable month to month. Pose-Not-Pixels turns a phone capture into five TUG numbers and a full knee/hip kinematics panel, stored per patient, with deltas vs the previous capture — and it does it without ever sending video to a model. The video is reduced to 33 skeleton landmarks on the device; deterministic code computes every number; the model only phrases ~100 tokens of metrics JSON into a plain-language note. That is the whole privacy and cost story: **video = 0 tokens, 0 model calls, 0 frames in any gateway log** (asserted in CI).

**Demand / gap.** Falls: CDC reports over 14 million people (adults 65+) report falling each year — 1 in 4 older adults — with over 43,000 fall deaths in 2024, and fewer than half of fallers tell their doctor. The STEADI TUG value is ≥12 s as a screening flag for further assessment only — never a risk label (see below). Validated pose-based gait work exists in papers (MediaPipe pose vs lab motion capture), but no self-hostable, small-clinic/patient product was found that does "do the TUG, get a trend, share it with your clinician" — commercial fall tech is mostly wearables and subscriptions. Knee/hip screening has the same gap: alignment and gait patterns are observable from walking, but there is no open, landmark-only screening panel with stated, reviewable thresholds. See EVIDENCE.md for claim → citation → evidence tier.

## How it works

1. **Capture (on device, documented integration path).** MediaPipe Pose runs on the phone and outputs 33 landmarks per frame (x, y, z, visibility). v0.1.0 accepts that landmark JSON directly — see "What v0.1.0 actually does" below. Raw video is deleted by default and is never an input type to this API.
2. **Maths in code.** `app/pose/kinematics.py` computes TUG phases, gait events (foot contacts from ankle-height runs, interpolated stance times), joint angles, asymmetries and cadence. A metric is `null` when its landmarks or view are missing — never guessed.
3. **Phrasing only.** The embedded OpenAI-compatible gateway (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`) phrases the metrics JSON. The gateway rejects any image/video content part (HTTP 400) and its ledger records kinds/tokens only, never contents.
4. **Trend.** Captures are stored per patient (in-memory in v0.1.0); `GET /api/patients/{id}/trend` lists them with deltas vs the previous capture of the same kind.

### What v0.1.0 actually does — stated plainly

v0.1.0 accepts **landmark JSON** (33 MediaPipe Pose landmarks per frame, as documented in `docs/ARCHITECTURE.md`). It does **not** implement live camera extraction: there is no on-device MediaPipe code in this repo, and no claim of live camera extraction is made. Real on-device MediaPipe extraction is the documented integration path in `docs/ARCHITECTURE.md` (convert MediaPipe output to this JSON, flipping y to the product's up-positive convention, on the device; delete the video). The demo UI and the test suite run on synthetic landmark sequences with known ground truth.

## Pack 1 — Falls / TUG trend

Input: a landmark sequence with per-frame timestamps, plus event markers **(preferred)** — `start`, `stand` (end of sit-to-stand), `turn_start`, `turn_end`, `sit` — or, without markers, phases inferred from hip-height and shoulder-width signatures (documented heuristic in `infer_tug_boundaries`; the API reports `phase_method: markers | inferred`).

Outputs (units in the API response):

| Metric | Unit |
|---|---|
| TUG total time | s |
| Sit-to-stand time | s |
| Walk time (out + back) | s |
| Turn time | s |
| Gait speed proxy (stride length / time, normalised by leg length) | leg lengths / s |
| Stance-time asymmetry | % |
| Step count estimate | steps |
| Cadence | steps / min |

Longitudinal: every capture is stored; the trend endpoint shows each capture's delta vs the previous one (e.g. TUG total +4.8 s on the slow demo).

**TUG presentation rule.** A TUG result is reported as a time + trend for clinician review. At ≥12 s the label is exactly "STEADI screening flag — discuss with clinician". CDC STEADI uses ≥12 s only as a screening flag for further assessment; Barry 2014 found sensitivity 0.31 at 13.5 s, and Schoene 2013 (53 studies) concluded no cut-point can be recommended. No risk-verdict or faller label appears anywhere in this product.

## Pack 2 — Knee & Hip Arthritis Kinematics Screening

Input: a standing-posture landmark frame set (front view, 5-second hold) plus walking landmark sequences (front/back/side passes, ~10–20 steps). **Raw video is never an input type** — the API rejects any video/frame payload shape (tested), and the gateway never receives frames.

Metrics (all computed in code; per leg where applicable; `null` when the required view is absent):

| Metric | Unit | Notes |
|---|---|---|
| Hip–knee–ankle frontal angle, deviation from straight | deg / leg | varus/valgus proxy |
| Pelvic obliquity (standing) | deg | hip line vs horizontal |
| Lateral trunk lean (standing) | deg | shoulder-mid vs hip-mid vs vertical |
| Mid-hip offset vs base of support | leg lengths | weight-shift proxy |
| Knee flexion ROM per gait cycle | deg | sagittal knee angle, side pass (or 3D angle when world landmarks given) |
| Hip flexion/extension ROM | deg | side pass |
| Varus thrust proxy (frontal knee alignment change during stance) | deg / leg | |
| Trendelenburg pelvic drop during single-leg support | deg / leg | mean pelvic angle during single support |
| Stance-time asymmetry | % | |
| Stride-length asymmetry | % | normalised by leg length |
| Cadence | steps / min | |
| Step width | leg lengths | median ankle separation, normalised |

Output: a **traffic-light screening flag for clinician review** (green / amber / red) per joint per metric. Each flag shows the measured value, the threshold used, the threshold's source note, and evidence (a skeleton landmark-coordinate summary — never video). **This is a screening pattern, not a finding about any condition.**

### Metrics + thresholds — ALL APP-DEFINED AND UNVALIDATED (except the STEADI flag)

Every threshold below is **app-defined and unvalidated**, labelled `APP_DEFINED_UNVALIDATED` in the README, SECURITY.md, demo UI and every API flag object, until tested on labelled clinical data. No sentence in this repo claims these thresholds are sourced from the literature as screening cutoffs. Each flag shows the measured value, the threshold, the caveat, and — for frontal HKA — a ±3 deg (or greater) uncertainty band. No validated performance numbers are claimed for this product: accuracy figures for real patients do not exist yet.

The only citable reference points kept, with their exact caveats, are:

- **STEADI ≥12 s — screening flag only.** CDC STEADI uses ≥12 s only as a screening flag for further assessment; Barry 2014 found sensitivity 0.31 at 13.5 s; Schoene 2013 (53 studies) concluded no cut-point can be recommended.
- **Gait speed 0.8 / 1.0 m/s (Studenski 2011) — interpretive survival references only, not diagnostic cutoffs, and not thresholds in this product.** This product's gait-speed proxy is in leg lengths per second, not m/s, so the Studenski references are context only and are never applied as cutoffs here.
- **Varus association with OA progression — no degree threshold.** Varus alignment is associated with OA progression in the literature, but no degree value is cited as a screening cutoff; the HKA split below is app-defined.
- **Trendelenburg static-test range 2–8 deg — contested, static test, not gait-validated.** A ≥4 deg flag here is borrowed/app-defined, not a validated gait cutoff.

Turn time has no threshold in this product: it is reported as a time only.

| Metric | Amber | Red | Note (see `THRESHOLDS` in `app/pose/kinematics.py` for the full text) |
|---|---|---|---|
| TUG total | — | ≥ 12 s | STEADI screening flag only (see above) |
| HKA deviation | ≥ 3 deg | > 5 deg | App-defined split; reported as value ±3 deg minimum uncertainty band (see angle reporting below) |
| Pelvic obliquity | ≥ 3 deg | > 5 deg | App-defined (provisional) |
| Lateral trunk lean | ≥ 3 deg | > 5 deg | App-defined (provisional) |
| Knee flexion ROM | < 50 deg | < 40 deg | App-defined (provisional); low values flagged |
| Hip ROM | < 30 deg | < 20 deg | App-defined (provisional); low values flagged |
| Varus thrust proxy | ≥ 3 deg | > 5 deg | App-defined (provisional); no citable cutoff for thrust degrees |
| Trendelenburg drop | ≥ 4 deg | > 8 deg | Borrowed/app-defined from the contested static-test range (see above) |
| Stance-time asymmetry | ≥ 10 % | > 20 % | App-defined (provisional) |
| Stride-length asymmetry | ≥ 10 % | > 20 % | App-defined (provisional) |
| Cadence | < 100 /min | < 80 /min | App-defined (provisional); low values flagged |
| Step width | ≥ 0.25 leg lengths | > 0.40 | App-defined (provisional) |

**Angle reporting.** A frontal HKA angle is never reported to 0.1 deg precision as if radiographic. It is always reported with a ±3 deg (or greater) uncertainty band. The band's narrow-scope basis: Ge 2025 found a mean absolute difference of 1.44 deg with 95% limits about ±3 deg — for OpenPose, in severe pre-surgical knee OA only. That scope is not generalised to MediaPipe or to mild/moderate OA, and the ±3 deg band here is a minimum, not a claim that real error is only ±3 deg.

## Tiers — same gateway, same vision, different phrasing budget

| | Frugal (default) | Full |
|---|---|---|
| Model alias | `pose-not-pixels-frugal` | `pose-not-pixels-full` |
| Summary | Phrasing only, max ~220 tokens | Richer longitudinal context (previous captures in the prompt) + second-pass verifier that the summary's numbers match the metrics JSON |
| Vision / landmark extraction | Identical (on-device, both tiers) | Identical (on-device, both tiers) |

**Full buys phrasing, context and verification only.** It buys no new vision, no new landmarks and no new measurements — by design. Select by the `model` name or the `X-Tier: frugal|full` header; `POSE_TIER` sets the default.

## Architecture

```
landmarks JSON ──► FastAPI /api/* ──► app/pose/kinematics.py (deterministic maths)
                        │                        │
                        │ metrics JSON only      ▼
                        └──────────► app/gateway.py (/v1/*, stub|local|upstream)
                                        │  frugal: phrase  │  full: phrase + verify
                                        ▼
                                   app/safety.py (PHI redaction + banned-language scan)
```

See `docs/ARCHITECTURE.md` for the full diagram, the MediaPipe integration path, and why the stub backend is the honest default for tests.

## Eval harness — measured on synthetic ground truth (CI)

`python -m pytest -q` — 54 tests green at v0.1.0. Headline numbers, all measured in CI on synthetic landmark sequences with known angles/times (not on patients). **These are reported synthetic errors, not clinical accuracy, and the old sub-two-degree target has been dropped.**

- **Angles, reported honestly (synthetic):** HKA deviation error **0.00 deg** on known deviations (0/3/5/8/12 deg); knee ROM error **0.09 deg** (measured 54.91 vs 55 truth); hip ROM error **0.02 deg** (19.98 vs 20); varus-thrust error **0.11 deg** (6.89 vs 7.0); Trendelenburg error **0.00 deg** (6.0 vs 6.0). Pre-specified clinical-acceptability bar to be demonstrated on real captures: **<5 deg** — synthetic results do not demonstrate it.
- **Published pose-error ballpark (for context, not this product):** published single-phone MediaPipe vs Qualisys mean error is **7.16 deg** (n=6, healthy, non-gait); multi-camera OpenCap is about **4.5 deg**. Real single-phone error on patients is expected to be of that order or worse, which is why synthetic sub-degree errors must never be quoted as product accuracy, and why frontal HKA always carries a ±3 deg (or greater) band.
- **Stance-time asymmetry error: 1.72 points** (measured 31.61 vs 33.33 truth). Equal-stance TUG asymmetry measures 0.00.
- **TUG time error: 0.00 s** with markers and **0.00 s** with inferred phases on the standard synthetic TUG (10.4 s total; sit-to-stand 1.2 s; walk 7.2 s; turn 2.0 s). The <0.5 s figure is a **target to be demonstrated for pose-video TUG only** — the 0.42 s figure in the literature was inertial-sensor, not pose video, and is not claimed here as a pose-video result.
- **Robustness:** landmark jitter (±0.01 units) keeps HKA error within the 5 deg bar; dropping every 3rd/4th frame leaves TUG totals exact (markers) and gait asymmetry within tolerance; occluded (visibility 0) landmarks degrade to `null`, never a crash.
- **Zero-video-payload test:** after gateway calls, the call ledger contains no image/video/frame keys or values.
- **Disclaimer test:** README line 1 is the proof-of-concept banner.
- **Honesty test:** README, SECURITY.md and the demo UI contain no percentage attached to the words for validated performance (see thresholds section).

These numbers are synthetic-fixture only. Real-camera, real-patient validation is roadmap issue #1 and has not been done.

## Capture guide

Camera at hip height, 3–4 m from the walk path, whole body in frame. Front pass: walk toward the camera. Side pass: walk across the frame. TUG: chair, 3 m walk, turn, walk back, sit. Fitted clothes (loose trousers hide knees). Note any cane, walker or helper — aids change gait numbers and are not comparable to unaided captures. 2D pose is a proxy: frontal alignment and range estimates degrade with camera angle, distance, loose clothing and occlusion. Repeat captures with the same phone, setup and shoes. The full guide is in the demo UI.

## Run it

```bash
docker compose up --build        # serves on http://localhost:8000
# or, without Docker:
pip install -r requirements.txt
AI_BACKEND=stub uvicorn app.main:app --port 8000
python -m pytest -q              # 54 tests
```

Environment: `AI_BACKEND` (stub | local | upstream), `LOCAL_MODEL_URL`, `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY`, `POSE_TIER` (frugal | full). See `.env.example`.

## API sketch

```bash
curl localhost:8000/healthz
curl localhost:8000/v1/models
curl -X POST localhost:8000/api/demo/tug?scenario=tug_standard
curl -X POST localhost:8000/api/demo/arthritis?scenario=arthritis_varus_left
curl localhost:8000/api/patients/demo/trend
# Real captures (landmark JSON only — video payloads are rejected):
curl -X POST localhost:8000/api/tug -H 'Content-Type: application/json' -d @tug.json
curl -X POST localhost:8000/api/arthritis -H 'Content-Type: application/json' -d @arthritis.json
```

## Limits (see SECURITY.md for the full list)

- Proof of concept, not clinically tested, not a medical device. Screening patterns only; a clinician reviews every flag.
- Landmark JSON in, numbers out. No live camera extraction in v0.1.0; the MediaPipe step is a documented integration path, not shipped code.
- In-memory storage in v0.1.0; captures do not persist across restarts.
- 2D pose is a proxy for 3D motion; thresholds are app-defined and unvalidated (STEADI flag excepted, as a screening flag only); a single capture never rules anything out.
- No treatment recommendations, no diagnosis language: outputs are scanned for banned phrases in CI.

## Roadmap

1. Real-camera, real-patient validation set (labelled TUG + gait captures) — required before any performance claim.
2. On-device MediaPipe extraction (phone/PWA) producing this JSON, video deleted by default.
3. Persistent storage + clinician export (CSV/FHIR-shaped).
4. Longitudinal anomaly notes ("turn time worsened 3 captures running") in the full tier.
5. Back/side pass fusion and 3D (world landmark) angle paths.
