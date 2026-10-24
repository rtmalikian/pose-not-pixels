# Security and honest limits — Pose-Not-Pixels

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

## What this product is

A screening-pattern and trending aid for clinician review. It computes
numbers from skeleton landmarks and phrases them. It is not a fall
predictor, not a gait diagnostic, and not a substitute for a clinical
gait examination.

## Privacy design (the central claim)

- **Video never leaves the device** in the documented integration path
  (MediaPipe Pose on the phone; video deleted by default). In v0.1.0
  the API accepts landmark JSON only.
- **Video is never an input type.** Request models forbid extra fields,
  so a video/frame payload shape is rejected before processing, and
  the gateway rejects image/video content parts with HTTP 400.
- **The gateway takes metrics JSON only.** Its ledger records call
  kinds, tiers, backends and token estimates — never message contents,
  landmarks or frames. A CI test scans the ledger for image/video/frame
  keys and must find none.
- **PHI redaction gate.** Patient labels pass through `app/safety.py`
  redaction (email, phone, SSN, MRN, address patterns) before storage
  or logging. Logs carry PHI kinds only, never values.
- **Backends.** `stub` (default, no model, no network) is safe anywhere.
  `local` (Ollama / llama.cpp / vLLM on your own hardware) keeps metrics
  in the building. `upstream` sends metrics JSON — never video — to the
  configured provider; do not use it with identifiable patient data
  unless you have reviewed that provider's terms.

## Thresholds — all app-defined and unvalidated (STEADI flag excepted)

Every screening threshold in `app/pose/kinematics.py` (`THRESHOLDS`)
except the STEADI TUG value is app-defined and labelled
`APP_DEFINED_UNVALIDATED` in the README, this file, the demo UI and
every API flag object, until tested on labelled clinical data. No
threshold here is presented as sourced from the literature as a
screening cutoff. The TUG 12-second value is a STEADI screening flag
only: CDC STEADI uses >=12 s only as a screening flag for further
assessment, never as a risk label; Barry 2014 found sensitivity 0.31
at 13.5 s, and Schoene 2013 (53 studies) concluded no cut-point can
be recommended. The only other citable reference points, with their
exact caveats, are gait-speed 0.8 / 1.0 m/s (Studenski 2011,
interpretive survival references only, not cutoffs, and not applied
to this product's leg-lengths-per-second proxy), varus association
with OA progression (no degree threshold), and the contested
static-test Trendelenburg range 2-8 deg (not gait-validated; a >=4 deg
flag is borrowed/app-defined). No validated performance numbers are
claimed for this product. Accuracy figures for real patients do not
exist yet; this repo states that plainly rather than implying
otherwise.

**Angle reporting.** A frontal HKA angle is never reported to 0.1 deg
precision as if radiographic. It always carries a +/-3 deg (or greater)
uncertainty band. Ge 2025 (OpenPose, severe pre-surgical knee OA only)
found a mean absolute difference of 1.44 deg with 95% limits about
+/-3 deg; that narrow scope is not generalised to MediaPipe or to
mild/moderate OA. Published single-phone MediaPipe vs Qualisys mean
error is 7.16 deg (n=6, healthy, non-gait); multi-camera OpenCap is
about 4.5 deg.

## Banned output language

Product-facing summaries are scanned for banned phrases (diagnosis
verdicts, treatment recommendations, "you should …" medical advice).
Tests fail the build if banned language appears in a summary. Pattern
language only: a flag is a prompt for clinician review, never a finding
about a condition.

## Known limits

- Not clinically tested; synthetic-fixture eval only (see README).
- No live camera extraction in v0.1.0; the MediaPipe step is a
  documented integration path, not shipped code.
- 2D pose is a proxy: camera angle, distance, loose clothing, walking
  aids and occlusion degrade landmarks. Missing landmarks yield `null`
  metrics, not guesses — but a degraded capture can still produce a
  plausible-looking number, which is why trend (same setup, repeated)
  matters more than any single capture.
- In-memory storage in v0.1.0: captures vanish on restart, and there
  is no authentication. Do not deploy this as-is with real patient data;
  put it behind your clinic's authentication and storage first.
- A single capture, and a green flag, never rule out the need for
  clinician assessment or any condition.

## Reporting issues

Open an issue on the repository. Do not include patient video, patient
landmarks from real people, or identifiable health information in any
report — synthetic fixtures only.
