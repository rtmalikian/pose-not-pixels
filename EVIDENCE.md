# Evidence base — Pose-Not-Pixels

Rule for this repo: every load-bearing claim carries a citation and an
evidence tier. Weak claims are downgraded, unsourced claims are removed,
and this product's own real-world efficacy is stated as **unproven**
until a labelled validation set exists (roadmap issue #1).

Tiers: **1** systematic review / meta-analysis / guideline ·
**2** RCT / large cohort / validated real-patient study ·
**3** small observational study ·
**4** vendor claim / synthetic-fixture result — labelled, never dressed
as clinical evidence.

## (a) Problem magnitude

| Claim | Citation | Tier |
|---|---|---|
| Over 14 million people (adults 65+) in the US report falling each year (1 in 4 older adults); over 43,000 fall deaths in 2024; fewer than half of fallers tell their doctor. Never phrased as a falls-per-year count. | CDC Older Adult Falls Data; CDC About Older Adult Fall Prevention (cdc.gov/falls). | 1 (national surveillance / guideline source) |
| Timed Up and Go >=12 s is a STEADI screening flag for further assessment only, never a risk label. Barry 2014 found sensitivity 0.31 at 13.5 s; Schoene 2013 (53 studies) concluded no cut-point can be recommended. | CDC STEADI guidance; Barry 2014; Schoene 2013. | 1 (guideline / systematic review) for the flag and the caveats |
| Knee/hip osteoarthritis is a leading cause of gait change and mobility limitation in older adults, and frontal knee alignment / gait asymmetry are among the patterns clinicians examine. | General clinical background; specific screening cutoffs for this product are NOT taken from a single validated source — see (c). | 1 for the broad problem; cutoffs downgraded below |
| MediaPipe Pose (33 landmarks, phone CPU) shows good-to-excellent agreement with lab motion capture (e.g. Vicon) for spatiotemporal gait parameters in published comparisons — but single-phone joint-angle error is large: published single-phone MediaPipe vs Qualisys mean error 7.16 deg (n=6, healthy, non-gait); multi-camera OpenCap about 4.5 deg. Frontal HKA reporting therefore always carries a +/-3 deg (or greater) band, based narrowly on Ge 2025 (OpenPose, severe pre-surgical knee OA only: mean absolute difference 1.44 deg, 95% limits about +/-3 deg — not generalised to MediaPipe or mild/moderate OA). | JMIR/PMC gait studies cited in the vision/voice gap report (2026-10-24), §2/§5.3; Ge 2025. | 2–3 (validation studies; agreement varies by parameter and setup — not a blanket equivalence claim) |
| Commercial fall tech is mostly wearables/subscriptions; no self-hostable small-clinic/patient phone-TUG product surfaced in the gap-report searches. | Vision/voice gap report (2026-10-24), §2 — search-bounded absence claim, not proof of non-existence. | 4 (search-bounded) |

## (b) Solution method, in principle

| Claim | Citation | Tier |
|---|---|---|
| Reducing a video to skeleton landmarks on-device, then computing clinical numbers deterministically in code, removes the need to send video to a model and makes the numbers auditable. | Design principle from the gap report (§4, "Deterministic code, LLM only for language"); no clinical trial of this product exists. | 4 (design argument, not outcome evidence) |
| A small language model phrasing structured metrics JSON into plain language is a bounded, checkable use of a model (numbers verifiable against the JSON in the full tier). | Same design principle; the full-tier verifier in this repo checks summary numbers against metrics JSON in code/stub. | 4 |
| Longitudinal trending (same patient, same setup, repeated captures) is more informative than a single capture for change over time. | Wound/gait longitudinal framing in the gap report (§1, §5.2); applied here by analogy, not proven for this product. | 4 (analogy) |
| The 12-second TUG value is a STEADI screening flag only (see above). Gait-speed 0.8 / 1.0 m/s (Studenski 2011) are interpretive survival references only, not diagnostic cutoffs, and are not applied to this product's leg-lengths-per-second proxy. Varus alignment is associated with OA progression, with no degree threshold cited. The static-test Trendelenburg range 2-8 deg is contested and not gait-validated; a >=4 deg flag is borrowed/app-defined. Every other cutoff in this repo (HKA split, pelvic obliquity, trunk lean, ROM, thrust, asymmetry percent, step width, cadence) is app-defined with no citable source as a screening cutoff. | CDC STEADI; Studenski 2011; static Trendelenburg literature as characterised in the evidence audit (2026-10-23). | 1 for the STEADI flag and Studenski references as stated; all product cutoffs tier 4 (app-defined) until validated |

## (c) What is NOT claimed

- **No validated performance numbers for this product.** Its accuracy on
  real patients is unproven. The eval numbers in README (angle errors,
  TUG errors) are tier-4 synthetic-fixture results: they prove the
  maths is correct on known ground truth, not that the product works
  on patients.
- **No threshold in this repo is a validated cutoff.** Every flag is
  labelled `APP_DEFINED_UNVALIDATED` in the UI, the API and the docs
  (STEADI flag excepted, as a screening flag only).
- **No diagnosis, treatment or risk prediction.** Output language is
  screening-pattern language for clinician review only, enforced by a
  banned-language scan in CI.
- **No live-camera claim in v0.1.0.** On-device MediaPipe extraction
  is a documented integration path, not shipped code (see
  docs/ARCHITECTURE.md).

## Downgrades made during the build

- Frontal-alignment "3-5 deg" split kept only as an app-defined
  amber/red screening split, not as a diagnostic cutoff.
- Gait-speed, cadence, asymmetry and step-width cutoffs kept as
  app-defined screening choices with "no validated cutoff is claimed"
  in every source note, rather than presented as sourced cutoffs.
- Any sentence implying this product detects falls or arthritis was
  removed; the product detects landmark patterns, for review.
- The old sub-two-degree angle target was dropped; synthetic errors
  are reported honestly and the published single-phone ballpark
  (7.16 deg vs Qualisys, n=6) is stated alongside them.
- TUG >=12 s is presented only as a STEADI screening flag, never as
  a risk label; Barry 2014 and Schoene 2013 caveats are stated.
