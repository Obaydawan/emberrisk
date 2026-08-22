# EmberRisk Phase 11 — Monitoring & Drift Detection (Design)

## Objective

Document what a production deployment of EmberRisk would monitor, why,
and what would trigger a model review — without building unnecessary
infrastructure for a project with no live data source (per the Phase 9
scope limitation: the modeling period is fixed to 2018-01-01 through
2025-12-31).

This is deliberately a **design document, not a built monitoring
system**. Building actual dashboards, alerting, or a metrics store for a
system with no live data feed would be infrastructure theater — effort
spent on tooling with nothing real to monitor. What's valuable instead is
demonstrating the thinking: knowing *what* to watch, *why it matters*,
and *what decision each signal would drive* is the actual skill being
assessed here, whether by a recruiter or a scholarship committee.

## Why this matters even though nothing is "live"

A model's TEST performance (Phase 7: F1 0.7172, PR-AUC 0.7985) describes
exactly one thing: how the locked model performed on one fixed, already-
seen historical period. It says nothing about how it would perform on
data generated after that period, under conditions the model never saw
during training. Any real deployment of EmberRisk — extending
`MODELING_END` forward, as flagged as future work in Phase 9 — would need
monitoring precisely because the TEST evaluation's guarantees stop being
valid the moment new data starts arriving.

## What would be monitored

### 1. Feature drift (input distribution monitoring)

Each of the 15 features in `ml.dataset.FEATURE_COLUMNS` has a known
distribution over the TRAIN period (2018-2022). In production, each new
batch of scored data would be compared against that TRAIN-period
distribution:

| Feature group | What could drift | Why it matters |
|---|---|---|
| Fire-history features (`fire_count`, `fire_count_3d/7d/14d/30d`, `days_since_last_detection`) | Sensor/satellite changes, wildfire pattern shifts (e.g. climate-driven changes in fire season length or intensity) | The model learned fire-history patterns specific to 2018-2022; a structurally different fire regime would mean those learned relationships no longer hold |
| FRP features (`frp_mean`, `frp_max`, `frp_mean_7d`, `frp_max_7d`) | Changes in satellite instrument (VIIRS_SNPP_SP is a specific, named source — a source swap would shift FRP value ranges) | FRP imputation (Phase 4) assumes NaN means "no qualifying detection," not "unknown" — a source change could break that assumption silently |
| Weather features (`temperature_max/min`, `relative_humidity`, `precipitation`, `wind_speed`) | Genuine climate shift, or a POWER API schema/methodology change | Phase 4 established these should have 0% missingness — an unexpected rise in missingness would itself be a monitorable signal, not just distributional drift |

**Method (if built):** population stability index (PSI) or a
Kolmogorov-Smirnov test per feature, comparing each new batch's
distribution to the frozen TRAIN-period distribution. A PSI above a
conventional threshold (commonly ~0.2, i.e. "significant shift") on any
feature would flag that feature for review, not automatically retrain.

### 2. Prediction drift (output distribution monitoring)

- **Predicted positive rate**: TEST's predicted positive rate was 19.95%
  (`docs/phase7-test-evaluation.md`), close to the true TEST positive rate
  of 18.99%. A production predicted-positive-rate drifting far from this
  historical baseline — in either direction — would be a first, cheap
  signal that something changed upstream, before needing to know *what*
  changed.
- **Probability distribution shape**: not just the positive rate, but
  whether the model's probability outputs are still spread similarly
  across [0, 1] or have collapsed toward extremes / toward 0.5
  (indicating the model has become under- or over-confident on new data).

### 3. Data freshness / completeness checks

- Already partially built: `processing.validate.run_all_checks()` (Phase
  3) checks dataset-level integrity every time `processing.pipeline.run()`
  executes (wired into Phase 9's `validate_pipeline_output` task as a
  hard gate). In a live deployment, the same category of check — expected
  row counts, expected cell coverage (323 cells), missingness bounds —
  would need to run per-batch, not just at pipeline-build time.
- **NASA POWER latency** (identified during Phase 9 ingestion audit):
  POWER data lags several days behind real-time. A live system would need
  an explicit freshness check — "is data for date X actually available
  yet, or are we about to score on padded/missing recent-day features" —
  before scoring, not just before ingesting.

### 4. Model performance monitoring (the hard one)

This is the category every real ML system struggles with: you cannot
directly measure precision/recall/F1 in production without ground truth,
and for a 7-day-ahead fire prediction, ground truth for a given
prediction isn't known until 7 days later. A production monitoring design
would need:

- **Delayed evaluation**: once 7 days have passed and the true
  `future_fire_7d` label is knowable for a given (cell, date), retroactively
  score the earlier prediction against it. This is structurally identical
  to how `ml.dataset.assemble_feature_label_table()` already drops rows
  where the label isn't yet knowable (Phase 4's end-of-period boundary
  logic) — the same waiting period applies to live monitoring.
- **Rolling window metrics**: F1/precision/recall computed over a trailing
  window (e.g. the last 90 days of now-resolved predictions), compared
  against the Phase 7 TEST baseline (F1 0.7172) as the reference point.

## What would trigger a model review (not automatic retraining)

Consistent with every prior phase's discipline — no automatic action ever
retrains or reselects a model without an explicit, documented decision —
a production version of EmberRisk would define **review triggers**, not
**auto-retrain triggers**:

| Signal | Threshold (example) | Action |
|---|---|---|
| Any single feature's PSI | > 0.2 | Flag feature for manual review; does not halt scoring |
| Predicted positive rate | Outside roughly 15%-25% (± ~5pp from the 19.95% TEST baseline) for 2+ consecutive weeks | Investigate upstream data before trusting predictions |
| Rolling 90-day F1 (once ground truth is available) | Drops more than ~10% relative to the 0.7172 TEST baseline | Triggers a full model review — possibly retraining under a new, explicitly-declared protocol (per Phase 6/7's rule: TEST is closed, any retrain uses a new split) |
| Data freshness check | Missing/incomplete data for the scoring window | Block scoring for that batch rather than score on incomplete features |

The threshold numbers above are illustrative defaults, not tuned or
validated — a real deployment would set them based on business tolerance
for false alarms vs. missed drift, which is outside this project's scope.

## What is explicitly NOT built in this phase

- No dashboard, alerting system, or metrics database — there is no live
  data source to feed one (see Phase 9 scope note).
- No automatic retraining pipeline — retraining remains a manual,
  explicit, documented decision, exactly as established in Phase 6/7.
- No SLA or uptime monitoring for the Phase 10 API — that belongs to a
  deployment/infrastructure phase, not a modeling-monitoring one.

## Relationship to prior phases

This document assumes and builds on:
- Phase 4's feature/target definitions and missingness expectations
- Phase 6/7's locked model, threshold, and TEST baseline numbers
- Phase 9's existing `processing.validate.run_all_checks()` data-quality
  gate, which already implements the "block on integrity failure" pattern
  this document extends conceptually to a live setting
- Phase 9's documented NASA POWER latency finding

## Status

Design complete. No code changes in this phase — this document defines
what a future "Phase 13: live monitoring implementation" would build, and
explicitly depends on first extending `MODELING_END` to accept live data
(flagged as out-of-scope future work since Phase 9).
