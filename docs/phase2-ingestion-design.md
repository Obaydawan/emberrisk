# EmberRisk — Phase 2: Ingestion Architecture Design (v2 — decisions locked)

Status: Design LOCKED for items 1-7 below. FIRMS client implemented as first
testable milestone (see section 14). Full historical backfill NOT yet run.
Date: 2026-08-12
Depends on: docs/data-source-study.md (Phase 1)

Changes from v1: modeling period and canonical source are now locked
decisions rather than proposals; grid description corrected to avoid
overstating what it literally is; 30-day fire-history warm-up period added.

---

## 1. Authoritative coverage results (recorded)

| Source | min_date | max_date |
|---|---|---|
| VIIRS_SNPP_SP | 2012-01-20 | 2026-04-27 |
| VIIRS_NOAA20_SP | 2018-04-01 | 2026-05-31 |

Both cover the locked 2018-2025 modeling period. NOAA-20's range is recorded
for future reference but is not used in V1 (see section 6).

## 2. NASA POWER coverage for 2018-2025 (California grid)

Literature-level confidence, not yet independently verified per-cell (see
Phase 2 v1 section 2 for the reasoning). The first live POWER ingestion call
(one cell, full range) doubles as the empirical check: it must return exactly
2,922 rows for the modeling period, or 2,952 if warm-up is extended to
POWER as well (currently it is not — see section 3).

## 3. LOCKED temporal and spatial boundaries + fire-history warm-up

```
MODELING PERIOD (locked):
  2018-01-01 through 2025-12-31 (inclusive)
  8 complete calendar years, 2,922 days
  No partial/incomplete 2026 data included.

FIRE-HISTORY WARM-UP PERIOD (new):
  2017-12-02 through 2017-12-31 (30 days immediately before the modeling
  period)
  Purpose: rolling fire-history features (fire_count_30d, days_since_last_
  detection, etc. — see section 9) need up to 30 days of prior detection
  data to be computable for cell-days at the very start of the modeling
  period (e.g. 2018-01-01's fire_count_30d needs detections back to
  2017-12-02). Without warm-up, the first ~30 days of 2018 would have
  artificially truncated/incorrect rolling features.

  IMPORTANT: warm-up data is ingested (raw FIRMS detections only) but is
  NOT part of the modeling dataset itself — no cell-day rows with T in the
  warm-up window get a label or become a training/val/test example. It
  exists purely to feed rolling-feature computation for early 2018 dates.

  FIRMS raw ingestion range therefore becomes:
    2017-12-02 through 2025-12-31 = 2,952 days total

  Open question flagged for Phase 5 (feature engineering), not resolved
  here: POWER-derived rolling weather features (if any are added, e.g.
  7-day precipitation sums) would have the same edge-effect problem at the
  start of the modeling period. Decision on whether POWER needs a matching
  warm-up is deferred until Phase 5 defines the actual weather rolling
  features.

SPATIAL RANGE (locked, unchanged from Phase 1):
  California bounding box (west, south, east, north)
  (-124.5, 32.5, -114.0, 42.0)
```

## 4. Handling FIRMS SP processing lag

Unchanged from v1: not a blocker for this historical backfill (SP coverage
already extends past 2025-12-31). Relevant only to future production/
incremental ingestion (deferred to Phase 9).

## 5. Canonical grid definition (corrected description)

```
Grid spacing:   0.5 deg latitude x 0.625 deg longitude
Bounding box:   (-124.5, 32.5, -114.0, 42.0)
Cell count:     323
Cell ID scheme: "{lat_index}_{lon_index}"
                lat_index = floor(latitude / 0.5)
                lon_index = floor(longitude / 0.625)
Cell centroid:  centroid_lat = cell_south_edge + 0.25
                centroid_lon = cell_west_edge + 0.3125
```

**Correction from v1:** this is described as the **EmberRisk canonical
grid**, deliberately chosen to *match POWER's native meteorological
resolution* (0.5 x 0.625 deg is POWER's documented grid spacing). It is not
claimed to be literally identical to POWER's internal grid cell boundaries/
alignment (POWER's actual grid origin/offset is an internal implementation
detail we haven't verified bit-for-bit). The distinction matters: EmberRisk
defines its own grid at the same resolution for defensibility, rather than
asserting exact alignment with an unverified internal boundary.

## 6. Fire target (formal definition) — VIIRS_SNPP_SP only in V1

```
Unit:   grid cell c (of the 323-cell canonical grid)
Grain:  daily
Label:  1 if cell c has >=1 qualifying VIIRS fire detection
           (confidence != "low") during [T+1, T+H]
        0 otherwise
Given:  features computed only from data with acq_date <= T
Source: VIIRS_SNPP_SP ONLY for V1.
```

**Locked decision:** `VIIRS_NOAA20_SP` is explicitly NOT combined into V1
ingestion, target construction, or features. Combining two satellites
requires resolving detection deduplication (the same fire observed by both
satellites) and confidence-scale reconciliation — real complexity that
isn't justified until the single-satellite V1 model is working end-to-end.
NOAA-20 remains a documented candidate enhancement, not a current input.

## 7. Horizon (H) — still candidates, 7 leading

Unchanged: 3, 7, 14 days remain candidates, 7 is the current leading choice.
Final selection happens in Phase 6 after the complete 2018-2025 dataset
(with warm-up-enabled features) is constructed — the Phase 1 sample numbers
are not the deciding evidence, just directional support.

## 8. Leakage-safe temporal dataset design

Unchanged from v1:

```
Feature cutoff:  day T (inclusive)
Label window:    T+1 through T+H (exclusive of T), used only for the target
```

Chronological split (draft, finalized in Phase 6):
```
TRAIN:      2018-01-01 to 2022-12-31  (5 years)
VALIDATION: 2023-01-01 to 2023-12-31  (1 year)
TEST:       2024-01-01 to 2025-12-31  (2 years)
```

## 9. Rolling fire-history features (design, now warm-up-aware)

Unchanged feature list from v1 (fire_count_3d/7d/14d/30d, days_since_last_
detection, frp_mean_7d/frp_max_7d), but now computable for the *entire*
modeling period including 2018-01-01, because the 30-day warm-up window
(section 3) supplies the necessary lookback data. Without the warm-up
addition, the first 30 days of the modeling period would have had
artificially incomplete rolling windows.

## 10. POWER-to-grid mapping (design)

Unchanged from v1: one POWER API call per grid-cell centroid, full date
range (2018-01-01 to 2025-12-31 — POWER is not extended with the fire
warm-up window, per the open question noted in section 3) in a single call.
323 total POWER calls for the full historical backfill.

## 11. Estimated API calls, storage, runtime, memory (updated for warm-up)

**FIRMS** (bbox-wide calls, 5-day chunks, now covering 2,952 days with warm-up):
```
2,952 days / 5-day chunks = 591 API calls (up from 585 without warm-up)
Raw row volume estimate: unchanged planning range, ~440,000-730,000 rows
  over the full ingestion window (rough estimate, not yet measured)
Storage: still well under 200 MB
```

**NASA POWER** (one call per cell, unchanged — no warm-up extension):
```
323 API calls, 943,806 grid-cell-day rows, under 100 MB as Parquet
```

**Runtime:** FIRMS ~591 calls at ~1-2s pacing -> roughly 15-25 minutes.
POWER unchanged, ~10-15 minutes. Total still under an hour.

**Memory:** unchanged — comfortably fits in-memory on 8 GB RAM.

## 12. Resumability design

Unchanged from v1: manifest-based (chunk id -> status/output path/timestamp),
idempotent per-chunk output files, complete only marked after successful
disk write. This is now IMPLEMENTED for FIRMS (section 14), not just
designed.

## 13. Locked-decisions log

| # | Decision | Status |
|---|---|---|
| 1 | Modeling period = 2018-01-01 to 2025-12-31 | LOCKED |
| 2 | Canonical V1 fire source = VIIRS_SNPP_SP only | LOCKED |
| 3 | Grid described as EmberRisk canonical grid (POWER-resolution-matched, not POWER-boundary-identical) | LOCKED |
| 4 | 30-day fire-history warm-up period added (2017-12-02 to 2017-12-31) | LOCKED |
| 5 | Horizons 3/7/14 remain candidates, 7 leading | UNCHANGED, still open |
| 6 | Leakage-safe cutoff + chronological split | LOCKED (split dates still draft) |
| 7 | Manifest-based resumability | LOCKED, now implemented for FIRMS |

## 14. Current implementation status

**Implemented:** `ingestion/common/manifest.py`, `ingestion/firms/client.py`,
`ingestion/firms/ingest.py` — a testable FIRMS ingestion client with
chunking, retry/backoff, manifest-based resumability, and idempotent
per-chunk Parquet output.

**Verified so far (this session, no network required):** chunk-generation
logic, manifest state transitions, and client retry/backoff behavior, via
mocked unit tests (network calls simulated, not real).

**NOT yet verified:** actual live FIRMS API behavior (real schema, real
confidence values, real dates returned) — requires running on the local
machine with the real `FIRMS_MAP_KEY` and real network access. This is the
next step, on a tiny date range only (2018-01-01 to 2018-01-05), NOT the
full 591-chunk backfill.
