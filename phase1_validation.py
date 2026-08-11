#!/usr/bin/env python3
"""
EmberRisk Phase 1 -- Empirical validation script (THROWAWAY, NOT production ingestion)

*** THIS PRODUCES A FEASIBILITY SAMPLE, NOT THE FINAL CALIFORNIA CLASS BALANCE. ***
It covers two short representative windows (one peak-season, one shoulder-season)
and a handful of POWER sample points. Full-scale numbers will only be known after
Phase 2 ingests the complete VIIRS SP historical archive and full POWER grid
coverage across all cells and years. Every printed/reported number in this script
is explicitly labeled as sample-derived for that reason.

Purpose:
  Pull a small, representative sample of FIRMS VIIRS SP detections and NASA POWER
  daily weather data for a California bounding box, across two sample windows
  (peak fire season + shoulder season), and report:

    - total / positive / negative cell-days (including zero-detection cells,
      verified via an explicit count assertion)
    - positive percentage per horizon (3 / 7 / 14 days)
    - number of unique grid cells with at least one detection
    - raw FIRMS detection counts, broken down by confidence category
    - exact dates covered by the pulled data
    - NASA POWER missingness per parameter
    - an approximate full-scale data volume estimate (clearly marked as approximate)

Grid definition:
  Uses POWER's actual native meteorological resolution: 0.5 deg latitude x
  0.625 deg longitude (per NASA POWER documentation / nasapower package docs).
  This is NOT a rounded/approximated grid -- it is POWER's real grid spacing,
  so FIRMS detections are aggregated into POWER's true cell boundaries.

Usage:
  1. FIRMS_MAP_KEY must be set in your .env (already confirmed working).
  2. pip install requests pandas python-dotenv   (if not already installed)
  3. python phase1_validation.py
  4. Paste the full contents of
     phase1_validation_output/phase1_validation_summary.json
     back into the EmberRisk chat for interpretation.

This is intentionally simple and not resilient/production-grade. It exists only
to answer the Phase 1 feasibility question empirically, then it can be discarded
or archived under docs/ -- it is not part of the ingestion package built in Phase 2.
"""

import os
import time
import json
from io import StringIO
from datetime import date, timedelta
from collections import defaultdict

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
MAP_KEY = os.getenv("FIRMS_MAP_KEY")
if not MAP_KEY:
    raise SystemExit("FIRMS_MAP_KEY not found in environment (.env)")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# California bounding box (west, south, east, north) with small margin
CA_BBOX = (-124.5, 32.5, -114.0, 42.0)

# POWER's actual native meteorological grid spacing -- not an approximation.
GRID_SIZE_LAT_DEG = 0.5
GRID_SIZE_LON_DEG = 0.625

FIRMS_SOURCE = "VIIRS_SNPP_SP"  # science-quality historical product

# Two representative sample windows: peak fire season + shoulder season.
# These are SAMPLE windows for feasibility testing, not the full historical range.
SAMPLE_WINDOWS = [
    ("peak_season_2023", date(2023, 8, 15), date(2023, 9, 28)),       # 45 days
    ("shoulder_season_2023", date(2023, 11, 1), date(2023, 11, 30)),  # 30 days
]

HORIZONS = [3, 7, 14]

# A handful of representative POWER sample points across CA (not full grid coverage)
POWER_SAMPLE_POINTS = {
    "coastal_LA": (34.05, -118.25),
    "central_valley_fresno": (36.75, -119.77),
    "sierra_foothills": (39.15, -120.95),
    "socal_desert": (33.75, -116.35),
    "north_coast": (40.80, -124.10),
}

POWER_PARAMS = ["T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M"]
POWER_COMMUNITY = "AG"

OUT_DIR = "phase1_validation_output"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# FIRMS PULL (chunked into <=10-day requests, per API day_range limit)
# ---------------------------------------------------------------------------

def firms_area_url(bbox, day_range, start_date):
    w, s, e, n = bbox
    area = f"{w},{s},{e},{n}"
    return (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{MAP_KEY}/{FIRMS_SOURCE}/{area}/{day_range}/{start_date.isoformat()}"
    )


def pull_firms_window(bbox, start_d, end_d):
    """Pull FIRMS detections for [start_d, end_d] inclusive, chunked in <=10-day calls."""
    all_frames = []
    cursor = start_d
    while cursor <= end_d:
        remaining = (end_d - cursor).days + 1
        day_range = min(5, remaining)
        url = firms_area_url(bbox, day_range, cursor)
        print(f"  FIRMS request: {cursor} .. {cursor + timedelta(days=day_range - 1)} "
              f"(day_range={day_range})")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        text = resp.text
        if text.startswith("Invalid") or "error" in text[:200].lower():
            raise RuntimeError(f"FIRMS API error: {text[:300]}")
        df = pd.read_csv(StringIO(text))
        all_frames.append(df)
        cursor += timedelta(days=day_range)
        time.sleep(1)  # polite pacing, well under the 5000/10min limit
    if not all_frames:
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "confidence"])
    return pd.concat(all_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# POWER PULL (single call per point per window, all params at once)
# ---------------------------------------------------------------------------

def pull_power_point(lat, lon, start_d, end_d):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": ",".join(POWER_PARAMS),
        "community": POWER_COMMUNITY,
        "latitude": lat,
        "longitude": lon,
        "start": start_d.strftime("%Y%m%d"),
        "end": end_d.strftime("%Y%m%d"),
        "format": "JSON",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "properties" not in data:
        raise RuntimeError(f"POWER API error for ({lat},{lon}): {data}")
    param_data = data["properties"]["parameter"]
    df = pd.DataFrame(param_data)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# GRID ASSIGNMENT + LABELING
# ---------------------------------------------------------------------------

def assign_grid_cell(lat, lon, grid_lat=GRID_SIZE_LAT_DEG, grid_lon=GRID_SIZE_LON_DEG):
    cell_lat = int(lat // grid_lat)
    cell_lon = int(lon // grid_lon)
    return f"{cell_lat}_{cell_lon}"


def enumerate_grid_cells(bbox, grid_lat=GRID_SIZE_LAT_DEG, grid_lon=GRID_SIZE_LON_DEG):
    """Enumerate EVERY cell in the bbox grid, regardless of whether it ever
    has a fire detection. This is what guarantees zero-detection cells are
    included as negatives rather than silently omitted."""
    w, s, e, n = bbox
    cells = []
    lat = s
    while lat < n:
        lon = w
        while lon < e:
            cells.append(assign_grid_cell(lat + grid_lat / 2, lon + grid_lon / 2, grid_lat, grid_lon))
            lon += grid_lon
        lat += grid_lat
    return cells


def build_label_matrix(detections_df, bbox, start_d, end_d, horizons):
    """
    For each grid cell (including cells with zero detections in the whole
    window) and each day T where the full horizon window is available inside
    [start_d, end_d], compute binary labels for each horizon.

    All horizons share the SAME set of evaluation days (trimmed by the
    longest horizon) so that differences in positive rate across horizons
    reflect the horizon length itself, not a different number of days
    evaluated.
    """
    all_cells = enumerate_grid_cells(bbox)

    cell_dates = defaultdict(set)
    confidence_counts = {}
    if len(detections_df):
        df = detections_df.copy()
        if "confidence" in df.columns:
            confidence_counts = df["confidence"].astype(str).value_counts().to_dict()
        df["cell"] = df.apply(lambda r: assign_grid_cell(r["latitude"], r["longitude"]), axis=1)
        df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.date

        # VIIRS confidence is categorical l/n/h -- drop low-confidence detections
        # for labeling purposes. (If confidence is numeric instead, this comparison
        # is a no-op and keeps all rows -- confidence_counts above still reports
        # the raw breakdown either way.)
        if "confidence" in df.columns:
            mask = df["confidence"].astype(str).str.lower() != "l"
            df = df[mask]

        for _, row in df.iterrows():
            cell_dates[row["cell"]].add(row["acq_date"])

    results = {h: {"pos": 0, "neg": 0} for h in horizons}
    max_horizon = max(horizons)
    day = start_d
    usable_end = end_d - timedelta(days=max_horizon)  # full horizon must fit inside sample window
    n_usable_days = max(0, (usable_end - start_d).days + 1)

    day_cursor = start_d
    while day_cursor <= usable_end:
        for cell in all_cells:
            dates_with_fire = cell_dates.get(cell, set())  # empty set -> correctly negative
            for h in horizons:
                window = [day_cursor + timedelta(days=i) for i in range(1, h + 1)]
                label = int(any(d in dates_with_fire for d in window))
                results[h]["pos" if label else "neg"] += 1
        day_cursor += timedelta(days=1)

    # Explicit sanity check: every horizon must cover the identical total
    # number of cell-days (n_total_cells * n_usable_days), since only the
    # pos/neg split should differ between horizons, not the denominator.
    expected_total = len(all_cells) * n_usable_days
    for h in horizons:
        actual_total = results[h]["pos"] + results[h]["neg"]
        assert actual_total == expected_total, (
            f"Cell-day accounting mismatch for horizon={h}: "
            f"expected {expected_total}, got {actual_total}"
        )

    return results, len(all_cells), n_usable_days, confidence_counts


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    summary = {
        "IMPORTANT_NOTE": (
            "This is a Phase 1 FEASIBILITY SAMPLE covering two short representative "
            "windows and five POWER sample points. It is NOT the final California "
            "class balance or the final data volume. Full-scale numbers require "
            "Phase 2 ingestion of the complete VIIRS SP historical archive and full "
            "POWER grid coverage across all cells and years."
        ),
        "windows": {},
        "config": {
            "bbox": CA_BBOX,
            "grid_lat_deg": GRID_SIZE_LAT_DEG,
            "grid_lon_deg": GRID_SIZE_LON_DEG,
            "grid_note": "0.5 x 0.625 deg matches POWER's actual native meteorological grid spacing",
            "firms_source": FIRMS_SOURCE,
            "horizons": HORIZONS,
        },
    }

    for wlabel, start_d, end_d in SAMPLE_WINDOWS:
        print(f"\n=== Window: {wlabel} ({start_d} to {end_d}) ===")

        print("Pulling FIRMS detections...")
        det_df = pull_firms_window(CA_BBOX, start_d, end_d)
        det_df.to_csv(f"{OUT_DIR}/firms_raw_{wlabel}.csv", index=False)
        n_detections = len(det_df)

        if n_detections:
            dates_in_data = pd.to_datetime(det_df["acq_date"]).dt.date
            dates_covered = [str(dates_in_data.min()), str(dates_in_data.max())]
            n_cells_with_fire = det_df.apply(
                lambda r: assign_grid_cell(r["latitude"], r["longitude"]), axis=1
            ).nunique()
        else:
            dates_covered = [None, None]
            n_cells_with_fire = 0
        print(f"  -> {n_detections} raw detections across {n_cells_with_fire} distinct grid cells")
        print(f"  -> dates actually covered in returned data: {dates_covered}")

        print("Computing class balance for horizons", HORIZONS, "(including zero-detection cells)...")
        label_results, n_total_cells, n_usable_days, confidence_counts = build_label_matrix(
            det_df, CA_BBOX, start_d, end_d, HORIZONS
        )
        print(f"  Sanity check passed: {n_total_cells} total grid cells x {n_usable_days} "
              f"usable days = {n_total_cells * n_usable_days} cell-days per horizon")

        window_summary = {
            "is_feasibility_sample": True,
            "configured_date_range": [str(start_d), str(end_d)],
            "n_days_configured": (end_d - start_d).days + 1,
            "dates_actually_covered_in_firms_response": dates_covered,
            "firms": {
                "n_raw_detections": n_detections,
                "detections_by_confidence_category": confidence_counts,
                "n_unique_cells_with_at_least_one_detection": n_cells_with_fire,
            },
            "grid": {
                "n_total_cells_in_bbox": n_total_cells,
                "n_usable_days_for_horizon_labeling": n_usable_days,
            },
            "horizons": {},
        }
        for h in HORIZONS:
            pos = label_results[h]["pos"]
            neg = label_results[h]["neg"]
            total = pos + neg
            pct = round(100 * pos / total, 3) if total else None
            window_summary["horizons"][h] = {
                "total_cell_days": total,
                "positive_cell_days": pos,
                "negative_cell_days": neg,
                "positive_percentage": pct,
            }
            print(f"  horizon={h}d  total_cell_days={total}  positive={pos} ({pct}%)  negative={neg}")

        summary["windows"][wlabel] = window_summary

        print("Pulling POWER samples for missingness check...")
        power_missing = {}
        for name, (lat, lon) in POWER_SAMPLE_POINTS.items():
            df = pull_power_point(lat, lon, start_d, end_d)
            df.to_csv(f"{OUT_DIR}/power_raw_{wlabel}_{name}.csv")
            missing_pct = {}
            for col in df.columns:
                # POWER's fill value for missing/unavailable data is typically -999
                n_missing = (df[col] <= -900).sum()
                missing_pct[col] = round(100 * n_missing / len(df), 2) if len(df) else None
            power_missing[name] = missing_pct
            time.sleep(1)
        window_summary["power_missingness_pct_by_point_and_param"] = power_missing
        print("  POWER missingness (%):", json.dumps(power_missing, indent=2))

    # Approximate full-scale volume estimate -- explicitly labeled as approximate.
    n_cells = len(enumerate_grid_cells(CA_BBOX))
    summary["approximate_full_scale_volume_estimate"] = {
        "note": (
            "APPROXIMATE ONLY. Based on the sample bbox cell count and a naive "
            "365-days/year multiplier; does not account for missing satellite "
            "coverage days, leap years, or FIRMS SP reprocessing lag."
        ),
        "n_grid_cells_in_bbox": n_cells,
    }
    for years in [5, 8]:
        rows = n_cells * 365 * years
        print(f"\nApprox full-scale volume estimate: {n_cells} cells x 365 days x {years} years "
              f"= {rows:,} grid-cell-day rows (APPROXIMATE)")
        summary["approximate_full_scale_volume_estimate"][f"{years}yr_rows_approx"] = rows

    out_path = f"{OUT_DIR}/phase1_validation_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Full summary written to: {out_path}")
    print(">>> Paste the FULL CONTENTS of that file back into the EmberRisk chat for interpretation. <<<")


if __name__ == "__main__":
    main()
