"""
FIRMS standardization: raw satellite detections -> clean per-cell-day
aggregates, ready for rolling feature computation.

Applies, in order:
  1. Canonical cell assignment via ingestion.common.grid.cell_id_for
     (rows outside the 323-cell canonical grid -- e.g. observed 80_-200,
     84_-198 -- are dropped and counted, NOT merged into a neighboring
     cell or silently kept).
  2. Confidence filter: drop low-confidence ('l') detections, per the
     locked target definition (docs/phase2-ingestion-design.md section 6).
     Only the two KNOWN non-low VIIRS confidence values ('n' nominal, 'h'
     high) qualify. Missing or unrecognized confidence values do NOT
     silently qualify -- they are treated as not-qualifying and counted,
     since a naive "!= low" check would otherwise let NaN/malformed values
     through by accident.
  3. Aggregation to one row per (cell_id, date) with a qualifying detection
     count and FRP statistics.
"""
import glob
import logging

import pandas as pd

from ingestion.common.grid import cell_id_for, enumerate_grid_cells, CA_BBOX

logger = logging.getLogger("emberrisk.processing.firms")

# VIIRS confidence is categorical: low / nominal / high (see Phase 1/2 docs).
KNOWN_CONFIDENCE_VALUES = {"l", "n", "h"}
QUALIFYING_CONFIDENCE_VALUES = {"n", "h"}


def load_raw_firms(raw_dir="data/raw/firms"):
    files = sorted(glob.glob(f"{raw_dir}/firms_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No FIRMS raw parquet files found under {raw_dir}")
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def standardize_firms(raw_df, bbox=CA_BBOX):
    """Assign canonical cells, drop out-of-grid rows, flag qualifying
    (non-low-confidence) detections. Returns the standardized row-level
    DataFrame -- aggregation happens separately in aggregate_to_cell_day."""
    df = raw_df.copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"])

    canonical_cell_ids = {c.cell_id for c in enumerate_grid_cells(bbox)}

    df["cell_id"] = [
        cell_id_for(lat, lon) for lat, lon in zip(df["latitude"], df["longitude"])
    ]

    outside_mask = ~df["cell_id"].isin(canonical_cell_ids)
    n_outside = int(outside_mask.sum())
    if n_outside:
        logger.warning(
            "%d raw FIRMS rows fall outside the canonical %d-cell grid and "
            "are excluded (not merged into a neighboring cell): %s",
            n_outside, len(canonical_cell_ids),
            df.loc[outside_mask, "cell_id"].value_counts().to_dict(),
        )
    df = df[~outside_mask].copy()

    conf_lower = df["confidence"].astype(str).str.lower()
    is_known = conf_lower.isin(KNOWN_CONFIDENCE_VALUES)
    n_unknown = int((~is_known).sum())
    if n_unknown:
        logger.warning(
            "%d FIRMS rows have missing/unrecognized confidence values and "
            "are treated as NOT qualifying (never default to qualifying): %s",
            n_unknown, df.loc[~is_known, "confidence"].value_counts(dropna=False).to_dict(),
        )
    df["qualifying"] = conf_lower.isin(QUALIFYING_CONFIDENCE_VALUES)  # only known 'n'/'h'

    return df


def aggregate_to_cell_day(standardized_df):
    """One row per (cell_id, date) using QUALIFYING detections only. Days
    with zero qualifying detections simply have no row here -- callers must
    left-join against the full scaffold and fill 0, not assume every
    (cell_id, date) is present."""
    qualifying = standardized_df[standardized_df["qualifying"]]

    agg = (
        qualifying.groupby(["cell_id", "acq_date"])
        .agg(
            fire_count=("acq_date", "size"),
            frp_mean=("frp", "mean"),
            frp_max=("frp", "max"),
        )
        .reset_index()
        .rename(columns={"acq_date": "date"})
    )
    return agg
