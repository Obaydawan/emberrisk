"""
POWER standardization: raw per-cell-per-window Parquet files ->
per-cell-day weather table with standardized, human-readable column names.

Locked raw parameters (not renamed at the SOURCE, only at the standardized
output stage): T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR, WS2M.
"""
import glob
import re

import pandas as pd

from ingestion.power.client import MISSING_VALUE_THRESHOLD

RAW_TO_STANDARD_NAMES = {
    "T2M_MAX": "temperature_max",
    "T2M_MIN": "temperature_min",
    "RH2M": "relative_humidity",
    "PRECTOTCORR": "precipitation",
    "WS2M": "wind_speed",
}

STANDARD_COLUMNS = list(RAW_TO_STANDARD_NAMES.values())

# ingestion/power/ingest.py writes files as
# power_{cell_id}_{start_iso}_{end_iso}.parquet -- cell_id itself may
# contain underscores/hyphens (e.g. "9_-199"), so anchor on the two
# fixed-format trailing date fields rather than splitting on "_" naively.
_FNAME_RE = re.compile(
    r"^power_(?P<cell_id>.+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})\.parquet$"
)


def _cell_id_from_filename(path):
    name = path.split("/")[-1]
    m = _FNAME_RE.match(name)
    if not m:
        raise ValueError(f"Could not parse cell_id from POWER filename: {name!r}")
    return m.group("cell_id")


def load_raw_power(raw_dir="data/raw/power"):
    files = sorted(glob.glob(f"{raw_dir}/power_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No POWER raw parquet files found under {raw_dir}")

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df["cell_id"] = _cell_id_from_filename(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def standardize_power(raw_df):
    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    missing_source_cols = [c for c in RAW_TO_STANDARD_NAMES if c not in df.columns]
    if missing_source_cols:
        raise ValueError(
            f"Expected POWER source columns missing before rename: {missing_source_cols}. "
            "Inspect the real POWER schema before proceeding -- do not silently substitute."
        )

    # Convert POWER's documented fill/sentinel values (-999) to proper NaN
    # BEFORE rename/output. Reuses MISSING_VALUE_THRESHOLD (-900) from
    # ingestion/power/client.py -- the single source of truth already
    # established and tested in Phase 2 -- rather than defining a second,
    # possibly-drifting threshold here.
    source_cols = list(RAW_TO_STANDARD_NAMES.keys())
    df[source_cols] = df[source_cols].where(df[source_cols] > MISSING_VALUE_THRESHOLD)

    df = df.rename(columns=RAW_TO_STANDARD_NAMES)
    keep = ["cell_id", "date"] + list(RAW_TO_STANDARD_NAMES.values())
    return df[keep]
