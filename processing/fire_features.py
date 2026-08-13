"""
Rolling fire-history features, computed strictly from data with date <= T
(the feature cutoff -- see CRITICAL LEAKAGE RULE). Must be run on a scaffold
that already spans the warm-up window through the modeling period, filled
to 0 for cell-days with no qualifying detections, so that early modeling
dates (e.g. 2018-01-01) have a full 30-day lookback available. Trimming to
the modeling-period-only rows happens AFTER this step, in build_dataset.py.
"""
import pandas as pd

ROLLING_WINDOWS = (3, 7, 14, 30)
DAYS_SINCE_CAP = 90


def fill_scaffold_with_fire_counts(scaffold, firms_cell_day_agg):
    """Left-join the full (cell_id, date) scaffold against the FIRMS
    per-cell-day aggregate, filling missing days with fire_count=0 and
    NaN FRP (no detection that day -- not a missing-data problem)."""
    merged = scaffold.merge(firms_cell_day_agg, on=["cell_id", "date"], how="left")
    merged["fire_count"] = merged["fire_count"].fillna(0).astype(int)
    # frp_mean / frp_max intentionally left as NaN where there's no detection
    return merged


def compute_fire_rolling_features(scaffold_with_counts):
    df = scaffold_with_counts.sort_values(["cell_id", "date"]).copy()

    out_frames = []
    for cell_id, g in df.groupby("cell_id", sort=False):
        g = g.set_index("date")

        for w in ROLLING_WINDOWS:
            # time-based trailing window ending at T inclusive -- strictly
            # backward-looking by construction
            g[f"fire_count_{w}d"] = g["fire_count"].rolling(f"{w}D", min_periods=1).sum()

        g["frp_mean_7d"] = g["frp_mean"].rolling("7D", min_periods=1).mean()
        g["frp_max_7d"] = g["frp_max"].rolling("7D", min_periods=1).max()

        has_fire = g["fire_count"] > 0
        last_fire_date = g.index.to_series().where(has_fire).ffill()
        days_since = (g.index.to_series() - last_fire_date).dt.days
        g["days_since_last_detection"] = (
            days_since.fillna(DAYS_SINCE_CAP).clip(upper=DAYS_SINCE_CAP).astype(int)
        )

        g = g.reset_index()
        g["cell_id"] = cell_id
        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True)
