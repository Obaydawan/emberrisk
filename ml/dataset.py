"""
ml/dataset.py -- Phase 4 Step 2: feature/label assembly.

Turns the Phase 3 cell-day dataset (data/processed/cell_day_dataset.parquet)
plus the chosen target table into a single feature+label table ready for
the chronological split (Phase 4 Step 3 -- NOT implemented here).

Decisions applied here were made in Phase 4 EDA, not re-decided by this
module:
  - Primary target: future_fire_7d (from data/processed/targets_h7.parquet).
    The 3d/14d target tables are intentionally NOT touched here -- they stay
    in their own files for later comparison, per the EDA decision to keep
    but not model them now.
  - FRP columns (frp_mean, frp_max, frp_mean_7d, frp_max_7d) get NaN filled
    with 0, because NaN there means "no qualifying detection that day/
    window", not "unknown value" -- distinct from POWER's sentinel-derived
    NaN, which means a genuinely missing measurement.
  - Weather features are expected to have 0% missingness (confirmed in
    Phase 3 validation). This module VERIFIES that expectation still holds
    on the real data rather than silently assuming it.
  - Rows with a null target label (Phase 3's documented end-of-period
    boundary -- the final H days of the modeling period can't have a fully
    observed future window) are dropped here, with the exact count logged,
    never silently.
"""
import logging

import pandas as pd

logger = logging.getLogger("emberrisk.ml.dataset")

ID_COLUMNS = ["cell_id", "date"]

FRP_COLUMNS = ["frp_mean", "frp_max", "frp_mean_7d", "frp_max_7d"]

FIRE_HISTORY_COLUMNS = [
    "fire_count", "fire_count_3d", "fire_count_7d", "fire_count_14d",
    "fire_count_30d", "days_since_last_detection",
] + FRP_COLUMNS

WEATHER_COLUMNS = [
    "temperature_max", "temperature_min", "relative_humidity",
    "precipitation", "wind_speed",
]

FEATURE_COLUMNS = FIRE_HISTORY_COLUMNS + WEATHER_COLUMNS

PRIMARY_TARGET_COLUMN = "future_fire_7d"
PRIMARY_TARGET_HORIZON = 7

DEFAULT_DATASET_PATH = "data/processed/cell_day_dataset.parquet"
DEFAULT_TARGET_PATH = "data/processed/targets_h7.parquet"


def load_cell_day_dataset(path=DEFAULT_DATASET_PATH):
    return pd.read_parquet(path)


def load_target(path=DEFAULT_TARGET_PATH):
    return pd.read_parquet(path)


def impute_frp(df):
    """Fill FRP NaNs with 0 -- NaN there means no qualifying detection that
    day/window, not an unknown measurement. Only touches the FRP columns;
    every other column is left exactly as it arrived."""
    df = df.copy()
    for col in FRP_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Expected FRP column missing from dataset: {col!r}")
        df[col] = df[col].fillna(0)
    return df


def verify_no_unexpected_missingness(df, columns):
    """Raises if any of the given columns have missing values. The Phase 4
    EDA established these should be 0%-missing after FRP imputation -- a
    silent NaN slipping through here would corrupt model training rather
    than fail loudly where the problem is easy to diagnose."""
    missing_counts = {c: int(df[c].isna().sum()) for c in columns if c in df.columns and df[c].isna().any()}
    if missing_counts:
        raise ValueError(
            f"Unexpected missingness found after imputation (expected 0 for "
            f"all feature columns): {missing_counts}"
        )


def assemble_feature_label_table(
    cell_day_dataset=None,
    target_df=None,
    target_column=PRIMARY_TARGET_COLUMN,
):
    """Joins features with the chosen target, imputes FRP, verifies no
    unexpected missingness remains, and drops end-of-period null-labeled
    rows (count logged, never silent). Returns (table, report_dict)."""
    if cell_day_dataset is None:
        cell_day_dataset = load_cell_day_dataset()
    if target_df is None:
        target_df = load_target()

    df = cell_day_dataset.copy()

    missing_feature_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_feature_cols:
        raise ValueError(
            f"Expected feature columns missing from cell_day_dataset: {missing_feature_cols}"
        )

    df = impute_frp(df)

    if target_column not in target_df.columns:
        raise ValueError(f"Target column {target_column!r} not found in target table")

    merged = df.merge(
        target_df[ID_COLUMNS + [target_column]],
        on=ID_COLUMNS,
        how="left",
    )

    n_before = len(merged)
    null_label_mask = merged[target_column].isna()
    n_null = int(null_label_mask.sum())
    if n_null:
        logger.info(
            "Dropping %d rows with a null %s label (Phase 3 end-of-period "
            "boundary) out of %d total rows", n_null, target_column, n_before,
        )
    merged = merged[~null_label_mask].copy()
    merged[target_column] = merged[target_column].astype(int)

    verify_no_unexpected_missingness(merged, FEATURE_COLUMNS)

    report = {
        "n_before_label_drop": n_before,
        "n_dropped_null_label": n_null,
        "n_after": len(merged),
        "target_column": target_column,
        "feature_columns": list(FEATURE_COLUMNS),
        "positive_rate": round(float(merged[target_column].mean()), 4) if len(merged) else None,
    }
    return merged, report


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    table, report = assemble_feature_label_table()
    logger.info("Assembled feature/label table: %s", report)
    print(json.dumps(report, indent=2))
