"""
ml/split.py -- Phase 4 Step 3: chronological train/validation/test split.

Locked boundaries (confirmed here, not decided by this module):

  TRAIN:      2018-01-01 to 2022-12-31
  VALIDATION: 2023-01-01 to 2023-12-31
  TEST:       2024-01-01 to 2025-12-31

These three ranges are contiguous, non-overlapping, and together span the
entire locked modeling period (2018-01-01 to 2025-12-31) exactly -- so every
row in a correctly-built modeling dataset should land in exactly one split.

No randomness anywhere in this module: splitting is a pure function of the
`date` column. This is the ONE split function every future model (baseline,
XGBoost, whatever comes later) must call -- it should never be
reimplemented per-script.
"""
import pandas as pd

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2022-12-31")

VALIDATION_START = pd.Timestamp("2023-01-01")
VALIDATION_END = pd.Timestamp("2023-12-31")

TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-12-31")


def _assign_split(date_series):
    dates = pd.to_datetime(date_series)
    split = pd.Series(index=dates.index, dtype="object")
    split[(dates >= TRAIN_START) & (dates <= TRAIN_END)] = "train"
    split[(dates >= VALIDATION_START) & (dates <= VALIDATION_END)] = "validation"
    split[(dates >= TEST_START) & (dates <= TEST_END)] = "test"
    return split


def chronological_split(df, date_col="date"):
    """
    Returns (train_df, validation_df, test_df) -- a pure date-based
    partition, no randomness anywhere. Rows whose date falls outside all
    three locked ranges raise ValueError rather than being silently
    dropped or misassigned to the nearest split.
    """
    if date_col not in df.columns:
        raise ValueError(f"date column {date_col!r} not found in DataFrame")

    split_labels = _assign_split(df[date_col])
    unassigned = split_labels.isna()
    if unassigned.any():
        bad_dates = pd.to_datetime(df.loc[unassigned, date_col])
        raise ValueError(
            f"{int(unassigned.sum())} row(s) have dates outside all three "
            f"locked split ranges (min={bad_dates.min()}, max={bad_dates.max()}). "
            "Fix the input data or the split boundaries -- these rows must "
            "not be silently dropped or assigned to the nearest split."
        )

    train_df = df[split_labels == "train"].copy()
    validation_df = df[split_labels == "validation"].copy()
    test_df = df[split_labels == "test"].copy()

    return train_df, validation_df, test_df


def validate_split_partition(df, train_df, validation_df, test_df, date_col="date"):
    """
    Confirms every row in df belongs to EXACTLY one of the three splits:
      - coverage: every row is accounted for (no row missing from all three)
      - no_overlap: no row appears in more than one split (via index identity)
      - date_order_ok: train dates < validation dates < test dates, with no
        range overlap -- the structural leakage guarantee

    Returns (passed: bool, detail: dict) rather than raising, so a caller
    can log/report the full picture even when something's wrong.
    """
    n_total = len(df)
    n_train, n_val, n_test = len(train_df), len(validation_df), len(test_df)
    n_sum = n_train + n_val + n_test
    coverage_ok = n_sum == n_total

    train_idx = set(train_df.index)
    val_idx = set(validation_df.index)
    test_idx = set(test_df.index)
    overlap_train_val = train_idx & val_idx
    overlap_train_test = train_idx & test_idx
    overlap_val_test = val_idx & test_idx
    no_overlap = not (overlap_train_val or overlap_train_test or overlap_val_test)

    date_order_ok = True
    if n_train and n_val:
        date_order_ok &= bool(train_df[date_col].max() < validation_df[date_col].min())
    if n_val and n_test:
        date_order_ok &= bool(validation_df[date_col].max() < test_df[date_col].min())
    if n_train and n_test:
        date_order_ok &= bool(train_df[date_col].max() < test_df[date_col].min())

    passed = coverage_ok and no_overlap and date_order_ok

    detail = {
        "n_total": n_total,
        "n_train": n_train,
        "n_validation": n_val,
        "n_test": n_test,
        "n_sum": n_sum,
        "coverage_ok": coverage_ok,
        "no_overlap": no_overlap,
        "date_order_ok": date_order_ok,
        "n_overlap_train_val": len(overlap_train_val),
        "n_overlap_train_test": len(overlap_train_test),
        "n_overlap_val_test": len(overlap_val_test),
    }
    return passed, detail


if __name__ == "__main__":
    import json
    import logging

    from ml.dataset import assemble_feature_label_table

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    table, _ = assemble_feature_label_table()
    train_df, val_df, test_df = chronological_split(table)
    passed, detail = validate_split_partition(table, train_df, val_df, test_df)
    print(json.dumps({"passed": passed, **detail}, indent=2, default=str))
