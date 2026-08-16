"""
Unit tests for ml/split.py (Phase 4 Step 3), using synthetic dates only.
"""
import pandas as pd
import pytest

from ml.split import (
    chronological_split, validate_split_partition,
    TRAIN_START, TRAIN_END, VALIDATION_START, VALIDATION_END, TEST_START, TEST_END,
)


def _full_range_df(freq="D"):
    dates = pd.date_range(TRAIN_START, TEST_END, freq=freq)
    return pd.DataFrame({"cell_id": "C1", "date": dates, "value": range(len(dates))})


# ---------------------------------------------------------------------------
# Boundary correctness
# ---------------------------------------------------------------------------

def test_boundary_dates_assigned_to_correct_split():
    df = pd.DataFrame({
        "date": [
            TRAIN_START, TRAIN_END,                 # train edges
            VALIDATION_START, VALIDATION_END,        # validation edges
            TEST_START, TEST_END,                    # test edges
        ]
    })
    train_df, val_df, test_df = chronological_split(df)

    assert set(train_df["date"]) == {TRAIN_START, TRAIN_END}
    assert set(val_df["date"]) == {VALIDATION_START, VALIDATION_END}
    assert set(test_df["date"]) == {TEST_START, TEST_END}


def test_day_before_train_end_is_train_not_validation():
    df = pd.DataFrame({"date": [pd.Timestamp("2022-12-30")]})
    train_df, val_df, test_df = chronological_split(df)
    assert len(train_df) == 1 and len(val_df) == 0 and len(test_df) == 0


def test_day_after_validation_end_is_test_not_validation():
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")]})
    train_df, val_df, test_df = chronological_split(df)
    assert len(test_df) == 1 and len(val_df) == 0


# ---------------------------------------------------------------------------
# Out-of-range dates must raise, never be silently dropped
# ---------------------------------------------------------------------------

def test_date_before_train_start_raises():
    df = pd.DataFrame({"date": [pd.Timestamp("2017-12-31")]})
    with pytest.raises(ValueError):
        chronological_split(df)


def test_date_after_test_end_raises():
    df = pd.DataFrame({"date": [pd.Timestamp("2026-01-01")]})
    with pytest.raises(ValueError):
        chronological_split(df)


def test_missing_date_column_raises():
    df = pd.DataFrame({"not_date": [pd.Timestamp("2018-01-01")]})
    with pytest.raises(ValueError):
        chronological_split(df)


# ---------------------------------------------------------------------------
# Full-range coverage and row counts
# ---------------------------------------------------------------------------

def test_full_range_split_sizes_match_calendar_years():
    df = _full_range_df()
    train_df, val_df, test_df = chronological_split(df)

    # 2018-2022 = 5 years incl. one leap year (2020) = 1826 days
    assert len(train_df) == 1826
    # 2023 = 365 days
    assert len(val_df) == 365
    # 2024-2025 = 2 years incl. one leap year (2024) = 731 days
    assert len(test_df) == 731
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_split_is_deterministic_no_randomness():
    df = _full_range_df(freq="30D")  # sparser, faster
    result_a = chronological_split(df)
    result_b = chronological_split(df)
    for a, b in zip(result_a, result_b):
        pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# validate_split_partition
# ---------------------------------------------------------------------------

def test_validate_split_partition_passes_for_correct_split():
    df = _full_range_df(freq="7D")
    train_df, val_df, test_df = chronological_split(df)
    passed, detail = validate_split_partition(df, train_df, val_df, test_df)
    assert passed
    assert detail["coverage_ok"]
    assert detail["no_overlap"]
    assert detail["date_order_ok"]
    assert detail["n_sum"] == detail["n_total"]


def test_validate_split_partition_detects_missing_coverage():
    df = _full_range_df(freq="7D")
    train_df, val_df, test_df = chronological_split(df)
    train_df_short = train_df.iloc[1:]  # drop one row -> coverage gap

    passed, detail = validate_split_partition(df, train_df_short, val_df, test_df)
    assert not passed
    assert not detail["coverage_ok"]


def test_validate_split_partition_detects_row_duplicated_across_splits():
    df = _full_range_df(freq="7D")
    train_df, val_df, test_df = chronological_split(df)
    # Deliberately leak one train row into validation too
    val_df_leaked = pd.concat([val_df, train_df.iloc[[0]]])

    passed, detail = validate_split_partition(df, train_df, val_df_leaked, test_df)
    assert not passed
    assert not detail["no_overlap"]
    assert detail["n_overlap_train_val"] == 1


def test_validate_split_partition_detects_date_order_violation():
    df = _full_range_df(freq="7D")
    train_df, val_df, test_df = chronological_split(df)
    # Swap a late-train row into validation and an early-validation row into
    # train -- coverage and index-overlap both still look fine, but the
    # date ordering guarantee is now violated. This is the check that
    # specifically catches "leakage" in the temporal sense, not just
    # bookkeeping errors.
    late_train_row = train_df.iloc[[-1]]
    early_val_row = val_df.iloc[[0]]

    train_df_swapped = pd.concat([train_df.iloc[:-1], early_val_row])
    val_df_swapped = pd.concat([late_train_row, val_df.iloc[1:]])

    passed, detail = validate_split_partition(df, train_df_swapped, val_df_swapped, test_df)
    assert not passed
    assert not detail["date_order_ok"]
    # coverage and index-overlap are still fine in this scenario -- proves
    # date_order_ok is catching something the other two checks would miss
    assert detail["coverage_ok"]
    assert detail["no_overlap"]


# ---------------------------------------------------------------------------
# Multi-cell realism (not just a single synthetic series)
# ---------------------------------------------------------------------------

def test_split_works_across_multiple_cells():
    dates = pd.date_range(TRAIN_START, TEST_END, freq="180D")
    df = pd.concat([
        pd.DataFrame({"cell_id": cell, "date": dates})
        for cell in ["C1", "C2", "C3"]
    ], ignore_index=True)

    train_df, val_df, test_df = chronological_split(df)
    passed, detail = validate_split_partition(df, train_df, val_df, test_df)
    assert passed
    assert detail["n_total"] == len(df)
