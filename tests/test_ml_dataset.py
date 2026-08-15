"""
Unit tests for ml/dataset.py (Phase 4 Step 2), using small synthetic data --
no real Parquet files needed.
"""
import numpy as np
import pandas as pd
import pytest

from ml.dataset import (
    impute_frp, verify_no_unexpected_missingness, assemble_feature_label_table,
    FEATURE_COLUMNS, FRP_COLUMNS, PRIMARY_TARGET_COLUMN,
)


def _synthetic_cell_day_dataset(n=5):
    dates = pd.date_range("2018-01-01", periods=n)
    df = pd.DataFrame({
        "cell_id": ["C1"] * n,
        "date": dates,
        "fire_count": [0, 1, 0, 0, 2],
        "frp_mean": [np.nan, 5.0, np.nan, np.nan, 8.0],
        "frp_max": [np.nan, 6.0, np.nan, np.nan, 9.0],
        "fire_count_3d": [0, 1, 1, 1, 2],
        "fire_count_7d": [0, 1, 1, 1, 3],
        "fire_count_14d": [0, 1, 1, 1, 3],
        "fire_count_30d": [0, 1, 1, 1, 3],
        "frp_mean_7d": [np.nan, 5.0, 5.0, 5.0, 6.5],
        "frp_max_7d": [np.nan, 6.0, 6.0, 6.0, 9.0],
        "days_since_last_detection": [90, 0, 1, 2, 0],
        "temperature_max": [20.0, 21.0, 19.0, 18.0, 22.0],
        "temperature_min": [5.0, 6.0, 4.0, 3.0, 7.0],
        "relative_humidity": [50.0, 55.0, 52.0, 48.0, 45.0],
        "precipitation": [0.0, 0.0, 1.2, 0.0, 0.0],
        "wind_speed": [2.0, 2.5, 3.0, 1.5, 2.2],
    })
    return df


def _synthetic_target(n=5, all_labeled=True):
    dates = pd.date_range("2018-01-01", periods=n)
    labels = [0, 1, 0, 1, 0]
    if not all_labeled:
        labels[-1] = None  # simulate an end-of-period null label
    return pd.DataFrame({
        "cell_id": ["C1"] * n,
        "date": dates,
        "future_fire_7d": pd.array(labels, dtype="Int64"),
    })


# ---------------------------------------------------------------------------
# impute_frp
# ---------------------------------------------------------------------------

def test_impute_frp_fills_nan_with_zero():
    df = _synthetic_cell_day_dataset()
    imputed = impute_frp(df)
    for col in FRP_COLUMNS:
        assert imputed[col].isna().sum() == 0
    assert imputed.loc[0, "frp_mean"] == 0.0  # was NaN
    assert imputed.loc[1, "frp_mean"] == 5.0  # real value untouched


def test_impute_frp_does_not_touch_other_columns():
    df = _synthetic_cell_day_dataset()
    df.loc[0, "temperature_max"] = np.nan  # inject unrelated NaN
    imputed = impute_frp(df)
    assert imputed["temperature_max"].isna().sum() == 1  # untouched by FRP imputation


def test_impute_frp_raises_on_missing_column():
    df = _synthetic_cell_day_dataset().drop(columns=["frp_mean"])
    with pytest.raises(ValueError):
        impute_frp(df)


# ---------------------------------------------------------------------------
# verify_no_unexpected_missingness
# ---------------------------------------------------------------------------

def test_verify_no_unexpected_missingness_passes_when_clean():
    df = impute_frp(_synthetic_cell_day_dataset())
    verify_no_unexpected_missingness(df, FEATURE_COLUMNS)  # should not raise


def test_verify_no_unexpected_missingness_raises_when_dirty():
    df = _synthetic_cell_day_dataset()
    df.loc[0, "wind_speed"] = np.nan  # simulate unexpected real missingness
    with pytest.raises(ValueError):
        verify_no_unexpected_missingness(df, FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# assemble_feature_label_table
# ---------------------------------------------------------------------------

def test_assemble_joins_target_and_imputes_frp():
    cell_day = _synthetic_cell_day_dataset()
    target = _synthetic_target()
    table, report = assemble_feature_label_table(cell_day, target)

    assert PRIMARY_TARGET_COLUMN in table.columns
    assert table[PRIMARY_TARGET_COLUMN].tolist() == [0, 1, 0, 1, 0]
    assert table["frp_mean"].isna().sum() == 0  # imputed
    assert report["n_dropped_null_label"] == 0
    assert report["n_after"] == 5


def test_assemble_drops_null_label_rows_and_reports_count():
    cell_day = _synthetic_cell_day_dataset()
    target = _synthetic_target(all_labeled=False)  # last row has null label
    table, report = assemble_feature_label_table(cell_day, target)

    assert len(table) == 4  # one row dropped
    assert report["n_dropped_null_label"] == 1
    assert report["n_before_label_drop"] == 5
    assert report["n_after"] == 4
    # target column must be a clean int, no leftover nullable/NaN
    assert table[PRIMARY_TARGET_COLUMN].dtype.kind in "iu"


def test_assemble_computes_positive_rate():
    cell_day = _synthetic_cell_day_dataset()
    target = _synthetic_target()  # labels [0,1,0,1,0] -> 2/5 positive
    _, report = assemble_feature_label_table(cell_day, target)
    assert report["positive_rate"] == 0.4


def test_assemble_raises_on_missing_feature_column():
    cell_day = _synthetic_cell_day_dataset().drop(columns=["wind_speed"])
    target = _synthetic_target()
    with pytest.raises(ValueError):
        assemble_feature_label_table(cell_day, target)


def test_assemble_raises_on_missing_target_column():
    cell_day = _synthetic_cell_day_dataset()
    target = _synthetic_target().drop(columns=["future_fire_7d"])
    with pytest.raises(ValueError):
        assemble_feature_label_table(cell_day, target)


def test_assemble_does_not_touch_3d_or_14d_targets():
    """Structural check: assemble_feature_label_table only ever reads the
    target_df it's given -- it has no knowledge of targets_h3/h14 files at
    all, so it cannot accidentally merge or model them."""
    import inspect
    source = inspect.getsource(assemble_feature_label_table)
    assert "h3" not in source.lower()
    assert "h14" not in source.lower()
