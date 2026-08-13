"""
Unit tests for Phase 3 processing, using small synthetic data (no real
FIRMS/POWER files needed). Focus areas: scaffold correctness, confidence
filtering, out-of-grid exclusion, rolling-feature leakage safety, target
horizon correctness, and end-of-period boundary handling.
"""
import pandas as pd
import pytest

from ingestion.common.grid import enumerate_grid_cells, CA_BBOX
from processing.grid_scaffold import build_scaffold
from processing.firms_standardize import standardize_firms, aggregate_to_cell_day
from processing.fire_features import fill_scaffold_with_fire_counts, compute_fire_rolling_features
from processing.power_standardize import standardize_power, _cell_id_from_filename, STANDARD_COLUMNS as POWER_STANDARD_COLUMNS
from processing.targets import build_targets
from processing.validate import (
    check_cell_count, check_row_count, check_no_duplicate_cell_date,
    check_fire_counts_non_negative, check_target_binary_or_null,
    check_power_cell_count, check_power_date_range, check_power_no_duplicates,
    check_power_row_count, check_power_join_completeness,
    check_cell_id_set_matches_canonical, check_target_cell_date_coverage,
    check_target_row_count, check_target_no_duplicate_keys,
)


# ---------------------------------------------------------------------------
# Scaffold
# ---------------------------------------------------------------------------

def test_scaffold_full_modeling_period_row_count():
    scaffold = build_scaffold(pd.Timestamp("2018-01-01"), pd.Timestamp("2025-12-31"))
    assert len(scaffold) == 323 * 2922


def test_scaffold_no_duplicates():
    scaffold = build_scaffold(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-10"))
    assert not scaffold.duplicated(subset=["cell_id", "date"]).any()


# ---------------------------------------------------------------------------
# FIRMS standardization
# ---------------------------------------------------------------------------

def _sample_cell():
    return enumerate_grid_cells(CA_BBOX)[10]  # arbitrary in-grid cell


def test_standardize_firms_drops_low_confidence_flagged_not_removed():
    cell = _sample_cell()
    raw = pd.DataFrame({
        "latitude": [cell.centroid_lat, cell.centroid_lat],
        "longitude": [cell.centroid_lon, cell.centroid_lon],
        "acq_date": ["2018-01-01", "2018-01-01"],
        "confidence": ["l", "n"],
        "frp": [1.0, 2.0],
    })
    std = standardize_firms(raw)
    assert len(std) == 2  # both rows kept, just flagged
    assert std["qualifying"].tolist() == [False, True]


def test_standardize_firms_unknown_confidence_does_not_qualify():
    """The historical bug: '!= low' alone would let NaN/malformed confidence
    values through as qualifying by accident. Missing/unrecognized values
    must default to NOT qualifying."""
    cell = _sample_cell()
    raw = pd.DataFrame({
        "latitude": [cell.centroid_lat] * 3,
        "longitude": [cell.centroid_lon] * 3,
        "acq_date": ["2018-01-01"] * 3,
        "confidence": [None, "", "unexpected_value"],
        "frp": [1.0, 2.0, 3.0],
    })
    std = standardize_firms(raw)
    assert std["qualifying"].tolist() == [False, False, False]


def test_standardize_firms_known_high_and_nominal_both_qualify():
    cell = _sample_cell()
    raw = pd.DataFrame({
        "latitude": [cell.centroid_lat] * 2,
        "longitude": [cell.centroid_lon] * 2,
        "acq_date": ["2018-01-01"] * 2,
        "confidence": ["n", "h"],
        "frp": [1.0, 2.0],
    })
    std = standardize_firms(raw)
    assert std["qualifying"].tolist() == [True, True]


def test_standardize_firms_excludes_out_of_grid_rows():
    raw = pd.DataFrame({
        "latitude": [80.0],   # far outside CA bbox -> outside canonical grid
        "longitude": [-200.0],
        "acq_date": ["2018-01-01"],
        "confidence": ["n"],
        "frp": [1.0],
    })
    std = standardize_firms(raw)
    assert len(std) == 0  # dropped, not merged into a neighboring cell


def test_aggregate_to_cell_day_only_counts_qualifying():
    cell = _sample_cell()
    raw = pd.DataFrame({
        "latitude": [cell.centroid_lat] * 3,
        "longitude": [cell.centroid_lon] * 3,
        "acq_date": ["2018-01-01", "2018-01-01", "2018-01-01"],
        "confidence": ["l", "n", "h"],
        "frp": [1.0, 2.0, 4.0],
    })
    std = standardize_firms(raw)
    agg = aggregate_to_cell_day(std)
    assert len(agg) == 1
    assert agg.iloc[0]["fire_count"] == 2  # only the 'n' and 'h' rows
    assert agg.iloc[0]["frp_mean"] == 3.0  # mean of 2.0 and 4.0, 'l' excluded


# ---------------------------------------------------------------------------
# Rolling fire features -- leakage safety is the critical property here
# ---------------------------------------------------------------------------

def _single_cell_scaffold_with_counts(fire_days, start, end, cell_id="TEST_CELL"):
    dates = pd.date_range(start, end, freq="D")
    scaffold = pd.DataFrame({"cell_id": cell_id, "date": dates})
    scaffold["fire_count"] = [1 if d in fire_days else 0 for d in dates]
    scaffold["frp_mean"] = [5.0 if d in fire_days else None for d in dates]
    scaffold["frp_max"] = [8.0 if d in fire_days else None for d in dates]
    return scaffold


def test_rolling_fire_count_only_includes_past_and_present():
    fire_day = pd.Timestamp("2018-01-10")
    scaffold = _single_cell_scaffold_with_counts(
        {fire_day}, "2018-01-01", "2018-01-20"
    )
    result = compute_fire_rolling_features(scaffold)

    day_before = result[result["date"] == fire_day - pd.Timedelta(days=1)].iloc[0]
    day_of = result[result["date"] == fire_day].iloc[0]
    day_after = result[result["date"] == fire_day + pd.Timedelta(days=1)].iloc[0]

    assert day_before["fire_count_3d"] == 0     # fire hasn't happened yet
    assert day_of["fire_count_3d"] == 1          # fire counted on the day itself
    assert day_after["fire_count_3d"] == 1        # still within trailing window
    # 3 days later the fire should have rolled out of the 3d window
    day_plus3 = result[result["date"] == fire_day + pd.Timedelta(days=3)].iloc[0]
    assert day_plus3["fire_count_3d"] == 0


def test_days_since_last_detection():
    fire_day = pd.Timestamp("2018-01-05")
    scaffold = _single_cell_scaffold_with_counts({fire_day}, "2018-01-01", "2018-01-15")
    result = compute_fire_rolling_features(scaffold)

    assert result[result["date"] == fire_day]["days_since_last_detection"].iloc[0] == 0
    assert result[result["date"] == fire_day + pd.Timedelta(days=4)]["days_since_last_detection"].iloc[0] == 4
    # before any fire has occurred, capped rather than negative/undefined
    assert result[result["date"] == pd.Timestamp("2018-01-01")]["days_since_last_detection"].iloc[0] == 90


def test_warmup_enables_full_lookback_at_modeling_period_start():
    # fire in warm-up window, 10 days before modeling start
    warmup_fire = pd.Timestamp("2017-12-22")
    scaffold = _single_cell_scaffold_with_counts(
        {warmup_fire}, "2017-12-02", "2018-01-05"
    )
    result = compute_fire_rolling_features(scaffold)

    first_modeling_day = result[result["date"] == pd.Timestamp("2018-01-01")].iloc[0]
    # 2018-01-01 minus warmup_fire (2017-12-22) = 10 days -- within the 14d/30d windows
    assert first_modeling_day["fire_count_14d"] == 1
    assert first_modeling_day["fire_count_30d"] == 1
    assert first_modeling_day["fire_count_3d"] == 0  # outside the shorter window


def test_fill_scaffold_zero_fills_missing_days():
    scaffold = build_scaffold(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-05"))
    empty_agg = pd.DataFrame(columns=["cell_id", "date", "fire_count", "frp_mean", "frp_max"])
    filled = fill_scaffold_with_fire_counts(scaffold, empty_agg)
    assert (filled["fire_count"] == 0).all()
    assert len(filled) == len(scaffold)


# ---------------------------------------------------------------------------
# POWER standardization
# ---------------------------------------------------------------------------

def test_cell_id_from_filename_handles_hyphenated_cell_ids():
    fname = "data/raw/power/power_9_-199_2018-01-01_2018-01-10.parquet"
    assert _cell_id_from_filename(fname) == "9_-199"


def test_standardize_power_renames_and_keeps_expected_columns():
    raw = pd.DataFrame({
        "cell_id": ["9_-199", "9_-199"],
        "date": ["2018-01-01", "2018-01-02"],
        "T2M_MAX": [18.0, 19.0],
        "T2M_MIN": [5.0, 6.0],
        "RH2M": [55.0, 56.0],
        "PRECTOTCORR": [0.0, 1.2],
        "WS2M": [2.0, 2.5],
    })
    std = standardize_power(raw)
    assert list(std.columns) == [
        "cell_id", "date", "temperature_max", "temperature_min",
        "relative_humidity", "precipitation", "wind_speed",
    ]
    assert std.iloc[0]["temperature_max"] == 18.0


def test_standardize_power_raises_on_missing_expected_column():
    raw = pd.DataFrame({
        "cell_id": ["9_-199"], "date": ["2018-01-01"],
        "T2M_MAX": [18.0],  # missing the rest
    })
    with pytest.raises(ValueError):
        standardize_power(raw)


def test_standardize_power_converts_sentinel_to_nan():
    raw = pd.DataFrame({
        "cell_id": ["9_-199", "9_-199"],
        "date": ["2018-01-01", "2018-01-02"],
        "T2M_MAX": [18.0, -999.0],   # -999 is POWER's documented fill value
        "T2M_MIN": [5.0, -999.0],
        "RH2M": [55.0, 56.0],
        "PRECTOTCORR": [0.0, 1.2],
        "WS2M": [2.0, 2.5],
    })
    std = standardize_power(raw)
    assert std.iloc[0]["temperature_max"] == 18.0        # untouched real value
    assert pd.isna(std.iloc[1]["temperature_max"])         # -999 -> NaN
    assert pd.isna(std.iloc[1]["temperature_min"])         # -999 -> NaN
    assert std.iloc[1]["relative_humidity"] == 56.0        # real value untouched


def test_standardize_power_does_not_touch_values_above_threshold():
    # A legitimately low-but-real value (e.g. cold temperature) must NOT be
    # mistaken for the sentinel -- only values <= the documented threshold
    # (-900, well below any physically plausible reading) get nulled.
    raw = pd.DataFrame({
        "cell_id": ["9_-199"], "date": ["2018-01-01"],
        "T2M_MAX": [-40.0],  # cold but physically real, far above -900
        "T2M_MIN": [-45.0],
        "RH2M": [30.0], "PRECTOTCORR": [0.0], "WS2M": [5.0],
    })
    std = standardize_power(raw)
    assert std.iloc[0]["temperature_max"] == -40.0
    assert std.iloc[0]["temperature_min"] == -45.0


# ---------------------------------------------------------------------------
# POWER completeness / uniqueness validation
# ---------------------------------------------------------------------------

def _full_power_df():
    from processing.grid_scaffold import build_scaffold
    from processing import MODELING_START, MODELING_END
    scaffold = build_scaffold(MODELING_START, MODELING_END)
    scaffold[POWER_STANDARD_COLUMNS] = 1.0
    return scaffold


def test_check_power_cell_count_pass_and_fail():
    power_df = _full_power_df()
    passed, _ = check_power_cell_count(power_df)
    assert passed

    fewer_cells = power_df[power_df["cell_id"] != power_df["cell_id"].iloc[0]]
    passed2, detail2 = check_power_cell_count(fewer_cells)
    assert not passed2
    assert detail2["actual"] == 322


def test_check_power_date_range_pass_and_fail():
    power_df = _full_power_df()
    passed, _ = check_power_date_range(power_df)
    assert passed

    truncated = power_df[power_df["date"] < power_df["date"].max()]
    passed2, _ = check_power_date_range(truncated)
    assert not passed2


def test_check_power_no_duplicates_detects_dupes():
    power_df = _full_power_df()
    dupe_row = power_df.iloc[[0]]
    with_dupe = pd.concat([power_df, dupe_row], ignore_index=True)
    passed, detail = check_power_no_duplicates(with_dupe)
    assert not passed
    assert detail["n_duplicates"] == 1


def test_check_power_row_count_matches_expected():
    power_df = _full_power_df()
    passed, detail = check_power_row_count(power_df)
    assert passed
    assert detail["actual"] == 943_806


def test_check_power_join_completeness_detects_unmatched_cell_days():
    from processing.grid_scaffold import build_scaffold
    from processing import MODELING_START, MODELING_END

    # Simulate a joined dataset where one cell-day has no POWER match at all
    # (all POWER columns null) -- this must be caught, not hidden by an
    # inner join silently removing the row.
    scaffold = build_scaffold(MODELING_START, MODELING_END)
    dataset = scaffold.copy()
    for c in POWER_STANDARD_COLUMNS:
        dataset[c] = 1.0
    dataset.loc[0, POWER_STANDARD_COLUMNS] = None  # simulate one unmatched row

    passed, detail = check_power_join_completeness(dataset)
    assert not passed
    assert detail["n_unmatched_cell_days"] == 1


def test_check_power_join_completeness_passes_when_fully_matched():
    from processing.grid_scaffold import build_scaffold
    from processing import MODELING_START, MODELING_END

    scaffold = build_scaffold(MODELING_START, MODELING_END)
    dataset = scaffold.copy()
    for c in POWER_STANDARD_COLUMNS:
        dataset[c] = 1.0

    passed, detail = check_power_join_completeness(dataset)
    assert passed
    assert detail["n_unmatched_cell_days"] == 0


# ---------------------------------------------------------------------------
# Target construction -- horizon correctness + end-of-period boundary
# ---------------------------------------------------------------------------

def test_target_positive_when_fire_in_future_window():
    dates = pd.date_range("2018-01-01", "2018-01-20")
    fire_daily = pd.DataFrame({
        "cell_id": "TEST_CELL", "date": dates,
        "fire_count": [1 if d == pd.Timestamp("2018-01-10") else 0 for d in dates],
    })
    targets = build_targets(fire_daily, pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-20"), horizons=(7,))
    t7 = targets[7]

    # T=2018-01-03: fire on 01-10 falls within [T+1, T+7] = [01-04, 01-10]
    row = t7[t7["date"] == pd.Timestamp("2018-01-03")].iloc[0]
    assert row["future_fire_7d"] == 1

    # T=2018-01-10 itself: future window is [01-11, 01-17], fire on 01-10 is NOT in it
    row_same_day = t7[t7["date"] == pd.Timestamp("2018-01-10")].iloc[0]
    assert row_same_day["future_fire_7d"] == 0


def test_target_negative_when_no_future_fire():
    dates = pd.date_range("2018-01-01", "2018-01-20")
    fire_daily = pd.DataFrame({"cell_id": "TEST_CELL", "date": dates, "fire_count": 0})
    targets = build_targets(fire_daily, pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-20"), horizons=(7,))
    t7 = targets[7]
    early_row = t7[t7["date"] == pd.Timestamp("2018-01-01")].iloc[0]
    assert early_row["future_fire_7d"] == 0


def test_target_end_of_period_boundary_is_null_not_fabricated():
    dates = pd.date_range("2018-01-01", "2018-01-20")
    fire_daily = pd.DataFrame({"cell_id": "TEST_CELL", "date": dates, "fire_count": 0})
    modeling_end = pd.Timestamp("2018-01-20")
    targets = build_targets(fire_daily, pd.Timestamp("2018-01-01"), modeling_end, horizons=(7,))
    t7 = targets[7]

    # last 7 days can't have a fully observed future window -> null, not 0/1
    last_day = t7[t7["date"] == modeling_end].iloc[0]
    assert pd.isna(last_day["future_fire_7d"])

    # the exclusion count must be attached, not silently swallowed
    assert t7.attrs["n_excluded_end_of_period"] == 7

    # a day just inside the usable range must still have a real 0/1 label
    usable_day = t7[t7["date"] == modeling_end - pd.Timedelta(days=7)].iloc[0]
    assert usable_day["future_fire_7d"] in (0, 1)


def test_target_no_leakage_changing_far_future_does_not_change_early_label():
    """If a feature/label computation is leaking, changing a fire far in the
    future could ripple backward incorrectly. This test pins down that only
    fires WITHIN the horizon window affect a given day's label."""
    dates = pd.date_range("2018-01-01", "2018-02-01")
    fire_daily_a = pd.DataFrame({"cell_id": "TEST_CELL", "date": dates, "fire_count": 0})
    fire_daily_b = fire_daily_a.copy()
    # add a fire far beyond any 14-day horizon from 2018-01-01
    fire_daily_b.loc[fire_daily_b["date"] == pd.Timestamp("2018-01-31"), "fire_count"] = 1

    targets_a = build_targets(fire_daily_a, pd.Timestamp("2018-01-01"), pd.Timestamp("2018-02-01"), horizons=(14,))
    targets_b = build_targets(fire_daily_b, pd.Timestamp("2018-01-01"), pd.Timestamp("2018-02-01"), horizons=(14,))

    label_a = targets_a[14][targets_a[14]["date"] == pd.Timestamp("2018-01-01")]["future_fire_14d"].iloc[0]
    label_b = targets_b[14][targets_b[14]["date"] == pd.Timestamp("2018-01-01")]["future_fire_14d"].iloc[0]
    assert label_a == label_b == 0  # the 01-31 fire is outside [01-02, 01-15], must not affect this label


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def test_check_row_count_pass_and_fail():
    scaffold = build_scaffold(pd.Timestamp("2018-01-01"), pd.Timestamp("2025-12-31"))
    passed, detail = check_row_count(scaffold)
    assert passed
    passed2, detail2 = check_row_count(scaffold.iloc[:-1])
    assert not passed2


def test_check_no_duplicate_cell_date_detects_dupes():
    df = pd.DataFrame({"cell_id": ["a", "a"], "date": [pd.Timestamp("2018-01-01")] * 2})
    passed, detail = check_no_duplicate_cell_date(df)
    assert not passed
    assert detail["n_duplicates"] == 1


def test_check_fire_counts_non_negative_flags_bad_data():
    df = pd.DataFrame({"fire_count_7d": [1, -1, 2]})
    passed, detail = check_fire_counts_non_negative(df)
    assert not passed
    assert detail["columns_with_negatives"]["fire_count_7d"] == 1


def test_check_target_binary_or_null_accepts_na():
    df = pd.DataFrame({"future_fire_7d": pd.array([0, 1, pd.NA], dtype="Int64")})
    passed, detail = check_target_binary_or_null(df, "future_fire_7d")
    assert passed


def test_check_target_binary_or_null_rejects_bad_values():
    df = pd.DataFrame({"future_fire_7d": pd.array([0, 1, 2], dtype="Int64")})
    passed, detail = check_target_binary_or_null(df, "future_fire_7d")
    assert not passed


# ---------------------------------------------------------------------------
# Exact cell-ID-set validation
# ---------------------------------------------------------------------------

def test_cell_id_set_matches_canonical_passes_for_full_grid():
    from processing.grid_scaffold import build_scaffold
    scaffold = build_scaffold(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-02"))
    passed, detail = check_cell_id_set_matches_canonical(scaffold)
    assert passed
    assert detail["n_missing_canonical_cells"] == 0
    assert detail["n_extra_non_canonical_cells"] == 0


def test_cell_id_set_matches_canonical_detects_missing_cell_same_count():
    """Count-based checks alone would miss this: swap one canonical cell ID
    for a bogus one, keeping the total count at 323. Only an exact set
    comparison catches it."""
    from processing.grid_scaffold import build_scaffold
    scaffold = build_scaffold(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-02"))
    swapped = scaffold.copy()
    real_cell_ids = swapped["cell_id"].unique()
    swapped.loc[swapped["cell_id"] == real_cell_ids[0], "cell_id"] = "BOGUS_CELL"

    passed, detail = check_cell_id_set_matches_canonical(swapped)
    assert not passed
    assert detail["n_missing_canonical_cells"] == 1
    assert detail["n_extra_non_canonical_cells"] == 1
    # count alone is unchanged -- proves this check adds real value
    assert swapped["cell_id"].nunique() == scaffold["cell_id"].nunique() == 323


def test_cell_id_set_matches_canonical_detects_missing_cell_entirely():
    from processing.grid_scaffold import build_scaffold
    scaffold = build_scaffold(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-02"))
    dropped_cell = scaffold["cell_id"].unique()[0]
    without_one_cell = scaffold[scaffold["cell_id"] != dropped_cell]

    passed, detail = check_cell_id_set_matches_canonical(without_one_cell)
    assert not passed
    assert detail["n_missing_canonical_cells"] == 1
    assert dropped_cell in detail["missing_sample"]


# ---------------------------------------------------------------------------
# Target (cell_id, date) coverage validation
# ---------------------------------------------------------------------------

def _full_targets_df(horizon_col="future_fire_7d"):
    from processing.grid_scaffold import build_scaffold
    from processing import MODELING_START, MODELING_END
    scaffold = build_scaffold(MODELING_START, MODELING_END)
    scaffold[horizon_col] = 0
    return scaffold


def test_target_cell_date_coverage_passes_for_full_scaffold():
    targets = _full_targets_df()
    passed, detail = check_target_cell_date_coverage(targets)
    assert passed
    assert detail["n_missing_cell_days"] == 0
    assert detail["n_extra_cell_days"] == 0


def test_target_cell_date_coverage_detects_missing_row():
    targets = _full_targets_df().iloc[1:]  # drop the first cell-day
    passed, detail = check_target_cell_date_coverage(targets)
    assert not passed
    assert detail["n_missing_cell_days"] == 1


def test_target_row_count_matches_expected():
    targets = _full_targets_df()
    passed, detail = check_target_row_count(targets)
    assert passed
    assert detail["actual"] == 943_806


def test_target_row_count_fails_on_mismatch():
    targets = _full_targets_df().iloc[:-100]
    passed, detail = check_target_row_count(targets)
    assert not passed


def test_target_no_duplicate_keys_detects_dupes():
    targets = _full_targets_df()
    dupe_row = targets.iloc[[0]]
    with_dupe = pd.concat([targets, dupe_row], ignore_index=True)
    passed, detail = check_target_no_duplicate_keys(with_dupe)
    assert not passed
    assert detail["n_duplicates"] == 1


def test_target_no_duplicate_keys_passes_clean():
    targets = _full_targets_df()
    passed, detail = check_target_no_duplicate_keys(targets)
    assert passed
