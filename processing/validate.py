"""
Data quality checks for the canonical cell-day dataset, per the checklist
in the Phase 3 requirements. Each check function returns (passed: bool,
detail: dict) so the aggregator can produce one structured report rather
than raising on the first failure -- useful for spotting several issues in
one run instead of fixing them one at a time.
"""
import pandas as pd

from ingestion.common.grid import enumerate_grid_cells, CA_BBOX
from processing import EXPECTED_CELL_COUNT, EXPECTED_CELL_DAY_COUNT, MODELING_START, MODELING_END
from processing.power_standardize import STANDARD_COLUMNS as POWER_STANDARD_COLUMNS


def check_cell_count(dataset, bbox=CA_BBOX):
    n = dataset["cell_id"].nunique()
    return n == EXPECTED_CELL_COUNT, {"expected": EXPECTED_CELL_COUNT, "actual": n}


def check_date_range(dataset, start=MODELING_START, end=MODELING_END):
    actual_min, actual_max = dataset["date"].min(), dataset["date"].max()
    ok = actual_min == start and actual_max == end
    return ok, {"expected": [str(start.date()), str(end.date())],
                "actual": [str(actual_min.date()), str(actual_max.date())]}


def check_row_count(dataset, expected=EXPECTED_CELL_DAY_COUNT):
    n = len(dataset)
    return n == expected, {"expected": expected, "actual": n}


def check_no_duplicate_cell_date(dataset):
    n_dupes = int(dataset.duplicated(subset=["cell_id", "date"]).sum())
    return n_dupes == 0, {"n_duplicates": n_dupes}


def check_no_null_cell_id(dataset):
    n_null = int(dataset["cell_id"].isna().sum())
    return n_null == 0, {"n_null": n_null}


def check_no_dates_outside_period(dataset, start=MODELING_START, end=MODELING_END):
    outside = dataset[(dataset["date"] < start) | (dataset["date"] > end)]
    return len(outside) == 0, {"n_outside": len(outside)}


def check_fire_counts_non_negative(dataset):
    cols = [c for c in dataset.columns if c.startswith("fire_count")]
    bad = {c: int((dataset[c] < 0).sum()) for c in cols if (dataset[c] < 0).any()}
    return len(bad) == 0, {"columns_with_negatives": bad}


def check_frp_sensible(dataset):
    issues = {}
    for c in ("frp_mean_7d", "frp_max_7d"):
        if c in dataset.columns:
            n_neg = int((dataset[c] < 0).sum())
            if n_neg:
                issues[c] = n_neg
    return len(issues) == 0, {"columns_with_negative_frp": issues}


def check_power_ranges_and_missingness(dataset):
    report = {}
    ranges = {
        "temperature_max": (-60, 60),
        "temperature_min": (-60, 60),
        "relative_humidity": (0, 100),
        "precipitation": (0, None),
        "wind_speed": (0, None),
    }
    for col, (lo, hi) in ranges.items():
        if col not in dataset.columns:
            continue
        n_missing = int(dataset[col].isna().sum())
        out_of_range = dataset[col].dropna()
        n_below = int((out_of_range < lo).sum()) if lo is not None else 0
        n_above = int((out_of_range > hi).sum()) if hi is not None else 0
        report[col] = {
            "missing_pct": round(100 * n_missing / len(dataset), 4) if len(dataset) else None,
            "n_below_expected_range": n_below,
            "n_above_expected_range": n_above,
        }
    # This check is informational (documents missingness/ranges) rather than
    # pass/fail, since some missingness may be legitimate -- always "passes"
    # but the detail must be inspected.
    return True, report


def check_target_binary_or_null(targets_df, horizon_col):
    vals = targets_df[horizon_col].dropna().unique()
    ok = set(vals).issubset({0, 1})
    return ok, {"unique_non_null_values": sorted(int(v) for v in vals)}


def check_power_cell_count(power_df, expected=EXPECTED_CELL_COUNT):
    n = power_df["cell_id"].nunique()
    return n == expected, {"expected": expected, "actual": n}


def check_power_date_range(power_df, start=MODELING_START, end=MODELING_END):
    actual_min, actual_max = power_df["date"].min(), power_df["date"].max()
    ok = actual_min == start and actual_max == end
    return ok, {"expected": [str(start.date()), str(end.date())],
                "actual": [str(actual_min.date()), str(actual_max.date())]}


def check_power_no_duplicates(power_df):
    n_dupes = int(power_df.duplicated(subset=["cell_id", "date"]).sum())
    return n_dupes == 0, {"n_duplicates": n_dupes}


def check_power_row_count(power_df, expected=EXPECTED_CELL_DAY_COUNT):
    n = len(power_df)
    return n == expected, {"expected": expected, "actual": n}


def check_power_join_completeness(dataset, power_cols=POWER_STANDARD_COLUMNS):
    """Detects cell-days in the final (left-joined) dataset that have NO
    matching POWER row at all -- i.e. every POWER column is null, which
    only happens when the join found nothing, not from an individual
    sentinel-derived NaN in one variable. This is deliberately a strict
    left join upstream (see build_dataset.py) specifically so this gap is
    visible here rather than silently dropped."""
    missing_mask = dataset[list(power_cols)].isna().all(axis=1)
    n_missing = int(missing_mask.sum())
    detail = {"n_unmatched_cell_days": n_missing, "power_cols_checked": list(power_cols)}
    if n_missing:
        sample = dataset.loc[missing_mask, ["cell_id", "date"]].head(10)
        detail["sample_unmatched"] = sample.astype(str).to_dict(orient="records")
    return n_missing == 0, detail



def check_cell_id_set_matches_canonical(df, bbox=CA_BBOX):
    """Stronger than a count check: verifies the ACTUAL set of cell IDs
    present equals the canonical 323-cell set exactly -- catches the case
    where the count happens to match (323) but the actual IDs differ (one
    canonical cell silently missing, one bogus cell silently present)."""
    canonical_ids = {c.cell_id for c in enumerate_grid_cells(bbox)}
    actual_ids = set(df["cell_id"].unique())
    missing = canonical_ids - actual_ids
    extra = actual_ids - canonical_ids
    ok = not missing and not extra
    return ok, {
        "n_missing_canonical_cells": len(missing),
        "n_extra_non_canonical_cells": len(extra),
        "missing_sample": sorted(missing)[:10],
        "extra_sample": sorted(extra)[:10],
    }


def check_target_cell_date_coverage(targets_df, bbox=CA_BBOX, start=MODELING_START, end=MODELING_END):
    """Verifies the target table's (cell_id, date) keys are EXACTLY the
    canonical scaffold -- no missing cell-days, no extra/unexpected ones."""
    from processing.grid_scaffold import build_scaffold
    expected_scaffold = build_scaffold(start, end, bbox)
    expected_keys = set(zip(expected_scaffold["cell_id"], expected_scaffold["date"]))
    actual_keys = set(zip(targets_df["cell_id"], targets_df["date"]))
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    ok = not missing and not extra
    return ok, {"n_missing_cell_days": len(missing), "n_extra_cell_days": len(extra)}


def check_target_row_count(targets_df, expected=EXPECTED_CELL_DAY_COUNT):
    n = len(targets_df)
    return n == expected, {"expected": expected, "actual": n}


def check_target_no_duplicate_keys(targets_df):
    n_dupes = int(targets_df.duplicated(subset=["cell_id", "date"]).sum())
    return n_dupes == 0, {"n_duplicates": n_dupes}


def run_all_checks(dataset, targets_by_horizon=None, power_df=None):
    """Aggregates all checks into one report dict. Does not raise -- caller
    decides whether to treat any failed check as blocking. power_df (the
    pre-join standardized POWER table) is optional -- POWER-specific checks
    are skipped without it, existing callers still work."""
    report = {}
    for name, fn in [
        ("cell_count", check_cell_count),
        ("date_range", check_date_range),
        ("row_count", check_row_count),
        ("no_duplicate_cell_date", check_no_duplicate_cell_date),
        ("no_null_cell_id", check_no_null_cell_id),
        ("no_dates_outside_period", check_no_dates_outside_period),
        ("fire_counts_non_negative", check_fire_counts_non_negative),
        ("frp_sensible", check_frp_sensible),
        ("power_ranges_and_missingness", check_power_ranges_and_missingness),
    ]:
        passed, detail = fn(dataset)
        report[name] = {"passed": passed, "detail": detail}

    if power_df is not None:
        for name, fn in [
            ("power_cell_count", check_power_cell_count),
            ("power_date_range", check_power_date_range),
            ("power_no_duplicates", check_power_no_duplicates),
            ("power_row_count", check_power_row_count),
        ]:
            passed, detail = fn(power_df)
            report[name] = {"passed": passed, "detail": detail}

    passed, detail = check_power_join_completeness(dataset)
    report["power_join_completeness"] = {"passed": passed, "detail": detail}

    passed, detail = check_cell_id_set_matches_canonical(dataset)
    report["dataset_cell_id_set_matches_canonical"] = {"passed": passed, "detail": detail}

    if power_df is not None:
        passed, detail = check_cell_id_set_matches_canonical(power_df)
        report["power_cell_id_set_matches_canonical"] = {"passed": passed, "detail": detail}

    if targets_by_horizon:
        for h, tdf in targets_by_horizon.items():
            col = f"future_fire_{h}d"
            passed, detail = check_target_binary_or_null(tdf, col)
            detail["n_excluded_end_of_period"] = tdf.attrs.get("n_excluded_end_of_period")
            report[f"target_h{h}_binary_or_null"] = {"passed": passed, "detail": detail}

            passed, detail = check_target_cell_date_coverage(tdf)
            report[f"target_h{h}_cell_date_coverage"] = {"passed": passed, "detail": detail}

            passed, detail = check_target_row_count(tdf)
            report[f"target_h{h}_row_count"] = {"passed": passed, "detail": detail}

            passed, detail = check_target_no_duplicate_keys(tdf)
            report[f"target_h{h}_no_duplicate_keys"] = {"passed": passed, "detail": detail}

    return report
