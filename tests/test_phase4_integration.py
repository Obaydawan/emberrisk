"""
Phase 4 integration test: wires ml.dataset -> ml.split -> ml.train ->
ml.evaluate together end-to-end on synthetic data. Unit tests already cover
each module in isolation (see test_ml_dataset.py, test_ml_split.py,
test_ml_baseline.py, test_ml_evaluate.py) -- this file exists only to catch
integration-level breakage (e.g. a column name drifting between modules)
that per-module unit tests with their own fixtures wouldn't necessarily
surface.
"""
import numpy as np
import pandas as pd

from ml.dataset import assemble_feature_label_table, impute_frp, FEATURE_COLUMNS, PRIMARY_TARGET_COLUMN
from ml.split import chronological_split, validate_split_partition
from ml.train import fit_baselines
from ml.evaluate import evaluate_all_baselines


def _synthetic_cell_day_dataset(n_cells=5, start="2018-01-01", end="2025-12-31", seed=0):
    """Builds a synthetic table with the SAME shape/columns as the real
    Phase 3 output (cell_day_dataset.parquet joined with a target), so this
    test exercises the exact column contract every downstream module
    expects -- without needing real Parquet files."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")

    frames = []
    for i in range(n_cells):
        df = pd.DataFrame({"cell_id": f"C{i}", "date": dates})
        df["fire_count"] = rng.integers(0, 3, size=len(dates))
        df["frp_mean"] = np.where(df["fire_count"] > 0, rng.uniform(1, 10, len(dates)), np.nan)
        df["frp_max"] = np.where(df["fire_count"] > 0, rng.uniform(5, 20, len(dates)), np.nan)
        df["fire_count_3d"] = df["fire_count"].rolling(3, min_periods=1).sum()
        df["fire_count_7d"] = df["fire_count"].rolling(7, min_periods=1).sum()
        df["fire_count_14d"] = df["fire_count"].rolling(14, min_periods=1).sum()
        df["fire_count_30d"] = df["fire_count"].rolling(30, min_periods=1).sum()
        df["frp_mean_7d"] = df["frp_mean"].rolling(7, min_periods=1).mean()
        df["frp_max_7d"] = df["frp_max"].rolling(7, min_periods=1).max()
        df["days_since_last_detection"] = rng.integers(0, 91, size=len(dates))
        df["temperature_max"] = rng.normal(25, 5, size=len(dates))
        df["temperature_min"] = rng.normal(10, 5, size=len(dates))
        df["relative_humidity"] = rng.uniform(20, 80, size=len(dates))
        df["precipitation"] = rng.exponential(1, size=len(dates))
        df["wind_speed"] = rng.uniform(0, 10, size=len(dates))
        frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)

    # synthetic target with a realistic-ish ~18% positive rate, correlated
    # with fire_count_7d so the logistic regression has something to learn
    prob = 0.08 + 0.25 * (dataset["fire_count_7d"] > 2)
    target = (rng.random(len(dataset)) < prob).astype(int)
    dataset[PRIMARY_TARGET_COLUMN] = target

    return dataset


def test_full_phase4_pipeline_runs_end_to_end_on_synthetic_data():
    dataset = _synthetic_cell_day_dataset(n_cells=4)
    # In the real pipeline, FRP NaN imputation happens inside
    # assemble_feature_label_table() before features ever reach a split or
    # a model -- apply the SAME real function here rather than hand-rolling
    # clean synthetic data, so this test exercises the actual contract.
    dataset = impute_frp(dataset)

    # exercise the real split boundaries, real fit_baselines, real evaluate
    train_df, val_df, test_df = chronological_split(dataset)
    passed, detail = validate_split_partition(dataset, train_df, val_df, test_df)
    assert passed, detail

    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0

    models = fit_baselines(train_df)
    results = evaluate_all_baselines(models, val_df)

    assert set(results.keys()) == {"majority_class", "persistence", "logistic_regression"}
    for name, m in results.items():
        assert m["n_samples"] == len(val_df)
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0

    # TEST split exists and is correctly sized but is never fed into evaluation
    assert len(test_df) > 0


def test_assemble_feature_label_table_output_is_split_and_evaluate_compatible():
    """Confirms ml.dataset's real output function produces a table that
    ml.split / ml.train / ml.evaluate can consume directly -- using a
    synthetic cell_day_dataset + target pair passed explicitly (no real
    files), but through the REAL assemble_feature_label_table() function,
    not a hand-rolled substitute."""
    cell_day = _synthetic_cell_day_dataset(n_cells=3).drop(columns=[PRIMARY_TARGET_COLUMN])
    target_df = pd.DataFrame({
        "cell_id": cell_day["cell_id"],
        "date": cell_day["date"],
    })
    rng = np.random.default_rng(42)
    target_df[PRIMARY_TARGET_COLUMN] = rng.integers(0, 2, size=len(target_df))

    table, report = assemble_feature_label_table(cell_day, target_df)

    train_df, val_df, test_df = chronological_split(table)
    models = fit_baselines(train_df)
    results = evaluate_all_baselines(models, val_df)

    assert report["n_after"] == len(table)
    assert all(col in table.columns for col in FEATURE_COLUMNS)
    assert set(results.keys()) == {"majority_class", "persistence", "logistic_regression"}
