"""
Phase 5 integration test: wires ml.dataset -> ml.split -> ml.compare
(baselines + tree models) -> ml.evaluate together end-to-end on synthetic
data. Confirms all 5 models go through the identical evaluation path and
that TEST is never touched anywhere in the Phase 5 orchestration.
"""
import inspect

import numpy as np
import pandas as pd

from ml.dataset import assemble_feature_label_table, impute_frp, FEATURE_COLUMNS, PRIMARY_TARGET_COLUMN
from ml.split import chronological_split, validate_split_partition
from ml.compare import fit_all_models, run as compare_run
import ml.compare as compare_module
from ml.evaluate import evaluate_all_baselines


def _synthetic_cell_day_dataset(n_cells=4, start="2018-01-01", end="2025-12-31", seed=0):
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
    prob = 0.08 + 0.25 * (dataset["fire_count_7d"] > 2)
    dataset[PRIMARY_TARGET_COLUMN] = (rng.random(len(dataset)) < prob).astype(int)
    return dataset


def test_fit_all_models_combines_baselines_and_tree_models():
    dataset = impute_frp(_synthetic_cell_day_dataset(n_cells=3))
    train_df, val_df, test_df = chronological_split(dataset)

    models = fit_all_models(train_df)

    assert set(models.keys()) == {
        "majority_class", "persistence", "logistic_regression",
        "random_forest", "gradient_boosting",
    }


def test_all_five_models_evaluate_through_identical_path_on_validation():
    dataset = impute_frp(_synthetic_cell_day_dataset(n_cells=3))
    train_df, val_df, test_df = chronological_split(dataset)
    passed, detail = validate_split_partition(dataset, train_df, val_df, test_df)
    assert passed, detail

    models = fit_all_models(train_df)
    results = evaluate_all_baselines(models, val_df)  # same function Phase 4 used, unmodified

    assert len(results) == 5
    for name, m in results.items():
        assert m["n_samples"] == len(val_df)
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert "confusion_matrix" in m

    # TEST split exists, correctly sized, but never fed into evaluation
    assert len(test_df) > 0


def test_compare_run_never_references_test_df_in_evaluation():
    """Source-level check on the real Phase 5 entrypoint: confirms
    evaluate_all_baselines is only ever called with val_df."""
    source = inspect.getsource(compare_module.run)
    assert "evaluate_all_baselines(models, val_df)" in source
    assert "evaluate_all_baselines(models, test_df)" not in source


def test_assemble_output_flows_through_full_phase5_pipeline():
    """Uses the REAL assemble_feature_label_table() (not a hand-rolled
    substitute) with synthetic inputs, through the full Phase 5 chain."""
    cell_day = _synthetic_cell_day_dataset(n_cells=3).drop(columns=[PRIMARY_TARGET_COLUMN])
    target_df = pd.DataFrame({"cell_id": cell_day["cell_id"], "date": cell_day["date"]})
    rng = np.random.default_rng(9)
    target_df[PRIMARY_TARGET_COLUMN] = rng.integers(0, 2, size=len(target_df))

    table, report = assemble_feature_label_table(cell_day, target_df)
    train_df, val_df, test_df = chronological_split(table)

    models = fit_all_models(train_df)
    results = evaluate_all_baselines(models, val_df)

    assert len(models) == 5
    assert len(results) == 5
    assert all(col in table.columns for col in FEATURE_COLUMNS)
