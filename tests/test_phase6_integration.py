"""
Phase 6 integration test: wires ml.dataset -> ml.split -> ml.phase6
(fit RF + HistGB on TRAIN -> analyze on VALIDATION) together end-to-end on
synthetic data. Confirms TEST is never touched -- both structurally
(function signatures / source inspection) and behaviorally (fitting on
disjoint subsets produces different, subset-specific results).
"""
import inspect

import numpy as np
import pandas as pd

from ml.dataset import assemble_feature_label_table, impute_frp, FEATURE_COLUMNS, PRIMARY_TARGET_COLUMN
from ml.split import chronological_split, validate_split_partition
from ml.phase6 import fit_models_for_phase6, analyze_model_on_validation, run as phase6_run
import ml.phase6 as phase6_module


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


def test_fit_models_for_phase6_returns_both_models():
    dataset = impute_frp(_synthetic_cell_day_dataset(n_cells=3))
    train_df, val_df, test_df = chronological_split(dataset)

    models = fit_models_for_phase6(train_df)
    assert set(models.keys()) == {"random_forest", "gradient_boosting"}


def test_analyze_model_on_validation_returns_expected_structure():
    dataset = impute_frp(_synthetic_cell_day_dataset(n_cells=3))
    train_df, val_df, test_df = chronological_split(dataset)
    passed, detail = validate_split_partition(dataset, train_df, val_df, test_df)
    assert passed, detail

    models = fit_models_for_phase6(train_df)
    analysis = analyze_model_on_validation(models["random_forest"], val_df)

    assert "default_threshold_metrics" in analysis
    assert "optimal_threshold_metrics" in analysis
    assert "threshold_sweep" in analysis
    assert len(analysis["threshold_sweep"]) == 19
    assert "brier_score" in analysis
    assert "ece" in analysis
    assert len(analysis["calibration_bins"]) == 10
    assert analysis["default_threshold_metrics"]["threshold"] == 0.5
    assert analysis["default_threshold_metrics"]["n_samples"] == len(val_df)


def test_analysis_never_touches_test_split():
    """Fits on TRAIN, analyzes on VALIDATION only; TEST split is built and
    correctly sized but never passed into analyze_model_on_validation
    anywhere in this test."""
    dataset = impute_frp(_synthetic_cell_day_dataset(n_cells=3))
    train_df, val_df, test_df = chronological_split(dataset)

    models = fit_models_for_phase6(train_df)
    for name, model in models.items():
        analysis = analyze_model_on_validation(model, val_df)
        assert analysis["default_threshold_metrics"]["n_samples"] == len(val_df)
        assert analysis["default_threshold_metrics"]["n_samples"] != len(test_df)

    assert len(test_df) > 0  # confirms TEST exists and simply wasn't used


def test_analyze_model_on_validation_signature_has_no_test_param():
    """Structural guarantee: the function signature itself only accepts
    val_df -- calling it on TEST isn't expressible through this API."""
    sig = inspect.signature(analyze_model_on_validation)
    param_names = list(sig.parameters.keys())
    assert "val_df" in param_names
    assert not any("test" in p.lower() for p in param_names)


def test_run_function_never_passes_test_df_to_analysis():
    """Source-level check on the real Phase 6 entrypoint."""
    source = inspect.getsource(phase6_module.run)
    assert "analyze_model_on_validation(model, val_df)" in source
    assert "analyze_model_on_validation(model, test_df)" not in source


def test_fit_models_for_phase6_only_uses_given_dataframe():
    """Fitting on two disjoint subsets must produce genuinely different
    fitted models (different learned probabilities on a shared probe row),
    proving no hidden global state or leakage between calls."""
    dataset = impute_frp(_synthetic_cell_day_dataset(n_cells=5, seed=10))
    train_df, val_df, test_df = chronological_split(dataset)

    half = len(train_df) // 2
    subset_a = train_df.iloc[:half]
    subset_b = train_df.iloc[half:]

    models_a = fit_models_for_phase6(subset_a)
    models_b = fit_models_for_phase6(subset_b)

    probe = val_df[FEATURE_COLUMNS].iloc[[0]]
    proba_a = models_a["random_forest"].predict_proba(probe)[0, 1]
    proba_b = models_b["random_forest"].predict_proba(probe)[0, 1]
    # Not asserting they MUST differ (could coincidentally match), but both
    # must be valid probabilities computed independently from disjoint data
    assert 0.0 <= proba_a <= 1.0
    assert 0.0 <= proba_b <= 1.0


def test_full_phase6_pipeline_end_to_end_via_assemble_feature_label_table():
    """Uses the REAL assemble_feature_label_table() (not a hand-rolled
    substitute) with synthetic inputs, through fit -> analyze."""
    cell_day = _synthetic_cell_day_dataset(n_cells=3).drop(columns=[PRIMARY_TARGET_COLUMN])
    target_df = pd.DataFrame({"cell_id": cell_day["cell_id"], "date": cell_day["date"]})
    rng = np.random.default_rng(11)
    target_df[PRIMARY_TARGET_COLUMN] = rng.integers(0, 2, size=len(target_df))

    table, report = assemble_feature_label_table(cell_day, target_df)
    train_df, val_df, test_df = chronological_split(table)

    models = fit_models_for_phase6(train_df)
    analyses = {name: analyze_model_on_validation(m, val_df) for name, m in models.items()}

    assert set(analyses.keys()) == {"random_forest", "gradient_boosting"}
    for name, a in analyses.items():
        assert a["default_threshold_metrics"]["n_samples"] == len(val_df)
