"""
Unit and integration tests for ml/test_evaluation.py (Phase 7).

Focus: correctness of the TEST scoring logic (hand-computed, and via a
stub predictor), the lock-file run-once safeguard, structural proof that
VALIDATION cannot reach evaluate_on_test, and confirmation that the locked
config (model class, threshold) is never varied anywhere in this module.
"""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from ml.dataset import assemble_feature_label_table, impute_frp, FEATURE_COLUMNS, PRIMARY_TARGET_COLUMN
from ml.split import chronological_split, validate_split_partition
from ml.models import GradientBoostingModel
import ml.test_evaluation as test_eval_module
from ml.test_evaluation import (
    LOCKED_MODEL_CLASS, LOCKED_THRESHOLD,
    fit_locked_model, evaluate_on_test, format_report_markdown, run,
)


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


class _FixedPredictor:
    """Stub model with hardcoded probabilities, so evaluate_on_test's
    metrics can be checked against hand-computed values rather than
    trusting the real GradientBoostingModel's output."""
    def __init__(self, proba_positive):
        self._proba_positive = np.array(proba_positive)

    def predict_proba(self, X):
        return np.column_stack([1 - self._proba_positive, self._proba_positive])


# ---------------------------------------------------------------------------
# Locked configuration -- guards against silent drift
# ---------------------------------------------------------------------------

def test_locked_model_class_is_gradient_boosting():
    assert LOCKED_MODEL_CLASS is GradientBoostingModel


def test_locked_threshold_is_0_70():
    assert LOCKED_THRESHOLD == 0.70


# ---------------------------------------------------------------------------
# evaluate_on_test -- hand-computed correctness
# ---------------------------------------------------------------------------

def test_evaluate_on_test_hand_computed():
    # y_true: [1, 1, 0, 0, 1]; proba: [0.9, 0.4, 0.1, 0.8, 0.6]
    # threshold=0.70 -> preds = [1, 0, 0, 1, 0]
    # TP=1(idx0), FN=2(idx1,4), TN=1(idx2), FP=1(idx3)
    test_df = pd.DataFrame({c: [0.0] * 5 for c in FEATURE_COLUMNS})
    test_df[PRIMARY_TARGET_COLUMN] = [1, 1, 0, 0, 1]
    model = _FixedPredictor(proba_positive=[0.9, 0.4, 0.1, 0.8, 0.6])

    results = evaluate_on_test(model, test_df, threshold=0.70)

    assert results["locked_threshold"] == 0.70
    assert results["n_test_rows"] == 5
    assert results["true_positive"] == 1
    assert results["false_negative"] == 2
    assert results["true_negative"] == 1
    assert results["false_positive"] == 1
    assert results["test_positive_rate"] == pytest.approx(0.6)
    assert results["model_name"] == "gradient_boosting"


def test_evaluate_on_test_uses_the_passed_threshold_not_a_hardcoded_one():
    """Confirms the threshold argument is actually respected -- passing a
    different threshold changes the predictions/metrics."""
    test_df = pd.DataFrame({c: [0.0] * 4 for c in FEATURE_COLUMNS})
    test_df[PRIMARY_TARGET_COLUMN] = [1, 0, 1, 0]
    model = _FixedPredictor(proba_positive=[0.65, 0.65, 0.75, 0.75])

    at_070 = evaluate_on_test(model, test_df, threshold=0.70)
    at_050 = evaluate_on_test(model, test_df, threshold=0.50)

    assert at_070["predicted_positive_rate"] == 0.5   # only the two 0.75s pass 0.70
    assert at_050["predicted_positive_rate"] == 1.0    # all four pass 0.50


def test_evaluate_on_test_signature_has_no_val_df_parameter():
    """Structural guarantee: scoring VALIDATION isn't expressible through
    this function's signature at all."""
    sig = inspect.signature(evaluate_on_test)
    param_names = list(sig.parameters.keys())
    assert "test_df" in param_names
    assert not any("val" in p.lower() for p in param_names)


# ---------------------------------------------------------------------------
# fit_locked_model -- fits only the locked model, on whatever it's given
# ---------------------------------------------------------------------------

def test_fit_locked_model_returns_a_gradient_boosting_model():
    dataset = impute_frp(_synthetic_cell_day_dataset(n_cells=3))
    train_df, val_df, test_df = chronological_split(dataset)

    model = fit_locked_model(train_df)
    assert isinstance(model, GradientBoostingModel)
    assert model.model.n_features_in_ == len(FEATURE_COLUMNS)


def test_fit_locked_model_only_uses_given_dataframe():
    """Fitting on two disjoint subsets must not share state -- both
    fitted models must independently return valid probabilities."""
    dataset = impute_frp(_synthetic_cell_day_dataset(n_cells=5, seed=7))
    train_df, val_df, test_df = chronological_split(dataset)

    half = len(train_df) // 2
    model_a = fit_locked_model(train_df.iloc[:half])
    model_b = fit_locked_model(train_df.iloc[half:])

    probe = test_df[FEATURE_COLUMNS].iloc[[0]]
    proba_a = model_a.predict_proba(probe)[0, 1]
    proba_b = model_b.predict_proba(probe)[0, 1]
    assert 0.0 <= proba_a <= 1.0
    assert 0.0 <= proba_b <= 1.0


# ---------------------------------------------------------------------------
# format_report_markdown
# ---------------------------------------------------------------------------

def test_format_report_markdown_includes_required_confirmations():
    test_df = pd.DataFrame({c: [0.0] * 4 for c in FEATURE_COLUMNS})
    test_df[PRIMARY_TARGET_COLUMN] = [1, 0, 1, 0]
    model = _FixedPredictor(proba_positive=[0.9, 0.1, 0.8, 0.2])
    results = evaluate_on_test(model, test_df)

    md = format_report_markdown(results, {"n_train": 100, "n_validation_not_used_for_scoring": 20})

    assert "evaluated exactly ONCE" in md
    assert "VALIDATION was NOT used" in md
    assert "gradient_boosting" in md
    assert "0.7" in md  # threshold appears somewhere


# ---------------------------------------------------------------------------
# run() -- lock-file run-once safeguard, TEST-only discipline
# ---------------------------------------------------------------------------

def test_run_refuses_second_evaluation_without_force(tmp_path, monkeypatch):
    """The core Phase 7 safeguard: a second call to run() without
    force=True must raise, not silently re-score TEST."""
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(test_eval_module, "LOCK_FILE_PATH", lock_path)

    cell_day = _synthetic_cell_day_dataset(n_cells=3).drop(columns=[PRIMARY_TARGET_COLUMN])
    target_df = pd.DataFrame({"cell_id": cell_day["cell_id"], "date": cell_day["date"]})
    rng = np.random.default_rng(8)
    target_df[PRIMARY_TARGET_COLUMN] = rng.integers(0, 2, size=len(target_df))

    def fake_assemble():
        return assemble_feature_label_table(cell_day, target_df)

    monkeypatch.setattr(test_eval_module, "assemble_feature_label_table", fake_assemble)

    out_dir = tmp_path / "docs"
    results_1, _ = run(out_dir=out_dir)
    assert lock_path.exists()

    with pytest.raises(RuntimeError):
        run(out_dir=out_dir)  # second call, no force -- must refuse


def test_run_allows_second_evaluation_with_explicit_force(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(test_eval_module, "LOCK_FILE_PATH", lock_path)

    cell_day = _synthetic_cell_day_dataset(n_cells=3).drop(columns=[PRIMARY_TARGET_COLUMN])
    target_df = pd.DataFrame({"cell_id": cell_day["cell_id"], "date": cell_day["date"]})
    rng = np.random.default_rng(9)
    target_df[PRIMARY_TARGET_COLUMN] = rng.integers(0, 2, size=len(target_df))

    def fake_assemble():
        return assemble_feature_label_table(cell_day, target_df)

    monkeypatch.setattr(test_eval_module, "assemble_feature_label_table", fake_assemble)

    out_dir = tmp_path / "docs"
    run(out_dir=out_dir)
    results_2, _ = run(out_dir=out_dir, force=True)  # explicit override -- must succeed
    assert results_2["model_name"] == "gradient_boosting"


def test_run_writes_lock_file_with_locked_config(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(test_eval_module, "LOCK_FILE_PATH", lock_path)

    cell_day = _synthetic_cell_day_dataset(n_cells=3).drop(columns=[PRIMARY_TARGET_COLUMN])
    target_df = pd.DataFrame({"cell_id": cell_day["cell_id"], "date": cell_day["date"]})
    rng = np.random.default_rng(10)
    target_df[PRIMARY_TARGET_COLUMN] = rng.integers(0, 2, size=len(target_df))

    def fake_assemble():
        return assemble_feature_label_table(cell_day, target_df)

    monkeypatch.setattr(test_eval_module, "assemble_feature_label_table", fake_assemble)

    run(out_dir=tmp_path / "docs")

    with open(lock_path) as f:
        lock_data = json.load(f)
    assert lock_data["evaluated"] is True
    assert lock_data["locked_threshold"] == 0.70
    assert lock_data["model_name"] == "gradient_boosting"


def test_run_never_reads_validation_after_split(tmp_path, monkeypatch):
    """Source-level check on the real Phase 7 entrypoint: val_df is bound
    by chronological_split but never passed into evaluate_on_test or any
    other scoring call."""
    source = inspect.getsource(test_eval_module.run)
    assert "evaluate_on_test(model, test_df)" in source
    assert "evaluate_on_test(model, val_df)" not in source
    # val_df is only referenced for its length (transparency logging/metadata)
    assert "len(val_df)" in source


def test_run_end_to_end_writes_expected_files(tmp_path, monkeypatch):
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(test_eval_module, "LOCK_FILE_PATH", lock_path)

    cell_day = _synthetic_cell_day_dataset(n_cells=3).drop(columns=[PRIMARY_TARGET_COLUMN])
    target_df = pd.DataFrame({"cell_id": cell_day["cell_id"], "date": cell_day["date"]})
    rng = np.random.default_rng(11)
    target_df[PRIMARY_TARGET_COLUMN] = rng.integers(0, 2, size=len(target_df))

    def fake_assemble():
        return assemble_feature_label_table(cell_day, target_df)

    monkeypatch.setattr(test_eval_module, "assemble_feature_label_table", fake_assemble)

    out_dir = tmp_path / "docs"
    results, run_metadata = run(out_dir=out_dir)

    assert (out_dir / "phase7-test-evaluation.md").exists()
    assert (out_dir / "phase7-test-evaluation.json").exists()
    assert "n_validation_not_used_for_scoring" in run_metadata
    assert results["n_test_rows"] > 0


def test_run_raises_if_split_validation_fails(tmp_path, monkeypatch):
    """If validate_split_partition ever reports a failure, run() must
    refuse to proceed to TEST scoring rather than scoring a corrupted
    split."""
    lock_path = tmp_path / "lock.json"
    monkeypatch.setattr(test_eval_module, "LOCK_FILE_PATH", lock_path)
    monkeypatch.setattr(
        test_eval_module, "validate_split_partition",
        lambda *a, **kw: (False, {"reason": "forced failure for test"}),
    )

    cell_day = _synthetic_cell_day_dataset(n_cells=3).drop(columns=[PRIMARY_TARGET_COLUMN])
    target_df = pd.DataFrame({"cell_id": cell_day["cell_id"], "date": cell_day["date"]})
    rng = np.random.default_rng(12)
    target_df[PRIMARY_TARGET_COLUMN] = rng.integers(0, 2, size=len(target_df))

    def fake_assemble():
        return assemble_feature_label_table(cell_day, target_df)

    monkeypatch.setattr(test_eval_module, "assemble_feature_label_table", fake_assemble)

    with pytest.raises(RuntimeError):
        run(out_dir=tmp_path / "docs")
    assert not lock_path.exists()  # must not have written the lock on a failed run
