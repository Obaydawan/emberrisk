"""
Unit tests for ml/evaluate.py (Phase 4 Step 5), using synthetic data only.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from ml.dataset import FEATURE_COLUMNS
from ml.baseline import MajorityClassBaseline, LogisticRegressionBaseline
from ml.train import fit_baselines
from ml.evaluate import evaluate_model, evaluate_all_baselines, format_results_markdown
import ml.evaluate as evaluate_module


def _synthetic_features_and_target(n=200, seed=0, positive_rate_hint=0.2):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    X["fire_count"] = rng.integers(0, 3, size=n)
    X["fire_count_3d"] = rng.integers(0, 5, size=n)
    X["fire_count_7d"] = rng.integers(0, 8, size=n)
    X["fire_count_14d"] = rng.integers(0, 12, size=n)
    X["fire_count_30d"] = rng.integers(0, 20, size=n)
    X["days_since_last_detection"] = rng.integers(0, 91, size=n)

    base_prob = positive_rate_hint * 0.5 + positive_rate_hint * (X["fire_count_7d"] > 3)
    y = (rng.random(n) < base_prob).astype(int)
    return X, pd.Series(y)


# ---------------------------------------------------------------------------
# evaluate_model -- correctness against a known, hand-computable case
# ---------------------------------------------------------------------------

class _FixedPredictor:
    """A stub model with hardcoded predictions/probabilities, so metrics can
    be checked against values computed by hand rather than trusting another
    library call."""
    def __init__(self, preds, proba_positive):
        self._preds = np.array(preds)
        self._proba_positive = np.array(proba_positive)

    def predict(self, X):
        return self._preds

    def predict_proba(self, X):
        return np.column_stack([1 - self._proba_positive, self._proba_positive])


def test_evaluate_model_confusion_matrix_hand_computed():
    # y_true:  [1, 1, 0, 0, 1]
    # y_pred:  [1, 0, 0, 1, 1]
    # -> TP=2 (idx0,4), FN=1 (idx1), TN=1 (idx2), FP=1 (idx3)
    y = [1, 1, 0, 0, 1]
    model = _FixedPredictor(preds=[1, 0, 0, 1, 1], proba_positive=[0.9, 0.4, 0.1, 0.6, 0.8])
    X = pd.DataFrame(index=range(5))

    metrics = evaluate_model(model, X, y, model_name="fixed")

    cm = metrics["confusion_matrix"]
    assert cm == {"true_negative": 1, "false_positive": 1, "false_negative": 1, "true_positive": 2}
    # precision = TP/(TP+FP) = 2/3, recall = TP/(TP+FN) = 2/3
    assert metrics["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["n_positive"] == 3
    assert metrics["positive_rate"] == pytest.approx(0.6)


def test_evaluate_model_perfect_predictions():
    y = [0, 0, 1, 1]
    model = _FixedPredictor(preds=[0, 0, 1, 1], proba_positive=[0.05, 0.1, 0.9, 0.95])
    metrics = evaluate_model(model, pd.DataFrame(index=range(4)), y, model_name="perfect")

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["false_positive"] == 0
    assert metrics["confusion_matrix"]["false_negative"] == 0


def test_evaluate_model_handles_single_class_gracefully():
    """When y has only one class (edge case, shouldn't crash), ROC-AUC/PR-AUC
    are undefined -- must return None rather than raising or fabricating a
    number."""
    y = [0, 0, 0, 0]
    model = _FixedPredictor(preds=[0, 0, 0, 0], proba_positive=[0.1, 0.2, 0.1, 0.3])
    metrics = evaluate_model(model, pd.DataFrame(index=range(4)), y, model_name="degenerate")

    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["precision"] == 0.0  # zero_division=0, no positives predicted or true


# ---------------------------------------------------------------------------
# evaluate_all_baselines -- structural guarantee: VALIDATION only
# ---------------------------------------------------------------------------

def test_evaluate_all_baselines_end_to_end_with_fitted_models():
    X_train, y_train = _synthetic_features_and_target(n=300, seed=1)
    X_val, y_val = _synthetic_features_and_target(n=100, seed=2)

    train_df = X_train.copy()
    train_df["future_fire_7d"] = y_train
    val_df = X_val.copy()
    val_df["future_fire_7d"] = y_val

    models = fit_baselines(train_df)
    results = evaluate_all_baselines(models, val_df)

    assert set(results.keys()) == {"majority_class", "persistence", "logistic_regression"}
    for name, m in results.items():
        assert m["n_samples"] == 100
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0


def test_evaluate_all_baselines_signature_has_no_test_split_parameter():
    """Structural guarantee: the function signature itself only accepts
    val_df, with no test_df parameter at all -- calling it on TEST isn't
    just discouraged, it's not expressible through this API."""
    sig = inspect.signature(evaluate_all_baselines)
    param_names = list(sig.parameters.keys())
    assert "val_df" in param_names
    assert "test_df" not in param_names
    assert not any("test" in p.lower() for p in param_names)


def test_run_function_never_passes_test_df_to_evaluation():
    """Source-level check on the real orchestration function: confirms
    evaluate_all_baselines is only ever called with val_df, never test_df,
    in the actual pipeline entrypoint."""
    source = inspect.getsource(evaluate_module.run)
    assert "evaluate_all_baselines(models, val_df)" in source
    assert "evaluate_all_baselines(models, test_df)" not in source


# ---------------------------------------------------------------------------
# format_results_markdown
# ---------------------------------------------------------------------------

def test_format_results_markdown_includes_all_models_and_disclaimer():
    X, y = _synthetic_features_and_target(n=50, seed=5)
    df = X.copy()
    df["future_fire_7d"] = y
    models = fit_baselines(df)
    results = evaluate_all_baselines(models, df)

    md = format_results_markdown(results, run_metadata={"n_train": 100})

    assert "majority_class" in md
    assert "persistence" in md
    assert "logistic_regression" in md
    assert "VALIDATION" in md
    assert "TEST was not evaluated" in md
    assert "| Precision | Recall | F1 | ROC-AUC | PR-AUC |" in md
