"""
Unit tests for ml/baseline.py and ml/train.py (Phase 4 Step 4), using
synthetic data only -- no real Parquet files needed.
"""
import numpy as np
import pandas as pd
import pytest

from ml.dataset import FEATURE_COLUMNS
from ml.baseline import (
    MajorityClassBaseline, PersistenceBaseline, LogisticRegressionBaseline,
    _select_features,
)
from ml.train import fit_baselines


def _synthetic_features_and_target(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    # fire_count/fire_count_*d/days_since_last_detection should look count-like
    X["fire_count"] = rng.integers(0, 3, size=n)
    X["fire_count_3d"] = rng.integers(0, 5, size=n)
    X["fire_count_7d"] = rng.integers(0, 8, size=n)
    X["fire_count_14d"] = rng.integers(0, 12, size=n)
    X["fire_count_30d"] = rng.integers(0, 20, size=n)
    X["days_since_last_detection"] = rng.integers(0, 91, size=n)

    # target loosely correlated with fire_count so LogisticRegression has
    # something learnable, but still imbalanced like the real data (~18%)
    base_prob = 0.10 + 0.15 * (X["fire_count_7d"] > 3)
    y = (rng.random(n) < base_prob).astype(int)

    # extra columns a real caller's DataFrame would carry, which MUST be
    # ignored by every model in this module
    X["cell_id"] = [f"C{i % 5}" for i in range(n)]
    X["date"] = pd.date_range("2018-01-01", periods=n)

    return X, pd.Series(y)


# ---------------------------------------------------------------------------
# _select_features -- the shared column-selection contract
# ---------------------------------------------------------------------------

def test_select_features_returns_only_feature_columns():
    X, _ = _synthetic_features_and_target(n=10)
    selected = _select_features(X)
    assert list(selected.columns) == FEATURE_COLUMNS
    assert "cell_id" not in selected.columns
    assert "date" not in selected.columns


def test_select_features_raises_on_missing_column():
    X, _ = _synthetic_features_and_target(n=10)
    X = X.drop(columns=["wind_speed"])
    with pytest.raises(ValueError):
        _select_features(X)


# ---------------------------------------------------------------------------
# MajorityClassBaseline
# ---------------------------------------------------------------------------

def test_majority_class_predicts_constant_value():
    X, _ = _synthetic_features_and_target(n=50)
    y = pd.Series([0] * 40 + [1] * 10)  # 20% positive, majority = 0

    model = MajorityClassBaseline().fit(X, y)
    preds = model.predict(X)

    assert model.majority_class_ == 0
    assert (preds == 0).all()


def test_majority_class_predict_proba_matches_train_positive_rate():
    X, _ = _synthetic_features_and_target(n=50)
    y = pd.Series([0] * 35 + [1] * 15)  # 30% positive

    model = MajorityClassBaseline().fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (50, 2)
    assert np.allclose(proba[:, 1], 0.30)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_majority_class_ignores_features_entirely():
    """Two wildly different X inputs with the same y must produce identical
    predictions -- proves the model genuinely ignores X, as documented."""
    X1, y = _synthetic_features_and_target(n=30, seed=1)
    X2, _ = _synthetic_features_and_target(n=30, seed=99)  # different features, same length

    model = MajorityClassBaseline().fit(X1, y)
    preds1 = model.predict(X1)
    preds2 = model.predict(X2)
    assert (preds1 == preds2).all()


# ---------------------------------------------------------------------------
# PersistenceBaseline
# ---------------------------------------------------------------------------

def test_persistence_predicts_based_on_current_fire_count():
    X, y = _synthetic_features_and_target(n=20)
    X["fire_count"] = [0, 1, 0, 2, 0, 0, 3, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0]

    model = PersistenceBaseline().fit(X, y)
    preds = model.predict(X)

    expected = (X["fire_count"] > 0).astype(int).to_numpy()
    assert (preds == expected).all()


def test_persistence_raises_if_indicator_column_missing():
    X, y = _synthetic_features_and_target(n=10)
    X = X.drop(columns=["fire_count"])
    model = PersistenceBaseline()
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_persistence_predict_proba_is_degenerate_but_valid():
    X, y = _synthetic_features_and_target(n=10)
    X["fire_count"] = [0, 1] * 5

    model = PersistenceBaseline().fit(X, y)
    proba = model.predict_proba(X)

    assert proba.shape == (10, 2)
    assert set(np.unique(proba)) <= {0.0, 1.0}  # deliberately uncalibrated
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_persistence_requires_fit_before_predict():
    X, _ = _synthetic_features_and_target(n=5)
    model = PersistenceBaseline()
    with pytest.raises(RuntimeError):
        model.predict(X)


# ---------------------------------------------------------------------------
# LogisticRegressionBaseline
# ---------------------------------------------------------------------------

def test_logistic_regression_uses_balanced_class_weight():
    model = LogisticRegressionBaseline()
    assert model.model.class_weight == "balanced"


def test_logistic_regression_fits_and_predicts():
    X, y = _synthetic_features_and_target(n=300, seed=7)
    model = LogisticRegressionBaseline().fit(X, y)

    preds = model.predict(X)
    proba = model.predict_proba(X)

    assert len(preds) == len(X)
    assert set(np.unique(preds)) <= {0, 1}
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_logistic_regression_only_trains_on_feature_columns():
    """The core requirement-10 guarantee: even though X carries cell_id and
    date, the fitted model's coefficient count must equal len(FEATURE_COLUMNS),
    proving those columns never reached sklearn's fit() call."""
    X, y = _synthetic_features_and_target(n=300, seed=3)
    model = LogisticRegressionBaseline().fit(X, y)
    assert model.model.coef_.shape[1] == len(FEATURE_COLUMNS)


def test_logistic_regression_raises_before_fit():
    X, _ = _synthetic_features_and_target(n=5)
    model = LogisticRegressionBaseline()
    with pytest.raises(RuntimeError):
        model.predict(X)
    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_logistic_regression_raises_on_missing_feature_column():
    X, y = _synthetic_features_and_target(n=50)
    X = X.drop(columns=["precipitation"])
    model = LogisticRegressionBaseline()
    with pytest.raises(ValueError):
        model.fit(X, y)


# ---------------------------------------------------------------------------
# fit_baselines (ml/train.py) -- train-only contract
# ---------------------------------------------------------------------------

def test_fit_baselines_returns_all_three_models():
    X, y = _synthetic_features_and_target(n=200, seed=11)
    train_df = X.copy()
    train_df["future_fire_7d"] = y

    models = fit_baselines(train_df)

    assert set(models.keys()) == {"majority_class", "persistence", "logistic_regression"}
    assert models["majority_class"].majority_class_ is not None
    assert models["logistic_regression"].model.coef_.shape[1] == len(FEATURE_COLUMNS)


def test_fit_baselines_only_uses_given_dataframe_not_a_hidden_split():
    """fit_baselines has no concept of validation/test -- it fits on exactly
    the rows it's handed. Fitting on two different subsets must produce two
    different majority-class positive rates when the underlying rates
    differ, proving no hidden global state/leakage across calls."""
    X, y = _synthetic_features_and_target(n=400, seed=21)
    df = X.copy()
    df["future_fire_7d"] = y

    subset_a = df.iloc[:200]
    subset_b = df.iloc[200:]

    models_a = fit_baselines(subset_a)
    models_b = fit_baselines(subset_b)

    rate_a = models_a["majority_class"].positive_rate_
    rate_b = models_b["majority_class"].positive_rate_
    # They need not differ (random data might coincide), but each must
    # exactly match its OWN subset's true rate -- that's the real assertion.
    assert rate_a == pytest.approx(subset_a["future_fire_7d"].mean())
    assert rate_b == pytest.approx(subset_b["future_fire_7d"].mean())
