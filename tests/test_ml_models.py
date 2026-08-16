"""
Unit tests for ml/models.py (Phase 5), using synthetic data only.
"""
import numpy as np
import pandas as pd
import pytest

from ml.dataset import FEATURE_COLUMNS
from ml.models import RandomForestModel, GradientBoostingModel


def _synthetic_features_and_target(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    X["fire_count"] = rng.integers(0, 3, size=n)
    X["fire_count_3d"] = rng.integers(0, 5, size=n)
    X["fire_count_7d"] = rng.integers(0, 8, size=n)
    X["fire_count_14d"] = rng.integers(0, 12, size=n)
    X["fire_count_30d"] = rng.integers(0, 20, size=n)
    X["days_since_last_detection"] = rng.integers(0, 91, size=n)

    base_prob = 0.10 + 0.15 * (X["fire_count_7d"] > 3)
    y = (rng.random(n) < base_prob).astype(int)

    # columns that must never reach the underlying sklearn model
    X["cell_id"] = [f"C{i % 5}" for i in range(n)]
    X["date"] = pd.date_range("2018-01-01", periods=n)

    return X, pd.Series(y)


# ---------------------------------------------------------------------------
# RandomForestModel
# ---------------------------------------------------------------------------

def test_random_forest_uses_balanced_class_weight():
    model = RandomForestModel()
    assert model.model.class_weight == "balanced"


def test_random_forest_fits_and_predicts():
    X, y = _synthetic_features_and_target(n=300, seed=1)
    model = RandomForestModel().fit(X, y)

    preds = model.predict(X)
    proba = model.predict_proba(X)

    assert len(preds) == len(X)
    assert set(np.unique(preds)) <= {0, 1}
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_random_forest_excludes_cell_id_and_date():
    """Even though X carries cell_id/date, the fitted forest's feature
    count must equal len(FEATURE_COLUMNS) -- proves those columns never
    reached sklearn's fit() call."""
    X, y = _synthetic_features_and_target(n=300, seed=2)
    model = RandomForestModel().fit(X, y)
    assert model.model.n_features_in_ == len(FEATURE_COLUMNS)


def test_random_forest_raises_before_fit():
    X, _ = _synthetic_features_and_target(n=5)
    model = RandomForestModel()
    with pytest.raises(RuntimeError):
        model.predict(X)
    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_random_forest_raises_on_missing_feature_column():
    X, y = _synthetic_features_and_target(n=50)
    X = X.drop(columns=["wind_speed"])
    model = RandomForestModel()
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_random_forest_is_deterministic_with_fixed_random_state():
    X, y = _synthetic_features_and_target(n=200, seed=3)
    model_a = RandomForestModel(random_state=42).fit(X, y)
    model_b = RandomForestModel(random_state=42).fit(X, y)
    np.testing.assert_array_equal(model_a.predict(X), model_b.predict(X))


# ---------------------------------------------------------------------------
# GradientBoostingModel
# ---------------------------------------------------------------------------

def test_gradient_boosting_uses_balanced_class_weight():
    """Confirms class_weight='balanced' is actually set on the underlying
    model -- checking the real attribute, not assuming support exists."""
    model = GradientBoostingModel()
    assert model.model.class_weight == "balanced"


def test_gradient_boosting_fits_and_predicts():
    X, y = _synthetic_features_and_target(n=300, seed=4)
    model = GradientBoostingModel().fit(X, y)

    preds = model.predict(X)
    proba = model.predict_proba(X)

    assert len(preds) == len(X)
    assert set(np.unique(preds)) <= {0, 1}
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_gradient_boosting_excludes_cell_id_and_date():
    X, y = _synthetic_features_and_target(n=300, seed=6)
    model = GradientBoostingModel().fit(X, y)
    assert model.model.n_features_in_ == len(FEATURE_COLUMNS)


def test_gradient_boosting_raises_before_fit():
    X, _ = _synthetic_features_and_target(n=5)
    model = GradientBoostingModel()
    with pytest.raises(RuntimeError):
        model.predict(X)
    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_gradient_boosting_raises_on_missing_feature_column():
    X, y = _synthetic_features_and_target(n=50)
    X = X.drop(columns=["precipitation"])
    model = GradientBoostingModel()
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_gradient_boosting_is_deterministic_with_fixed_random_state():
    X, y = _synthetic_features_and_target(n=200, seed=7)
    model_a = GradientBoostingModel(random_state=42).fit(X, y)
    model_b = GradientBoostingModel(random_state=42).fit(X, y)
    np.testing.assert_array_equal(model_a.predict(X), model_b.predict(X))


# ---------------------------------------------------------------------------
# fit_tree_models (ml/train_models.py)
# ---------------------------------------------------------------------------

def test_fit_tree_models_returns_both_models():
    from ml.train_models import fit_tree_models

    X, y = _synthetic_features_and_target(n=200, seed=8)
    train_df = X.copy()
    train_df["future_fire_7d"] = y

    models = fit_tree_models(train_df)

    assert set(models.keys()) == {"random_forest", "gradient_boosting"}
    assert models["random_forest"].model.n_features_in_ == len(FEATURE_COLUMNS)
    assert models["gradient_boosting"].model.n_features_in_ == len(FEATURE_COLUMNS)
