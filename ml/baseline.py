"""
ml/baseline.py -- Phase 4 Step 4: baseline models.

Three models, increasing in sophistication, all sharing one contract:
  fit(X, y) / predict(X) / predict_proba(X)
so ml/train.py (and later evaluation code) can treat them uniformly.

Every model in this module reads features through _select_features(), which
selects EXACTLY ml.dataset.FEATURE_COLUMNS in a fixed order -- this is the
single enforcement point that guarantees cell_id/date (or any other stray
column X happens to carry) never reaches a model, no matter what extra
columns the caller's DataFrame contains.

No threshold tuning, no hyperparameter search, no calibration -- explicitly
out of scope for this step.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.dataset import FEATURE_COLUMNS


def _select_features(X):
    """Selects exactly FEATURE_COLUMNS from X, in a fixed order. Raises if
    any expected feature is missing. This is the ONLY place feature
    selection happens in this module -- every model routes through it
    rather than reading raw X, so cell_id/date exclusion is structural,
    not a convention that could be forgotten in one model but not another."""
    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    return X[FEATURE_COLUMNS]


class MajorityClassBaseline:
    """Always predicts the majority class observed in TRAIN. The floor any
    real model must beat -- ignores X entirely by design, so a model that
    can't outperform this has learned nothing useful from the features."""

    def __init__(self):
        self.majority_class_ = None
        self.positive_rate_ = None

    def fit(self, X, y):
        y = pd.Series(y)
        self.positive_rate_ = float(y.mean())
        self.majority_class_ = int(y.mode().iloc[0])
        return self

    def predict(self, X):
        _select_features(X)  # validates shape/columns even though X is otherwise unused
        return np.full(len(X), self.majority_class_, dtype=int)

    def predict_proba(self, X):
        _select_features(X)
        p = self.positive_rate_
        return np.tile([1 - p, p], (len(X), 1))


class PersistenceBaseline:
    """Predicts fire will continue if there's a qualifying detection on the
    SAME day T (fire_count > 0). A domain-informed second floor, distinct
    from the majority-class floor: 'is there a fire right now' is genuinely
    informative for 'will there be one in the next 7 days', so a real model
    should be expected to beat this too, not just the majority class."""

    def __init__(self, indicator_column="fire_count"):
        self.indicator_column = indicator_column
        self._fitted = False

    def fit(self, X, y=None):
        if self.indicator_column not in X.columns:
            raise ValueError(f"Expected indicator column {self.indicator_column!r} not found in X")
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("PersistenceBaseline must be fit before predict")
        _select_features(X)  # enforce the same column contract as other models
        return (X[self.indicator_column] > 0).astype(int).to_numpy()

    def predict_proba(self, X):
        preds = self.predict(X).astype(float)
        # Deliberately degenerate/uncalibrated probabilities (0.0 or 1.0) --
        # this is a heuristic, not a calibrated model. Calibration/threshold
        # tuning is out of scope for this baseline step.
        return np.column_stack([1 - preds, preds])


class LogisticRegressionBaseline:
    """First real ML model: scikit-learn LogisticRegression with
    class_weight='balanced' to address the ~18.45% positive rate. No
    threshold tuning, no hyperparameter search beyond class_weight -- plain
    defaults otherwise, per the Step 4 scope. Trains on whatever X/y it's
    given; it is ml/train.py's responsibility to only ever pass the TRAIN
    split here."""

    def __init__(self, random_state=42, max_iter=1000):
        self.model = LogisticRegression(
            class_weight="balanced", random_state=random_state, max_iter=max_iter,
        )
        self._fitted = False

    def fit(self, X, y):
        X_selected = _select_features(X)
        self.model.fit(X_selected, y)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("LogisticRegressionBaseline must be fit before predict")
        return self.model.predict(_select_features(X))

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("LogisticRegressionBaseline must be fit before predict_proba")
        return self.model.predict_proba(_select_features(X))
