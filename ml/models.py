"""
ml/models.py -- Phase 5: stronger tree-based models.

Same fit/predict/predict_proba contract as ml/baseline.py's classes, so
ml.evaluate.evaluate_all_baselines() (unmodified, reused as-is) can treat
these identically to the Phase 4 baselines. Both models route feature
selection through ml.features.select_features() -- the same shared guard
ml/baseline.py now uses -- so cell_id/date exclusion is enforced identically
across every model in the project, not just within one file.

Class imbalance handling: both models use scikit-learn's native
class_weight="balanced" directly. Earlier planning assumed
HistGradientBoostingClassifier didn't expose class_weight (true in some
older sklearn versions) and planned a sample_weight-based workaround for
it -- checking the actual installed version (sklearn 1.8.0) showed
class_weight IS supported here, so the simpler, consistent native approach
is used for both models instead of carrying an unnecessary workaround.
No hyperparameter search in either model -- fixed, hardware-appropriate
configs only, chosen for reasonable training time/memory on an 8GB/dual-core
machine, not tuned for performance.
"""
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from ml.features import select_features


class RandomForestModel:
    """class_weight='balanced' handled natively by scikit-learn."""

    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X, y):
        X_selected = select_features(X)
        self.model.fit(X_selected, y)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("RandomForestModel must be fit before predict")
        return self.model.predict(select_features(X))

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("RandomForestModel must be fit before predict_proba")
        return self.model.predict_proba(select_features(X))


class GradientBoostingModel:
    """scikit-learn's HistGradientBoostingClassifier -- histogram-based,
    same algorithmic family as LightGBM, built in (no new dependency),
    designed to be fast/memory-light at this data scale versus plain
    GradientBoostingClassifier (which isn't parallelized).

    class_weight='balanced' is supported natively in the installed sklearn
    version (1.8.0) and used directly here, same as RandomForestModel."""

    def __init__(self, random_state=42):
        self.model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, X, y):
        X_selected = select_features(X)
        self.model.fit(X_selected, y)
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("GradientBoostingModel must be fit before predict")
        return self.model.predict(select_features(X))

    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("GradientBoostingModel must be fit before predict_proba")
        return self.model.predict_proba(select_features(X))
