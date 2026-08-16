"""
ml/features.py -- shared feature-selection guard.

Single source of truth for how every model (baseline or Phase 5 tree-based)
reads its feature matrix from a caller's DataFrame. Selects EXACTLY
ml.dataset.FEATURE_COLUMNS in a fixed order and raises if any expected
feature is missing.

This is the ONE place feature selection happens across ml/baseline.py and
ml/models.py -- every model routes through it rather than reading raw X, so
cell_id/date exclusion is structural, not a convention that could be
forgotten in one model but not another.
"""
from ml.dataset import FEATURE_COLUMNS


def select_features(X):
    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    return X[FEATURE_COLUMNS]
