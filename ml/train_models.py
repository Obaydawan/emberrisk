"""
ml/train_models.py -- Phase 5: fits the tree-based models on TRAIN only.

Mirrors ml.train.fit_baselines() exactly, as a separate file rather than
appended to ml/train.py, so Phase 4's tested entrypoint stays untouched.
Does NOT evaluate on validation/test and does NOT tune hyperparameters --
same scope discipline as Phase 4 Step 4.
"""
import logging

from ml.dataset import FEATURE_COLUMNS, PRIMARY_TARGET_COLUMN
from ml.models import RandomForestModel, GradientBoostingModel

logger = logging.getLogger("emberrisk.ml.train_models")


def fit_tree_models(train_df, target_column=PRIMARY_TARGET_COLUMN):
    """Fits RandomForestModel and GradientBoostingModel on TRAIN ONLY.
    Returns a dict of fitted models. Callers must not pass anything but the
    train split here -- same convention as ml.train.fit_baselines()."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[target_column]

    models = {
        "random_forest": RandomForestModel(),
        "gradient_boosting": GradientBoostingModel(),
    }
    for name, model in models.items():
        logger.info("Fitting %s on %d TRAIN rows...", name, len(X_train))
        model.fit(X_train, y_train)

    return models
