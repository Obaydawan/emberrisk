"""
ml/train.py -- Phase 4 Step 4: fits baseline models on TRAIN only.

Does NOT evaluate on validation/test (that's Phase 4 Step 5) and does NOT
tune thresholds or hyperparameters (explicitly out of scope for this step).
"""
import logging

from ml.dataset import assemble_feature_label_table, FEATURE_COLUMNS, PRIMARY_TARGET_COLUMN
from ml.split import chronological_split
from ml.baseline import MajorityClassBaseline, PersistenceBaseline, LogisticRegressionBaseline

logger = logging.getLogger("emberrisk.ml.train")


def fit_baselines(train_df, target_column=PRIMARY_TARGET_COLUMN):
    """Fits all three baseline models on TRAIN ONLY. Returns a dict of
    fitted models. Callers must not pass anything but the train split here
    -- this function has no way to enforce that itself, so ml/train.py's
    __main__ block is the one place that decides which split gets used."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[target_column]

    models = {
        "majority_class": MajorityClassBaseline(),
        "persistence": PersistenceBaseline(),
        "logistic_regression": LogisticRegressionBaseline(),
    }
    for name, model in models.items():
        logger.info("Fitting %s on %d TRAIN rows...", name, len(X_train))
        model.fit(X_train, y_train)

    return models


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    table, report = assemble_feature_label_table()
    logger.info("Assembled table: %s", report)

    train_df, val_df, test_df = chronological_split(table)
    logger.info(
        "Split sizes -- train: %d, validation: %d, test: %d (validation/test NOT used for fitting)",
        len(train_df), len(val_df), len(test_df),
    )

    models = fit_baselines(train_df)

    logger.info(
        "MajorityClassBaseline: majority_class=%s, positive_rate=%.4f",
        models["majority_class"].majority_class_, models["majority_class"].positive_rate_,
    )
    logger.info(
        "LogisticRegressionBaseline: fitted with %d features, classes_=%s",
        len(FEATURE_COLUMNS), models["logistic_regression"].model.classes_,
    )

    print(
        "Baseline models fitted successfully on the TRAIN split only. "
        "No evaluation performed here -- that's Phase 4 Step 5."
    )
