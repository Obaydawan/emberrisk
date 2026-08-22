"""
ml/train_and_save_locked_model.py -- Phase 9: persist the Phase 6/7 locked
model as a reusable artifact.

Phases 5-7 always fit GradientBoostingModel fresh from TRAIN inside the
same run that used it (model comparison, threshold sweep, TEST evaluation).
That was correct for those phases -- each was a one-time analysis pass.

Phase 9 introduces a scheduled, repeatedly-run pipeline. Refitting on all
589,798 TRAIN rows every single DAG run would be slow and wasteful, and
isn't how a production system would work: training and serving should be
separate concerns.

This script does ONLY ONE THING: fits the SAME locked model
(GradientBoostingModel / HistGradientBoostingClassifier), on the SAME
TRAIN split, using the SAME feature set, and saves the fitted object to
disk via joblib. It does not change the model class, hyperparameters, or
training data in any way -- it persists a decision already made in Phase
6/7, exactly like ml/test_evaluation.py does for the TEST report.

This is a manual, explicit step -- it is NOT run automatically by the
Airflow DAG. Re-running this script means re-fitting the locked model
(e.g. because you deliberately refreshed TRAIN data under a new,
documented protocol), not something that should happen silently as a side
effect of a scheduled pipeline run.

Usage:
    PYTHONPATH=. python -m ml.train_and_save_locked_model
"""
import json
import logging
from pathlib import Path

import joblib

from ml.dataset import assemble_feature_label_table, FEATURE_COLUMNS, PRIMARY_TARGET_COLUMN
from ml.split import chronological_split, validate_split_partition
from ml.models import GradientBoostingModel

logger = logging.getLogger("emberrisk.ml.train_and_save_locked_model")

# Must stay identical to ml/test_evaluation.py's locked constants. Not
# imported from there directly to keep test_evaluation.py's TEST-scoring
# path fully self-contained and unmodified by this Phase 9 addition -- but
# any change here must be made in both places together, deliberately.
LOCKED_MODEL_CLASS = GradientBoostingModel
LOCKED_MODEL_NAME = "gradient_boosting"
LOCKED_THRESHOLD = 0.70

DEFAULT_MODEL_DIR = Path("models")
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "gradient_boosting_locked.joblib"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "gradient_boosting_locked.metadata.json"


def train_locked_model():
    """Assembles the feature/label table, splits it, fits the locked model
    on TRAIN only, and validates the split before returning. Does not touch
    VALIDATION or TEST for anything beyond confirming split integrity."""
    table, assemble_report = assemble_feature_label_table()
    train_df, val_df, test_df = chronological_split(table)
    passed, split_detail = validate_split_partition(table, train_df, val_df, test_df)
    if not passed:
        raise RuntimeError(f"Split validation failed, refusing to train: {split_detail}")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[PRIMARY_TARGET_COLUMN]

    logger.info(
        "Fitting locked model (%s) on %d TRAIN rows (%d features)...",
        LOCKED_MODEL_NAME, len(X_train), len(FEATURE_COLUMNS),
    )
    model = LOCKED_MODEL_CLASS().fit(X_train, y_train)

    metadata = {
        "model_name": LOCKED_MODEL_NAME,
        "locked_threshold": LOCKED_THRESHOLD,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": PRIMARY_TARGET_COLUMN,
        "n_train_rows": len(train_df),
        "split_validated": passed,
        "note": (
            "This artifact reproduces the Phase 6/7 locked model exactly "
            "-- same class, same hyperparameters, same TRAIN split. It "
            "does not represent a new model-selection decision."
        ),
    }
    return model, metadata


def save_model(model, metadata, model_path=DEFAULT_MODEL_PATH, metadata_path=DEFAULT_METADATA_PATH):
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Saved locked model to %s", model_path)
    logger.info("Saved model metadata to %s", metadata_path)
    return model_path, metadata_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    model, metadata = train_locked_model()
    model_path, metadata_path = save_model(model, metadata)
    print(json.dumps(metadata, indent=2, default=str))
    print(f"\nLocked model saved to: {model_path}")
