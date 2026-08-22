"""
ml/predict.py -- Phase 9: batch scoring using the persisted locked model.

Loads the model artifact saved by ml/train_and_save_locked_model.py and
scores new rows at the locked threshold (0.70). This module does NOT fit,
tune, or select anything -- it is a pure scoring/inference layer on top of
a decision already made in Phase 6/7 and persisted in Phase 9.

Reuses ml.features.select_features (the same shared feature-selection
guard used by every model in this project) so batch-scored input goes
through the identical column-selection contract as training/evaluation,
rather than a parallel, potentially-drifting implementation.

Usage (as a library):
    from ml.predict import load_locked_model, score_batch
    model, metadata = load_locked_model()
    scored_df = score_batch(model, new_data_df, threshold=metadata["locked_threshold"])

Usage (as a script, scores today's feature table and writes predictions):
    PYTHONPATH=. python -m ml.predict
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from ml.features import select_features
from ml.train_and_save_locked_model import DEFAULT_MODEL_PATH, DEFAULT_METADATA_PATH

logger = logging.getLogger("emberrisk.ml.predict")

DEFAULT_PREDICTIONS_DIR = Path("data/predictions")


def load_locked_model(model_path=DEFAULT_MODEL_PATH, metadata_path=DEFAULT_METADATA_PATH):
    """Loads the persisted locked model and its metadata. Raises a clear
    error (not a generic FileNotFoundError) if the artifact hasn't been
    created yet, since that's a specific, actionable setup step."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Locked model artifact not found at {model_path}. Run "
            f"'PYTHONPATH=. python -m ml.train_and_save_locked_model' once "
            f"to create it before scoring."
        )

    model = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    logger.info(
        "Loaded locked model '%s' (threshold=%s, trained on %d TRAIN rows)",
        metadata["model_name"], metadata["locked_threshold"], metadata["n_train_rows"],
    )
    return model, metadata


def score_batch(model, feature_df, threshold=0.70, id_columns=("cell_id", "date")):
    """Scores feature_df with the given model at the given threshold.
    feature_df may contain extra columns (e.g. cell_id, date) -- they are
    preserved in the output but never passed to the model, since
    select_features() enforces exactly ml.dataset.FEATURE_COLUMNS.

    Returns a DataFrame with the original id_columns (if present) plus
    predicted_probability and predicted_positive (bool at the given
    threshold)."""
    X = select_features(feature_df)
    probabilities = model.predict_proba(X)[:, 1]

    output_columns = {}
    for col in id_columns:
        if col in feature_df.columns:
            output_columns[col] = feature_df[col].values

    output_columns["predicted_probability"] = probabilities
    output_columns["predicted_positive"] = probabilities >= threshold

    result = pd.DataFrame(output_columns)
    logger.info(
        "Scored %d rows at threshold=%.2f: %d predicted positive (%.2f%%)",
        len(result), threshold, int(result["predicted_positive"].sum()),
        100 * result["predicted_positive"].mean() if len(result) else 0.0,
    )
    return result


def write_predictions(scored_df, out_dir=DEFAULT_PREDICTIONS_DIR, run_date=None):
    """Writes scored predictions to a timestamped Parquet file, so each
    DAG run's output is kept separately rather than overwritten -- this
    keeps batch scoring idempotent and auditable (a given run_date's
    predictions can be regenerated and compared, not silently replaced
    with no history)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_date is None:
        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    out_path = out_dir / f"predictions_{run_date}.parquet"
    scored_df.to_parquet(out_path, index=False)
    logger.info("Wrote %d predictions to %s", len(scored_df), out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Score a feature table with the locked EmberRisk model.")
    parser.add_argument(
        "--features-path", default=None,
        help="Path to a Parquet feature table to score. If omitted, this script does nothing "
             "on its own -- it must be called with real feature data, either via this flag or "
             "by importing score_batch() directly, e.g. from an Airflow task.",
    )
    args = parser.parse_args()

    if args.features_path is None:
        raise SystemExit(
            "No --features-path given. This script does not assume a default "
            "feature source -- pass the path to the feature table you want scored."
        )

    model, metadata = load_locked_model()
    feature_df = pd.read_parquet(args.features_path)
    scored_df = score_batch(model, feature_df, threshold=metadata["locked_threshold"])
    out_path = write_predictions(scored_df)
    print(f"Predictions written to: {out_path}")
