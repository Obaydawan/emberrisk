"""
ml/compare.py -- Phase 5: fair comparison of baselines vs. tree-based models
on VALIDATION only.

Reuses ml.train.fit_baselines(), ml.train_models.fit_tree_models(), and
ml.evaluate.evaluate_all_baselines() / format_results_markdown() UNMODIFIED.
The fairness guarantee of this whole phase rests on all 5 models (3 Phase 4
baselines + 2 Phase 5 tree models) being evaluated through the exact same
evaluate_all_baselines() code path -- this file does not compute any metric
itself, it only builds a combined models dict and hands it to that
unmodified function.

TEST is never referenced anywhere in this file, same discipline as
ml/evaluate.py.
"""
import json
import logging
from pathlib import Path

from ml.dataset import assemble_feature_label_table, PRIMARY_TARGET_COLUMN
from ml.split import chronological_split
from ml.train import fit_baselines
from ml.train_models import fit_tree_models
from ml.evaluate import evaluate_all_baselines, format_results_markdown

logger = logging.getLogger("emberrisk.ml.compare")


def fit_all_models(train_df):
    """Combines Phase 4 baselines and Phase 5 tree models into ONE dict, so
    evaluate_all_baselines() (unmodified) evaluates all 5 through the
    identical path. No new fitting logic here -- just composition of the
    two existing fit_* functions."""
    models = {}
    models.update(fit_baselines(train_df))
    models.update(fit_tree_models(train_df))
    return models


def run(out_dir="docs"):
    """Full Phase 5 run: assemble -> split -> fit ALL models on TRAIN ->
    evaluate ALL models on VALIDATION ONLY -> write
    docs/phase5-model-comparison.{md,json}. TEST is not evaluated."""
    table, assemble_report = assemble_feature_label_table()
    train_df, val_df, test_df = chronological_split(table)
    logger.info(
        "Split sizes -- train: %d, validation: %d, test: %d (test NOT evaluated in Phase 5)",
        len(train_df), len(val_df), len(test_df),
    )

    models = fit_all_models(train_df)
    logger.info("Fitted %d models: %s", len(models), list(models.keys()))

    results = evaluate_all_baselines(models, val_df)  # unmodified Phase 4 function

    for name, m in results.items():
        logger.info("[%s] %s", name, m)

    run_metadata = {
        "target_column": PRIMARY_TARGET_COLUMN,
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test_not_evaluated": len(test_df),
        "validation_positive_rate": round(float(val_df[PRIMARY_TARGET_COLUMN].mean()), 4),
        "models_compared": list(models.keys()),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = format_results_markdown(results, run_metadata)
    md = md.replace(
        "# EmberRisk Phase 4 -- Baseline Results (VALIDATION split only)",
        "# EmberRisk Phase 5 -- Model Comparison (VALIDATION split only)",
    )
    md = md.replace(
        "TEST was not evaluated and no model selection has occurred -- these are VALIDATION-only results, per Phase 4 scope.",
        "TEST was not evaluated. These are VALIDATION-only results; model selection is deferred until after Phase 5 review.",
    )

    with open(out_dir / "phase5-model-comparison.md", "w") as f:
        f.write(md)

    with open(out_dir / "phase5-model-comparison.json", "w") as f:
        json.dump({"run_metadata": run_metadata, "results": results}, f, indent=2, default=str)

    return results, run_metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results, run_metadata = run()
    print(json.dumps({"run_metadata": run_metadata, "results": results}, indent=2, default=str))
