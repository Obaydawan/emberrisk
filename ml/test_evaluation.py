"""
ml/test_evaluation.py -- Phase 7: the SINGLE, FINAL evaluation of the
locked model on the previously untouched TEST split.

Locked operating point (from Phase 6, NOT re-derived or re-selected here):
  Model:     GradientBoostingModel (HistGradientBoostingClassifier)
  Threshold: 0.70
  (VALIDATION F1 0.7340, PR-AUC 0.8176 -- Phase 6 reference numbers only,
  NOT used in any computation in this file.)

This module does NOT:
  - tune hyperparameters
  - search thresholds
  - compare models
  - modify features
  - use TEST for anything except this one scoring pass
  - allow going back to VALIDATION after TEST has been scored

Safety design, not just convention:
  - evaluate_on_test(model, test_df, threshold) has NO val_df parameter at
    all -- scoring VALIDATION isn't expressible through this function's
    signature, same pattern as ml.evaluate.evaluate_all_baselines and
    ml.phase6.analyze_model_on_validation.
  - run() writes a lock file (docs/.phase7_test_evaluation.lock) after a
    successful evaluation. Re-running run() without force=True raises
    RuntimeError -- a real, enforced guarantee that TEST is scored exactly
    once, not just a comment saying it should be.
  - LOCKED_MODEL_CLASS / LOCKED_THRESHOLD are module-level constants read
    by run(); there is no code path in this file that searches, sweeps, or
    otherwise varies either of them.

Reuses ml.dataset.assemble_feature_label_table, ml.split.chronological_split
+ validate_split_partition, ml.models.GradientBoostingModel, and
ml.threshold.evaluate_threshold (the SAME single-threshold scoring function
already used and tested in Phase 6) rather than reimplementing metric
computation for TEST.
"""
import json
import logging
from pathlib import Path

from sklearn.metrics import roc_auc_score, average_precision_score

from ml.dataset import assemble_feature_label_table, FEATURE_COLUMNS, PRIMARY_TARGET_COLUMN
from ml.split import chronological_split, validate_split_partition
from ml.models import GradientBoostingModel
from ml.threshold import evaluate_threshold

logger = logging.getLogger("emberrisk.ml.test_evaluation")

# LOCKED per the Phase 6 decision. Do not change without explicitly
# reopening threshold/model selection on VALIDATION in a new phase -- this
# file evaluates a decision already made, it does not make one.
LOCKED_MODEL_CLASS = GradientBoostingModel
LOCKED_MODEL_NAME = "gradient_boosting"
LOCKED_THRESHOLD = 0.70

# Phase 6 VALIDATION reference numbers, for the report only -- never read
# or used in any computation in this file.
VALIDATION_F1_REFERENCE = 0.7340
VALIDATION_PR_AUC_REFERENCE = 0.8176

LOCK_FILE_PATH = Path("docs/.phase7_test_evaluation.lock")


def fit_locked_model(train_df, target_column=PRIMARY_TARGET_COLUMN):
    """Fits ONLY the locked model on TRAIN ONLY. Deliberately narrower than
    ml.train_models.fit_tree_models -- Phase 7 is not a comparison, there
    is exactly one model to fit."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[target_column]
    logger.info("Fitting locked model (%s) on %d TRAIN rows...", LOCKED_MODEL_NAME, len(X_train))
    return LOCKED_MODEL_CLASS().fit(X_train, y_train)


def evaluate_on_test(model, test_df, threshold=LOCKED_THRESHOLD, target_column=PRIMARY_TARGET_COLUMN):
    """The one-time TEST scoring pass. Signature has NO val_df parameter --
    scoring VALIDATION isn't expressible through this function at all.
    Reuses ml.threshold.evaluate_threshold for precision/recall/F1/
    confusion-matrix at the locked threshold -- the identical computation
    already used and tested in Phase 6, not a parallel reimplementation
    that could subtly disagree."""
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[target_column].to_numpy()
    y_prob = model.predict_proba(X_test)[:, 1]

    threshold_metrics = evaluate_threshold(y_test, y_prob, threshold)

    both_classes = len(set(y_test)) > 1
    roc_auc = round(float(roc_auc_score(y_test, y_prob)), 4) if both_classes else None
    pr_auc = round(float(average_precision_score(y_test, y_prob)), 4) if both_classes else None

    return {
        "model_name": LOCKED_MODEL_NAME,
        "locked_threshold": threshold,
        "n_test_rows": int(len(y_test)),
        "test_positive_rate": round(float(y_test.mean()), 4),
        "precision": threshold_metrics["precision"],
        "recall": threshold_metrics["recall"],
        "f1": threshold_metrics["f1"],
        "predicted_positive_rate": threshold_metrics["positive_rate"],
        "true_positive": threshold_metrics["true_positive"],
        "true_negative": threshold_metrics["true_negative"],
        "false_positive": threshold_metrics["false_positive"],
        "false_negative": threshold_metrics["false_negative"],
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def format_report_markdown(results, run_metadata):
    lines = ["# EmberRisk Phase 7 -- Final TEST Evaluation", ""]
    lines.append(
        "**This is the single, final evaluation of the Phase 6 locked "
        "model on the previously untouched TEST split.**"
    )
    lines.append("")
    lines.append("Confirmations:")
    lines.append(
        "- TEST was evaluated exactly ONCE (enforced by the lock file "
        "mechanism in `ml/test_evaluation.py` -- a second run without "
        "`force=True` raises `RuntimeError`)."
    )
    lines.append("- VALIDATION was NOT used anywhere in this evaluation.")
    lines.append("- No tuning, threshold search, or model comparison occurred in this phase.")
    lines.append("")

    lines.append("## Locked operating point (from Phase 6, not re-derived here)")
    lines.append("")
    lines.append(f"- **Model**: {results['model_name']} (HistGradientBoostingClassifier)")
    lines.append(f"- **Threshold**: {results['locked_threshold']}")
    lines.append(f"- **VALIDATION F1 at this operating point (Phase 6, reference only)**: {VALIDATION_F1_REFERENCE}")
    lines.append(f"- **VALIDATION PR-AUC at this operating point (Phase 6, reference only)**: {VALIDATION_PR_AUC_REFERENCE}")
    lines.append("")

    lines.append("## Run metadata")
    lines.append("")
    for k, v in run_metadata.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## TEST results")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for key in [
        "n_test_rows", "test_positive_rate", "precision", "recall", "f1",
        "predicted_positive_rate", "roc_auc", "pr_auc",
    ]:
        lines.append(f"| {key} | {results[key]} |")
    lines.append("")

    lines.append("### Confusion matrix")
    lines.append("")
    lines.append("| | Predicted 0 | Predicted 1 |")
    lines.append("|---|---|---|")
    lines.append(f"| **Actual 0** | {results['true_negative']} | {results['false_positive']} |")
    lines.append(f"| **Actual 1** | {results['false_negative']} | {results['true_positive']} |")
    lines.append("")

    return "\n".join(lines)


def run(out_dir="docs", force=False):
    """Full Phase 7 run: assemble -> split -> fit locked model on TRAIN ->
    evaluate ONCE on TEST -> write docs/phase7-test-evaluation.{md,json}.

    Refuses to run a second time (raises RuntimeError) unless force=True is
    passed explicitly -- a real safeguard against accidentally re-scoring
    TEST, not just a comment. VALIDATION is split out and its size is
    logged for transparency, but it is never read again after the split."""
    if LOCK_FILE_PATH.exists() and not force:
        raise RuntimeError(
            f"TEST has already been evaluated once (lock file found at "
            f"{LOCK_FILE_PATH}). Re-running would mean scoring TEST more "
            f"than once, which Phase 7 explicitly forbids. Pass force=True "
            f"only if you are certain you understand the implications."
        )

    table, assemble_report = assemble_feature_label_table()
    train_df, val_df, test_df = chronological_split(table)
    passed, split_detail = validate_split_partition(table, train_df, val_df, test_df)
    if not passed:
        raise RuntimeError(f"Split validation failed, refusing to proceed: {split_detail}")

    logger.info(
        "Split sizes -- train: %d, validation: %d (NOT used below), test: %d",
        len(train_df), len(val_df), len(test_df),
    )

    model = fit_locked_model(train_df)
    results = evaluate_on_test(model, test_df)  # test_df ONLY -- val_df never read again

    run_metadata = {
        "n_train": len(train_df),
        "n_validation_not_used_for_scoring": len(val_df),
        "target_column": PRIMARY_TARGET_COLUMN,
        "split_validated": passed,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = format_report_markdown(results, run_metadata)
    with open(out_dir / "phase7-test-evaluation.md", "w") as f:
        f.write(md)
    with open(out_dir / "phase7-test-evaluation.json", "w") as f:
        json.dump({"run_metadata": run_metadata, "results": results}, f, indent=2, default=str)

    LOCK_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE_PATH, "w") as f:
        json.dump({
            "evaluated": True,
            "model_name": results["model_name"],
            "locked_threshold": results["locked_threshold"],
        }, f, indent=2)

    return results, run_metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results, run_metadata = run()
    print(json.dumps({"run_metadata": run_metadata, "results": results}, indent=2, default=str))
    print("\nTEST evaluated exactly once. VALIDATION was not used for TEST scoring.")
