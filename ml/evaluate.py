"""
ml/evaluate.py -- Phase 4 Step 5: baseline evaluation on VALIDATION only.

Evaluates MajorityClassBaseline, PersistenceBaseline, and
LogisticRegressionBaseline (all fit on TRAIN via ml.train.fit_baselines)
using the VALIDATION split ONLY. TEST is never touched here -- test-set
evaluation and final model selection happen in a later phase, once a model
has actually been chosen, so TEST stays genuinely held out.
"""
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)

from ml.dataset import assemble_feature_label_table, FEATURE_COLUMNS, PRIMARY_TARGET_COLUMN
from ml.split import chronological_split
from ml.train import fit_baselines

logger = logging.getLogger("emberrisk.ml.evaluate")


def evaluate_model(model, X, y, model_name=""):
    """Computes precision/recall/F1/ROC-AUC/PR-AUC/confusion matrix for one
    fitted model against the given X/y. Caller decides which split X/y
    come from -- this function has no concept of splits at all."""
    y = np.asarray(y)
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    both_classes_present = len(set(y)) > 1
    metrics = {
        "model": model_name,
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "positive_rate": round(float(y.mean()), 4) if len(y) else None,
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4) if both_classes_present else None,
        "pr_auc": round(float(average_precision_score(y, proba)), 4) if both_classes_present else None,
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
    }
    return metrics


def evaluate_all_baselines(models, val_df, target_column=PRIMARY_TARGET_COLUMN):
    """models: dict from ml.train.fit_baselines (already fit on TRAIN).
    val_df: the VALIDATION split ONLY. This function accepts no test_df
    parameter by design -- there is no way to call this on TEST through
    this signature."""
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[target_column]

    results = {}
    for name, model in models.items():
        logger.info("Evaluating %s on VALIDATION (%d rows)...", name, len(X_val))
        results[name] = evaluate_model(model, X_val, y_val, model_name=name)
    return results


def format_results_markdown(results, run_metadata=None):
    lines = ["# EmberRisk Phase 4 -- Baseline Results (VALIDATION split only)", ""]
    lines.append(
        "TEST was not evaluated and no model selection has occurred -- "
        "these are VALIDATION-only results, per Phase 4 scope."
    )
    lines.append("")

    if run_metadata:
        lines.append("## Run metadata")
        lines.append("")
        for k, v in run_metadata.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## Metrics (VALIDATION)")
    lines.append("")
    lines.append("| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |")
    lines.append("|---|---|---|---|---|---|")
    for name, m in results.items():
        lines.append(
            f"| {name} | {m['precision']} | {m['recall']} | {m['f1']} | "
            f"{m['roc_auc']} | {m['pr_auc']} |"
        )
    lines.append("")

    for name, m in results.items():
        cm = m["confusion_matrix"]
        lines.append(f"### {name} confusion matrix")
        lines.append("")
        lines.append("| | Predicted 0 | Predicted 1 |")
        lines.append("|---|---|---|")
        lines.append(f"| **Actual 0** | {cm['true_negative']} | {cm['false_positive']} |")
        lines.append(f"| **Actual 1** | {cm['false_negative']} | {cm['true_positive']} |")
        lines.append("")

    return "\n".join(lines)


def run(out_dir="docs"):
    """Full Step 5 run: assemble -> split -> fit on TRAIN -> evaluate on
    VALIDATION ONLY -> write docs/phase4-baseline-results.{md,json}."""
    table, assemble_report = assemble_feature_label_table()
    train_df, val_df, test_df = chronological_split(table)
    logger.info(
        "Split sizes -- train: %d, validation: %d, test: %d (test NOT evaluated in Phase 4)",
        len(train_df), len(val_df), len(test_df),
    )

    models = fit_baselines(train_df)
    results = evaluate_all_baselines(models, val_df)  # val_df ONLY -- never test_df

    for name, m in results.items():
        logger.info("[%s] %s", name, m)

    run_metadata = {
        "target_column": PRIMARY_TARGET_COLUMN,
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test_not_evaluated": len(test_df),
        "validation_positive_rate": round(float(val_df[PRIMARY_TARGET_COLUMN].mean()), 4),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = format_results_markdown(results, run_metadata)
    with open(out_dir / "phase4-baseline-results.md", "w") as f:
        f.write(md)
    with open(out_dir / "phase4-baseline-results.json", "w") as f:
        json.dump({"run_metadata": run_metadata, "results": results}, f, indent=2, default=str)

    return results, run_metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results, run_metadata = run()
    print(json.dumps({"run_metadata": run_metadata, "results": results}, indent=2, default=str))
