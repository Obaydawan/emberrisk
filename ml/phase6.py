"""
ml/phase6.py -- Phase 6: threshold & calibration analysis, VALIDATION only.

assemble -> chronological split -> fit RF + HistGradientBoosting on TRAIN ->
predict probabilities on VALIDATION -> threshold sweep + F1-optimal
threshold -> calibration analysis (Brier / 10-bin / ECE) -> report.

TEST is never referenced anywhere in this file. No hyperparameter tuning,
no calibrated model is created -- analysis only, per Phase 6 scope. Reuses
ml/dataset.py, ml/split.py, ml/models.py, ml/features.py unmodified.
"""
import json
import logging
from pathlib import Path

from sklearn.metrics import roc_auc_score, average_precision_score

from ml.dataset import assemble_feature_label_table, FEATURE_COLUMNS, PRIMARY_TARGET_COLUMN
from ml.split import chronological_split
from ml.models import RandomForestModel, GradientBoostingModel
from ml.threshold import evaluate_threshold, find_f1_optimal_threshold
from ml.calibration import compute_brier_score, compute_calibration_curve, compute_ece

logger = logging.getLogger("emberrisk.ml.phase6")

MODELS_TO_ANALYZE = {
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
}

DEFAULT_THRESHOLD = 0.5


def fit_models_for_phase6(train_df, target_column=PRIMARY_TARGET_COLUMN):
    """Fits RF + HistGradientBoosting on TRAIN ONLY -- same convention as
    ml.train.fit_baselines / ml.train_models.fit_tree_models. Callers must
    not pass anything but the train split here."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[target_column]

    models = {}
    for name, cls in MODELS_TO_ANALYZE.items():
        logger.info("Fitting %s on %d TRAIN rows...", name, len(X_train))
        models[name] = cls().fit(X_train, y_train)
    return models


def analyze_model_on_validation(model, val_df, target_column=PRIMARY_TARGET_COLUMN):
    """Threshold sweep + F1-optimal threshold + calibration analysis for
    ONE already-fitted model, on VALIDATION ONLY. This function's signature
    has no test_df parameter -- calling it on TEST isn't expressible
    through this API, same pattern as ml.evaluate.evaluate_all_baselines."""
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[target_column].to_numpy()
    y_prob = model.predict_proba(X_val)[:, 1]

    default_metrics = evaluate_threshold(y_val, y_prob, DEFAULT_THRESHOLD)
    optimal_metrics, sweep_results = find_f1_optimal_threshold(y_val, y_prob)

    both_classes = len(set(y_val)) > 1
    roc_auc = round(float(roc_auc_score(y_val, y_prob)), 4) if both_classes else None
    pr_auc = round(float(average_precision_score(y_val, y_prob)), 4) if both_classes else None

    brier = compute_brier_score(y_val, y_prob)
    calibration_bins = compute_calibration_curve(y_val, y_prob)
    ece = compute_ece(y_val, y_prob)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "default_threshold_metrics": default_metrics,
        "optimal_threshold_metrics": optimal_metrics,
        "threshold_sweep": sweep_results,
        "brier_score": brier,
        "ece": ece,
        "calibration_bins": calibration_bins,
    }


def format_report_markdown(analyses, run_metadata):
    lines = ["# EmberRisk Phase 6 -- Threshold & Calibration Analysis (VALIDATION only)", ""]
    lines.append("**TEST was NOT evaluated, scored, or used in any calculation in this phase.**")
    lines.append("")

    lines.append("## Run metadata")
    lines.append("")
    for k, v in run_metadata.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    lines.append("## Default threshold (0.5) vs F1-optimal threshold")
    lines.append("")
    lines.append("| Model | Default F1 | Optimal Threshold | Optimal F1 | ROC-AUC | PR-AUC | Brier | ECE |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, a in analyses.items():
        d = a["default_threshold_metrics"]
        o = a["optimal_threshold_metrics"]
        lines.append(
            f"| {name} | {d['f1']} | {o['threshold']} | {o['f1']} | "
            f"{a['roc_auc']} | {a['pr_auc']} | {a['brier_score']} | {a['ece']} |"
        )
    lines.append("")

    for name, a in analyses.items():
        lines.append(f"### {name} -- detail")
        lines.append("")
        lines.append(f"**Default (0.5) threshold:** `{a['default_threshold_metrics']}`")
        lines.append("")
        lines.append(f"**F1-optimal threshold:** `{a['optimal_threshold_metrics']}`")
        lines.append("")
        lines.append("**Calibration bins (10-bin, fixed-width):**")
        lines.append("")
        lines.append("| Bin | Range | N | Mean Predicted | Observed Rate |")
        lines.append("|---|---|---|---|---|")
        for b in a["calibration_bins"]:
            lines.append(
                f"| {b['bin_index']} | {b['bin_range']} | {b['n_samples']} | "
                f"{b['mean_predicted_prob']} | {b['observed_positive_rate']} |"
            )
        lines.append("")

    lines.append("## Current preferred candidate")
    lines.append("")
    lines.append(
        "NOT auto-determined here -- this section is filled in after "
        "reviewing the real numbers above, per Phase 6's scope (analysis "
        "only, no automatic final-model declaration)."
    )
    lines.append("")

    return "\n".join(lines)


def run(out_dir="docs"):
    """Full Phase 6 run. Returns (analyses, run_metadata); also writes
    docs/phase6-threshold-calibration.{md,json}."""
    table, assemble_report = assemble_feature_label_table()
    train_df, val_df, test_df = chronological_split(table)
    logger.info(
        "Split sizes -- train: %d, validation: %d, test: %d "
        "(test NOT used anywhere in Phase 6)",
        len(train_df), len(val_df), len(test_df),
    )

    models = fit_models_for_phase6(train_df)

    analyses = {}
    for name, model in models.items():
        logger.info("Analyzing %s on VALIDATION (%d rows)...", name, len(val_df))
        analyses[name] = analyze_model_on_validation(model, val_df)  # val_df ONLY, never test_df

    run_metadata = {
        "target_column": PRIMARY_TARGET_COLUMN,
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test_not_evaluated": len(test_df),
        "validation_positive_rate": round(float(val_df[PRIMARY_TARGET_COLUMN].mean()), 4),
        "threshold_sweep_range": "0.05-0.95, step 0.05",
        "calibration_bins": 10,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = format_report_markdown(analyses, run_metadata)
    with open(out_dir / "phase6-threshold-calibration.md", "w") as f:
        f.write(md)
    with open(out_dir / "phase6-threshold-calibration.json", "w") as f:
        json.dump({"run_metadata": run_metadata, "analyses": analyses}, f, indent=2, default=str)

    return analyses, run_metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    analyses, run_metadata = run()
    print(json.dumps({"run_metadata": run_metadata}, indent=2, default=str))
    for name, a in analyses.items():
        print(f"\n[{name}] default F1={a['default_threshold_metrics']['f1']}  "
              f"optimal threshold={a['optimal_threshold_metrics']['threshold']}  "
              f"optimal F1={a['optimal_threshold_metrics']['f1']}  "
              f"ROC-AUC={a['roc_auc']}  PR-AUC={a['pr_auc']}  "
              f"Brier={a['brier_score']}  ECE={a['ece']}")
    print("\nFull analysis written to docs/phase6-threshold-calibration.md and .json")
    print("TEST was not evaluated.")
