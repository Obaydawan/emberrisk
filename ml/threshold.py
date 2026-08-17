"""
ml/threshold.py -- Phase 6: threshold analysis, VALIDATION only.

Pure functions of (y_true, y_prob, threshold) -- no randomness, no model
fitting, no knowledge of splits at all. It is the caller's responsibility
(ml/phase6.py) to only ever pass VALIDATION data here; nothing in this
module can enforce that itself, same convention as ml/evaluate.py.
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

DEFAULT_SWEEP_START = 0.05
DEFAULT_SWEEP_STOP = 0.95
DEFAULT_SWEEP_STEP = 0.05


def evaluate_threshold(y_true, y_prob, threshold):
    """Precision/recall/F1/positive-rate/confusion-matrix counts at a
    single threshold. threshold is applied as y_prob >= threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    preds = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "positive_rate": round(float(preds.mean()), 4) if len(preds) else None,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "n_samples": int(len(y_true)),
    }


def sweep_thresholds(y_true, y_prob, start=DEFAULT_SWEEP_START,
                      stop=DEFAULT_SWEEP_STOP, step=DEFAULT_SWEEP_STEP):
    """Deterministic sweep from start to stop (inclusive), fixed step.
    Thresholds are rounded to 2 decimals to avoid floating-point drift
    (e.g. 0.30000000000000004) so results are exactly reproducible and
    human-readable in the report."""
    thresholds = [round(t, 2) for t in np.arange(start, stop + step / 2, step)]
    return [evaluate_threshold(y_true, y_prob, t) for t in thresholds]


def find_f1_optimal_threshold(y_true, y_prob, start=DEFAULT_SWEEP_START,
                               stop=DEFAULT_SWEEP_STOP, step=DEFAULT_SWEEP_STEP):
    """Returns (best_result, all_sweep_results). Ties are broken by
    preferring the LOWER threshold: thresholds are swept in ascending
    order and Python's max() keeps the first maximal element, so this is a
    deterministic, documented tie-break rather than an arbitrary one."""
    results = sweep_thresholds(y_true, y_prob, start, stop, step)
    best = max(results, key=lambda r: r["f1"])
    return best, results
