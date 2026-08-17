"""
ml/calibration.py -- Phase 6: probability calibration analysis, VALIDATION
only.

Implements Brier score, a 10-bin fixed-width reliability/calibration curve,
and Expected Calibration Error (ECE). Pure functions of (y_true, y_prob) --
no model fitting, no threshold decisions, and this module does NOT create a
calibrated model (e.g. via CalibratedClassifierCV). Phase 6 is analysis
only; calibrating a model for actual use is a separate, later decision.
"""
import numpy as np
from sklearn.metrics import brier_score_loss

DEFAULT_N_BINS = 10


def compute_brier_score(y_true, y_prob):
    """Mean squared error between predicted probability and the true
    binary outcome -- lower is better, 0 is perfect."""
    return round(float(brier_score_loss(y_true, y_prob)), 6)


def compute_calibration_curve(y_true, y_prob, n_bins=DEFAULT_N_BINS):
    """Deterministic n_bins reliability data: for each FIXED-WIDTH bin over
    [0, 1] (not quantile bins, so bin boundaries never depend on the data
    distribution -- fully reproducible across models/runs), the number of
    samples, mean predicted probability, and observed positive rate."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right=True with interior edges: values exactly on an edge fall into
    # the lower bin, except 0.0 which correctly falls into bin 0.
    bin_indices = np.digitize(y_prob, bin_edges[1:-1], right=True)

    bins = []
    for i in range(n_bins):
        mask = bin_indices == i
        n_in_bin = int(mask.sum())
        bin_range = [round(float(bin_edges[i]), 2), round(float(bin_edges[i + 1]), 2)]
        if n_in_bin == 0:
            bins.append({
                "bin_index": i, "bin_range": bin_range, "n_samples": 0,
                "mean_predicted_prob": None, "observed_positive_rate": None,
            })
            continue
        bins.append({
            "bin_index": i,
            "bin_range": bin_range,
            "n_samples": n_in_bin,
            "mean_predicted_prob": round(float(y_prob[mask].mean()), 4),
            "observed_positive_rate": round(float(y_true[mask].mean()), 4),
        })
    return bins


def compute_ece(y_true, y_prob, n_bins=DEFAULT_N_BINS):
    """Expected Calibration Error: sample-count-weighted average of
    |mean_predicted_prob - observed_positive_rate| across bins. Reuses
    compute_calibration_curve rather than recomputing bin membership
    separately, so ECE and the reported reliability curve can never
    disagree with each other. Returns None only if y_true/y_prob are empty
    (nothing to compute)."""
    bins = compute_calibration_curve(y_true, y_prob, n_bins)
    n_total = sum(b["n_samples"] for b in bins)
    if n_total == 0:
        return None

    ece = 0.0
    for b in bins:
        if b["n_samples"] == 0:
            continue
        gap = abs(b["mean_predicted_prob"] - b["observed_positive_rate"])
        ece += (b["n_samples"] / n_total) * gap
    return round(float(ece), 6)
