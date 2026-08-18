"""
ml/calibration.py -- Phase 6: probability calibration analysis, VALIDATION
only.

Implements Brier score, a 10-bin fixed-width reliability/calibration curve,
and Expected Calibration Error (ECE). Pure functions of (y_true, y_prob) --
no model fitting, no threshold decisions, and this module does NOT create a
calibrated model (e.g. via CalibratedClassifierCV). Phase 6 is analysis
only; calibrating a model for actual use is a separate, later decision.

Bin convention: conventional fixed-width bins
  [0.0, 0.1), [0.1, 0.2), ..., [0.8, 0.9), [0.9, 1.0]
i.e. every bin is half-open on the right EXCEPT the final bin, which is
closed on both ends so a probability of exactly 1.0 has somewhere to go.
A value exactly on an interior boundary (e.g. 0.1, 0.9) belongs to the
UPPER bin.

Precision policy: per-bin statistics are computed and used internally at
full floating-point precision. ECE is computed from those full-precision
values, not from the rounded values returned by compute_calibration_curve()
-- rounding is applied ONLY at the final reporting step (the values
returned to a caller / written to a report), never to values used in a
downstream calculation.
"""
import numpy as np
from sklearn.metrics import brier_score_loss

DEFAULT_N_BINS = 10


def compute_brier_score(y_true, y_prob):
    """Mean squared error between predicted probability and the true
    binary outcome -- lower is better, 0 is perfect."""
    return round(float(brier_score_loss(y_true, y_prob)), 6)


def compute_calibration_bins_raw(y_true, y_prob, n_bins=DEFAULT_N_BINS):
    """FULL-PRECISION per-bin statistics -- no rounding anywhere in this
    function. This is the single source of truth both
    compute_calibration_curve() (rounds for reporting) and compute_ece()
    (uses full precision directly) build on, so a report-only rounding
    decision can never silently change what ECE actually measures.

    Bin membership uses right=False against the interior edges, which
    gives exactly [0.0,0.1), [0.1,0.2), ..., [0.9,1.0] -- verified
    directly against numpy.digitize's documented semantics, not assumed.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1], right=False)

    bins = []
    for i in range(n_bins):
        mask = bin_indices == i
        n_in_bin = int(mask.sum())
        bin_range = (float(bin_edges[i]), float(bin_edges[i + 1]))
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
            "mean_predicted_prob": float(y_prob[mask].mean()),
            "observed_positive_rate": float(y_true[mask].mean()),
        })
    return bins


def compute_calibration_curve(y_true, y_prob, n_bins=DEFAULT_N_BINS):
    """Reporting-facing reliability table: same bins as
    compute_calibration_bins_raw(), with values rounded ONLY for display
    (mean_predicted_prob / observed_positive_rate to 4 decimals, bin_range
    to 2 decimals). Not used internally by compute_ece() -- that function
    reads the raw, unrounded bins directly."""
    raw_bins = compute_calibration_bins_raw(y_true, y_prob, n_bins)
    rounded = []
    for b in raw_bins:
        rounded.append({
            "bin_index": b["bin_index"],
            "bin_range": [round(b["bin_range"][0], 2), round(b["bin_range"][1], 2)],
            "n_samples": b["n_samples"],
            "mean_predicted_prob": round(b["mean_predicted_prob"], 4) if b["mean_predicted_prob"] is not None else None,
            "observed_positive_rate": round(b["observed_positive_rate"], 4) if b["observed_positive_rate"] is not None else None,
        })
    return rounded


def compute_ece(y_true, y_prob, n_bins=DEFAULT_N_BINS):
    """Expected Calibration Error: sample-count-weighted average of
    |mean_predicted_prob - observed_positive_rate| across bins, computed
    from FULL-PRECISION bin statistics (compute_calibration_bins_raw), NOT
    from the rounded values compute_calibration_curve() returns for
    reporting. Only the final scalar result is rounded, at the very end.
    Returns None only if there are no samples at all."""
    bins = compute_calibration_bins_raw(y_true, y_prob, n_bins)
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
