"""
Unit tests for ml/calibration.py (Phase 6).
"""
import numpy as np
import pytest

from ml.calibration import compute_brier_score, compute_calibration_curve, compute_ece


# ---------------------------------------------------------------------------
# compute_brier_score -- hand-computed
# ---------------------------------------------------------------------------

def test_brier_score_perfect_predictions_is_zero():
    y_true = [0, 1, 0, 1]
    y_prob = [0.0, 1.0, 0.0, 1.0]
    assert compute_brier_score(y_true, y_prob) == 0.0


def test_brier_score_hand_computed():
    # (0.5-0)^2 + (0.5-1)^2 = 0.25 + 0.25 -> mean = 0.25
    y_true = [0, 1]
    y_prob = [0.5, 0.5]
    assert compute_brier_score(y_true, y_prob) == pytest.approx(0.25)


def test_brier_score_worst_case():
    # completely wrong confident predictions -> Brier = 1.0
    y_true = [0, 1]
    y_prob = [1.0, 0.0]
    assert compute_brier_score(y_true, y_prob) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_calibration_curve
# ---------------------------------------------------------------------------

def test_calibration_curve_returns_ten_bins_by_default():
    y_true = [0, 1] * 50
    y_prob = list(np.linspace(0.01, 0.99, 100))
    bins = compute_calibration_curve(y_true, y_prob)
    assert len(bins) == 10


def test_calibration_curve_bin_ranges_are_fixed_width():
    y_true = [0] * 10
    y_prob = [0.05] * 10
    bins = compute_calibration_curve(y_true, y_prob)
    expected_ranges = [[round(i / 10, 2), round((i + 1) / 10, 2)] for i in range(10)]
    assert [b["bin_range"] for b in bins] == expected_ranges


def test_calibration_curve_hand_computed_single_bin():
    # All 4 samples land in bin 0 ([0.0, 0.1))
    y_true = [0, 0, 1, 1]
    y_prob = [0.05, 0.05, 0.05, 0.05]
    bins = compute_calibration_curve(y_true, y_prob)

    bin0 = bins[0]
    assert bin0["n_samples"] == 4
    assert bin0["mean_predicted_prob"] == 0.05
    assert bin0["observed_positive_rate"] == 0.5  # 2 of 4 are positive

    # every other bin is empty
    for b in bins[1:]:
        assert b["n_samples"] == 0
        assert b["mean_predicted_prob"] is None
        assert b["observed_positive_rate"] is None


def test_calibration_curve_perfectly_calibrated_case():
    """If predicted probability always equals the true observed rate in
    each bin, mean_predicted_prob and observed_positive_rate should match
    closely for a large enough sample."""
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0, 1, size=5000)
    y_true = (rng.random(5000) < y_prob).astype(int)  # well-calibrated by construction

    bins = compute_calibration_curve(y_true, y_prob)
    for b in bins:
        if b["n_samples"] > 50:  # ignore sparse bins, noisy at small N
            assert abs(b["mean_predicted_prob"] - b["observed_positive_rate"]) < 0.1


def test_calibration_curve_custom_bin_count():
    y_true = [0, 1] * 10
    y_prob = list(np.linspace(0.01, 0.99, 20))
    bins = compute_calibration_curve(y_true, y_prob, n_bins=5)
    assert len(bins) == 5


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------

def test_ece_zero_for_perfect_calibration_large_sample():
    rng = np.random.default_rng(1)
    y_prob = rng.uniform(0, 1, size=20000)
    y_true = (rng.random(20000) < y_prob).astype(int)

    ece = compute_ece(y_true, y_prob)
    assert ece < 0.02  # should be very small (not exactly 0 due to sampling noise)


def test_ece_hand_computed_two_bins():
    # bin 0 ([0.0,0.1)): 2 samples, mean_pred=0.05, observed=0.0  -> gap=0.05
    # bin 9 ([0.9,1.0]): 2 samples, mean_pred=0.95, observed=1.0  -> gap=0.05
    y_true = [0, 0, 1, 1]
    y_prob = [0.05, 0.05, 0.95, 0.95]
    ece = compute_ece(y_true, y_prob)
    # weighted average of two equal-size bins with identical gap = 0.05
    assert ece == pytest.approx(0.05, abs=1e-4)


def test_ece_matches_calibration_curve_bins():
    """ECE must be derivable from the exact bins returned by
    compute_calibration_curve -- recompute it manually from the bins and
    confirm it matches, proving the two functions can't silently disagree."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=500)
    y_prob = rng.random(500)

    bins = compute_calibration_curve(y_true, y_prob)
    n_total = sum(b["n_samples"] for b in bins)
    manual_ece = sum(
        (b["n_samples"] / n_total) * abs(b["mean_predicted_prob"] - b["observed_positive_rate"])
        for b in bins if b["n_samples"] > 0
    )

    ece = compute_ece(y_true, y_prob)
    assert ece == pytest.approx(manual_ece, abs=1e-6)


def test_ece_is_deterministic():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.random(200)
    assert compute_ece(y_true, y_prob) == compute_ece(y_true, y_prob)
