"""
Unit tests for ml/threshold.py (Phase 6).
"""
import numpy as np
import pytest

from ml.threshold import evaluate_threshold, sweep_thresholds, find_f1_optimal_threshold


# ---------------------------------------------------------------------------
# evaluate_threshold -- hand-computed correctness
# ---------------------------------------------------------------------------

def test_evaluate_threshold_hand_computed():
    # y_true:  [1, 1, 0, 0, 1]
    # y_prob:  [0.9, 0.4, 0.1, 0.6, 0.8]
    # threshold=0.5 -> preds = [1, 0, 0, 1, 1]
    # TP=2 (idx0,4), FN=1 (idx1), TN=1 (idx2), FP=1 (idx3)
    y_true = [1, 1, 0, 0, 1]
    y_prob = [0.9, 0.4, 0.1, 0.6, 0.8]

    result = evaluate_threshold(y_true, y_prob, 0.5)

    assert result["threshold"] == 0.5
    assert result["true_positive"] == 2
    assert result["false_negative"] == 1
    assert result["true_negative"] == 1
    assert result["false_positive"] == 1
    assert result["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["n_samples"] == 5


def test_evaluate_threshold_boundary_is_inclusive():
    # threshold applied as y_prob >= threshold -- a value exactly at the
    # threshold must count as positive, not negative
    y_true = [1]
    y_prob = [0.5]
    result = evaluate_threshold(y_true, y_prob, 0.5)
    assert result["true_positive"] == 1


def test_evaluate_threshold_extreme_thresholds():
    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.1, 0.8, 0.2]

    all_negative = evaluate_threshold(y_true, y_prob, 1.0)  # nothing >= 1.0... except exact 1.0
    assert all_negative["positive_rate"] == 0.0

    all_positive = evaluate_threshold(y_true, y_prob, 0.0)  # everything >= 0.0
    assert all_positive["positive_rate"] == 1.0
    assert all_positive["recall"] == 1.0


def test_evaluate_threshold_zero_division_handled():
    y_true = [0, 0, 0, 0]
    y_prob = [0.1, 0.2, 0.3, 0.4]
    result = evaluate_threshold(y_true, y_prob, 0.5)
    assert result["precision"] == 0.0  # no positives predicted, zero_division=0
    assert result["recall"] == 0.0


# ---------------------------------------------------------------------------
# sweep_thresholds
# ---------------------------------------------------------------------------

def test_sweep_thresholds_covers_expected_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_prob = rng.random(100)

    results = sweep_thresholds(y_true, y_prob)
    thresholds = [r["threshold"] for r in results]

    assert len(results) == 19  # 0.05 to 0.95 inclusive, step 0.05
    assert thresholds[0] == 0.05
    assert thresholds[-1] == 0.95
    # strictly ascending, no duplicates, no floating point drift
    assert thresholds == sorted(set(thresholds))


def test_sweep_thresholds_is_deterministic():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=50)
    y_prob = rng.random(50)

    results_a = sweep_thresholds(y_true, y_prob)
    results_b = sweep_thresholds(y_true, y_prob)
    assert results_a == results_b


def test_sweep_thresholds_higher_threshold_never_increases_positive_rate():
    """Monotonicity sanity check: raising the threshold can only keep the
    same or fewer predictions positive."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.random(200)

    results = sweep_thresholds(y_true, y_prob)
    positive_rates = [r["positive_rate"] for r in results]
    assert all(positive_rates[i] >= positive_rates[i + 1] for i in range(len(positive_rates) - 1))


# ---------------------------------------------------------------------------
# find_f1_optimal_threshold
# ---------------------------------------------------------------------------

def test_find_f1_optimal_threshold_picks_the_actual_max():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 2, size=300)
    y_prob = rng.random(300)

    best, all_results = find_f1_optimal_threshold(y_true, y_prob)
    max_f1_in_sweep = max(r["f1"] for r in all_results)

    assert best["f1"] == max_f1_in_sweep


def test_find_f1_optimal_threshold_tie_break_prefers_lower_threshold():
    """Construct probabilities where multiple thresholds yield identical F1
    (e.g. a probability gap wide enough that several cutoffs give the same
    predictions) -- the LOWER threshold must be selected, deterministically."""
    # All probabilities cluster at exactly 0.3 or 0.7 -- any threshold in
    # (0.3, 0.7] gives identical predictions/F1, so thresholds 0.35..0.70
    # in the sweep should all tie, and 0.35 (or whichever sweep value is
    # smallest and still > 0.3) must be chosen.
    y_true = [1, 1, 0, 0]
    y_prob = [0.7, 0.7, 0.3, 0.3]

    best, all_results = find_f1_optimal_threshold(y_true, y_prob, start=0.1, stop=0.9, step=0.1)
    tied_thresholds = [r["threshold"] for r in all_results if r["f1"] == best["f1"]]

    assert len(tied_thresholds) > 1  # confirms a real tie occurred
    assert best["threshold"] == min(tied_thresholds)


def test_find_f1_optimal_threshold_deterministic_across_calls():
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 2, size=150)
    y_prob = rng.random(150)

    best_a, _ = find_f1_optimal_threshold(y_true, y_prob)
    best_b, _ = find_f1_optimal_threshold(y_true, y_prob)
    assert best_a == best_b
