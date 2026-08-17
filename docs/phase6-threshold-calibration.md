# EmberRisk Phase 6 -- Threshold & Calibration Analysis (VALIDATION only)

**TEST was NOT evaluated, scored, or used in any calculation in this phase.**

## Run metadata

- **target_column**: future_fire_7d
- **n_train**: 589798
- **n_validation**: 117895
- **n_test_not_evaluated**: 233852
- **validation_positive_rate**: 0.1811
- **threshold_sweep_range**: 0.05-0.95, step 0.05
- **calibration_bins**: 10

## Default threshold (0.5) vs F1-optimal threshold

| Model | Default F1 | Optimal Threshold | Optimal F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|---|---|
| random_forest | 0.7082 | 0.65 | 0.7307 | 0.9406 | 0.8156 | 0.087495 | 0.104211 |
| gradient_boosting | 0.6932 | 0.7 | 0.734 | 0.9417 | 0.8176 | 0.095452 | 0.123672 |

### random_forest -- detail

**Default (0.5) threshold:** `{'threshold': 0.5, 'precision': 0.6156, 'recall': 0.8337, 'f1': 0.7082, 'positive_rate': 0.2452, 'true_positive': 17796, 'true_negative': 85437, 'false_positive': 11112, 'false_negative': 3550, 'n_samples': 117895}`

**F1-optimal threshold:** `{'threshold': 0.65, 'precision': 0.7209, 'recall': 0.7407, 'f1': 0.7307, 'positive_rate': 0.1861, 'true_positive': 15812, 'true_negative': 90426, 'false_positive': 6123, 'false_negative': 5534, 'n_samples': 117895}`

**Calibration bins (10-bin, fixed-width):**

| Bin | Range | N | Mean Predicted | Observed Rate |
|---|---|---|---|---|
| 0 | [0.0, 0.1] | 56320 | 0.0438 | 0.0076 |
| 1 | [0.1, 0.2] | 15728 | 0.134 | 0.0374 |
| 2 | [0.2, 0.3] | 5543 | 0.2499 | 0.1037 |
| 3 | [0.3, 0.4] | 5861 | 0.3507 | 0.1525 |
| 4 | [0.4, 0.5] | 5535 | 0.4487 | 0.1922 |
| 5 | [0.5, 0.6] | 4793 | 0.5488 | 0.266 |
| 6 | [0.6, 0.7] | 4281 | 0.6496 | 0.3644 |
| 7 | [0.7, 0.8] | 4082 | 0.7499 | 0.4936 |
| 8 | [0.8, 0.9] | 4839 | 0.8514 | 0.6466 |
| 9 | [0.9, 1.0] | 10913 | 0.9658 | 0.8996 |

### gradient_boosting -- detail

**Default (0.5) threshold:** `{'threshold': 0.5, 'precision': 0.5776, 'recall': 0.8667, 'f1': 0.6932, 'positive_rate': 0.2717, 'true_positive': 18500, 'true_negative': 83020, 'false_positive': 13529, 'false_negative': 2846, 'n_samples': 117895}`

**F1-optimal threshold:** `{'threshold': 0.7, 'precision': 0.7227, 'recall': 0.7457, 'f1': 0.734, 'positive_rate': 0.1868, 'true_positive': 15917, 'true_negative': 90442, 'false_positive': 6107, 'false_negative': 5429, 'n_samples': 117895}`

**Calibration bins (10-bin, fixed-width):**

| Bin | Range | N | Mean Predicted | Observed Rate |
|---|---|---|---|---|
| 0 | [0.0, 0.1] | 53599 | 0.0448 | 0.0068 |
| 1 | [0.1, 0.2] | 16597 | 0.131 | 0.0274 |
| 2 | [0.2, 0.3] | 4350 | 0.2504 | 0.0867 |
| 3 | [0.3, 0.4] | 5453 | 0.3522 | 0.1265 |
| 4 | [0.4, 0.5] | 5867 | 0.4502 | 0.1633 |
| 5 | [0.5, 0.6] | 5215 | 0.5483 | 0.2209 |
| 6 | [0.6, 0.7] | 4790 | 0.6503 | 0.2987 |
| 7 | [0.7, 0.8] | 4321 | 0.7497 | 0.4393 |
| 8 | [0.8, 0.9] | 5360 | 0.853 | 0.5918 |
| 9 | [0.9, 1.0] | 12343 | 0.9651 | 0.8788 |

## Current preferred candidate

NOT auto-determined here -- this section is filled in after reviewing the real numbers above, per Phase 6's scope (analysis only, no automatic final-model declaration).
