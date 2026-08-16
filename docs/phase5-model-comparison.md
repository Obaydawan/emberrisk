# EmberRisk Phase 5 -- Model Comparison (VALIDATION split only)

TEST was not evaluated. These are VALIDATION-only results; model selection is deferred until after Phase 5 review.

## Run metadata

- **target_column**: future_fire_7d
- **n_train**: 589798
- **n_validation**: 117895
- **n_test_not_evaluated**: 233852
- **validation_positive_rate**: 0.1811
- **models_compared**: ['majority_class', 'persistence', 'logistic_regression', 'random_forest', 'gradient_boosting']

## Metrics (VALIDATION)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| majority_class | 0.0 | 0.0 | 0.0 | 0.5 | 0.1811 |
| persistence | 0.8543 | 0.3143 | 0.4595 | 0.6512 | 0.3926 |
| logistic_regression | 0.521 | 0.8997 | 0.6599 | 0.9279 | 0.7414 |
| random_forest | 0.6156 | 0.8337 | 0.7082 | 0.9406 | 0.8156 |
| gradient_boosting | 0.5776 | 0.8667 | 0.6932 | 0.9417 | 0.8176 |

### majority_class confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 96549 | 0 |
| **Actual 1** | 21346 | 0 |

### persistence confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 95405 | 1144 |
| **Actual 1** | 14638 | 6708 |

### logistic_regression confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 78894 | 17655 |
| **Actual 1** | 2140 | 19206 |

### random_forest confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 85437 | 11112 |
| **Actual 1** | 3550 | 17796 |

### gradient_boosting confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 83020 | 13529 |
| **Actual 1** | 2846 | 18500 |
