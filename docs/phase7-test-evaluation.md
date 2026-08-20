# EmberRisk Phase 7 -- Final TEST Evaluation

**This is the single, final evaluation of the Phase 6 locked model on the previously untouched TEST split.**

Confirmations:
- TEST was evaluated exactly ONCE (enforced by the lock file mechanism in `ml/test_evaluation.py` -- a second run without `force=True` raises `RuntimeError`).
- VALIDATION was NOT used anywhere in this evaluation.
- No tuning, threshold search, or model comparison occurred in this phase.

## Locked operating point (from Phase 6, not re-derived here)

- **Model**: gradient_boosting (HistGradientBoostingClassifier)
- **Threshold**: 0.7
- **VALIDATION F1 at this operating point (Phase 6, reference only)**: 0.734
- **VALIDATION PR-AUC at this operating point (Phase 6, reference only)**: 0.8176

## Run metadata

- **n_train**: 589798
- **n_validation_not_used_for_scoring**: 117895
- **target_column**: future_fire_7d
- **split_validated**: True

## TEST results

| Metric | Value |
|---|---|
| n_test_rows | 233852 |
| test_positive_rate | 0.1899 |
| precision | 0.7001 |
| recall | 0.7352 |
| f1 | 0.7172 |
| predicted_positive_rate | 0.1995 |
| roc_auc | 0.9311 |
| pr_auc | 0.7985 |

### Confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 175445 | 13992 |
| **Actual 1** | 11759 | 32656 |
