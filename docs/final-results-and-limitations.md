# Final Results and Limitations

**Project:** EmberRisk  
**Date:** August 2026  
**Status:** Phase 7 Complete – TEST Evaluation Frozen

---

## 1. Locked Configuration

The final operating point was selected exclusively on the VALIDATION set during Phase 6 and then frozen:

| Item                    | Value                          |
|-------------------------|--------------------------------|
| Model                   | HistGradientBoostingClassifier |
| Threshold               | 0.70                           |
| Selection basis         | VALIDATION only                |
| VALIDATION status       | Frozen after model/threshold lock |
| TEST status             | Evaluated exactly once         |

No model selection, threshold tuning, or calibration decisions were influenced by the TEST set.

---

## 2. VALIDATION vs TEST Comparison

| Metric       | VALIDATION (Phase 6, threshold 0.70) | TEST (Phase 7) | Absolute Difference | Relative Change |
|--------------|--------------------------------------|----------------|---------------------|-----------------|
| F1 Score     | 0.7340                               | 0.7172         | −0.0168             | −2.3%           |
| Precision    | 0.7227                               | 0.7001         | −0.0226             | −3.1%           |
| Recall       | 0.7457                               | 0.7352         | −0.0105             | −1.4%           |
| PR-AUC       | 0.8176                               | 0.7985         | −0.0191             | −2.3%           |
| ROC-AUC      | 0.9417                               | 0.9311         | −0.0106             | −1.1%           |

**Notes:**
- All VALIDATION metrics are taken from `docs/phase6-threshold-calibration.json`, specifically the `gradient_boosting.optimal_threshold_metrics` (threshold = 0.70) and the top-level `roc_auc` field.
- ROC-AUC is threshold-independent; the same VALIDATION value applies regardless of the chosen operating threshold.
- TEST metrics are taken directly from the one-time Phase 7 evaluation (`docs/phase7-test-evaluation.md`).

---

## 3. Interpretation

The observed performance drop from VALIDATION to TEST is small and consistent with a normal generalization gap:

- F1 dropped from 0.7340 → 0.7172 (−2.3% relative)
- PR-AUC dropped from 0.8176 → 0.7985 (−2.3% relative)
- Precision dropped more (−3.1%) than Recall (−1.4%)

The larger relative drop in precision (compared to recall) indicates that the model became slightly more aggressive in predicting the positive class on the TEST period. This remains a modest change and is still consistent with normal temporal generalization rather than overfitting or data leakage.

### Confusion Matrix (TEST)

|                        | Predicted Negative | Predicted Positive |
|------------------------|--------------------|--------------------|
| **Actual Negative**    | 175,445 (TN)       | 13,992 (FP)        |
| **Actual Positive**    | 11,759 (FN)        | 32,656 (TP)        |

Derived metrics from the confusion matrix (internal consistency check):

- Precision = TP / (TP + FP) = 32656 / (32656 + 13992) = **0.7001**
- Recall    = TP / (TP + FN) = 32656 / (32656 + 11759) = **0.7352**
- F1        = 2 × (Precision × Recall) / (Precision + Recall) = **0.7172**

All reported numbers are internally consistent.

---

## 4. Limitations (Honest Assessment)

The following limitations are deliberate scope decisions or inherent to the current experimental design. They are stated here so that downstream documentation (README, portfolio materials, etc.) remains accurate.

1. **Calibration is imperfect**  
   Phase 6 reported Brier score ≈ 0.0955 and ECE ≈ 0.1237 for the Gradient Boosting model. The project deliberately treats the model as a **binary classifier operating at threshold 0.70**, not as a source of well-calibrated probabilities. Adding calibration was considered and rejected because the product objective is binary fire-risk classification.

2. **PR-AUC is the more informative ranking metric**  
   The TEST set has a positive rate of approximately 19%. Under this level of class imbalance, PR-AUC (0.7985) is a more reliable indicator of ranking quality than ROC-AUC (0.9311). ROC-AUC alone would present an overly optimistic view of performance.

3. **Single temporal split – no cross-validation**  
   The pipeline uses one fixed TRAIN / VALIDATION / TEST temporal split. Consequently there are no confidence intervals or standard errors on the reported metrics. The numbers reflect performance on this particular held-out period only.

4. **Predictive, not causal**  
   EmberRisk is a correlational / predictive model trained on historical fire, weather, and related features. It does not support causal claims about fire behavior or the effect of interventions.

5. **TEST set is closed**  
   Per project rules established in Phase 6 and enforced in Phase 7:
   - No further threshold tuning on TEST
   - No model swapping based on TEST results
   - No addition of calibration using TEST
   - No repeated evaluation of TEST

   Any future experimentation would require a new experimental protocol and a fresh data split.

---

## 5. What These Results Do Not License

These TEST numbers must **not** be used to:

- Re-open model selection
- Adjust the decision threshold
- Add probability calibration
- Claim the model is production-ready without further validation on new time periods
- Overstate certainty (no cross-validated confidence intervals exist)

The evaluation protocol has been followed correctly. The project now moves from the experimentation phase into documentation, presentation, and potential future extension under a new protocol if required.

---

**Document status:** Final for Phase 7.  
**Next recommended artifact:** Portfolio-oriented README that summarizes (but does not re-analyze) these results.
