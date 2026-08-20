## Phase 7 — Final TEST Evaluation

### Objective
Perform the single, final evaluation of the Phase 6 locked model
(Gradient Boosting, threshold 0.70) on the previously untouched TEST
split. Answers exactly one question: how does the already-decided
EmberRisk model perform on genuinely unseen future data? Not a
comparison, not a tuning pass, not a chance to revisit VALIDATION.

### Data discipline
- **TRAIN** -- model fitting only (`fit_locked_model`).
- **VALIDATION** -- read only for its size (transparency in run metadata),
  never scored, never passed into `evaluate_on_test`.
- **TEST** -- scored exactly once. `evaluate_on_test()`'s signature has no
  `val_df` parameter at all -- scoring VALIDATION isn't expressible
  through this function, structurally, not just by convention.
- Enforced run-once safeguard: `run()` writes a lock file
  (`docs/.phase7_test_evaluation.lock`) after a successful evaluation. A
  second call without `force=True` raises `RuntimeError`. This is a real,
  tested guarantee, not a comment asking future-us to be careful.

### Reused components
`ml/dataset.py` (`assemble_feature_label_table`, `FEATURE_COLUMNS`,
`PRIMARY_TARGET_COLUMN`), `ml/split.py` (`chronological_split`,
`validate_split_partition`), `ml/models.py` (`GradientBoostingModel`),
`ml/threshold.py` (`evaluate_threshold` -- the exact same single-threshold
scoring function already used and tested in Phase 6, so TEST is scored
with identical logic rather than a parallel reimplementation that could
subtly disagree). No modifications to any Phase 2-6 file.

### Locked operating point (from Phase 6, not re-derived here)
- Model: Gradient Boosting (`HistGradientBoostingClassifier`)
- Threshold: 0.70
- VALIDATION F1 at this operating point (Phase 6, reference only): 0.7340
- VALIDATION PR-AUC at this operating point (Phase 6, reference only): 0.8176

### Results
### Results

The locked model (HistGradientBoostingClassifier) with threshold **0.70** was evaluated **exactly once** on the held-out TEST set.

| Metric                     | Value     |
|----------------------------|-----------|
| Test Size                  | 233,852   |
| Test Positive Rate         | 18.99%    |
| Precision                  | 0.7001    |
| Recall                     | 0.7352    |
| **F1 Score**               | **0.7172**|
| Predicted Positive Rate    | 19.95%    |
| ROC-AUC                    | 0.9311    |
| PR-AUC                     | 0.7985    |

**Confusion Matrix**
- True Positive:  32,656
- True Negative: 175,445
- False Positive: 13,992
- False Negative: 11,759

### Interpretation

- The model generalizes reasonably well from VALIDATION to TEST.
- VALIDATION F1 at threshold 0.70 was 0.7340 → TEST F1 is 0.7172 (small drop, expected and acceptable).
- Precision and Recall are well balanced.
- ROC-AUC (0.9311) and PR-AUC (0.7985) remain strong.
- No further tuning was performed on TEST (as required by project discipline).
- Evaluation lock file was created: `docs/.phase7_test_evaluation.lock`

**Status: Phase 7 complete. TEST evaluation is now frozen.**
### Interpretation
[To be filled in after the real TEST results are available -- comparing
TEST performance to the VALIDATION reference numbers above, and noting
whether the gap (if any) is consistent with normal generalization or
suggests something else worth investigating.]

### Testing
New: `tests/test_ml_test_evaluation.py`. Hand-computed correctness checks
for `evaluate_on_test` (not just trusting library calls), a test proving
the passed threshold argument is actually respected, structural
signature/source checks confirming VALIDATION can never reach the scoring
function, and -- the core Phase 7-specific guarantee -- tests proving the
lock-file mechanism actually refuses a second unforced evaluation and
actually permits one with explicit `force=True`. Full suite result and
exact new/total test counts are reported in the implementation response,
not fabricated here.

### Next (not started)
Once real TEST results are in: write the Interpretation section above,
decide whether EmberRisk's Phase 4-7 modeling work is complete as-is or
warrants a documented limitation/follow-up, and only then consider whether
Phase 8 (orchestration/dashboard/AI layer, per the original roadmap) is
worth starting. No further modeling iteration should occur without a new,
explicit decision to reopen VALIDATION-based selection -- TEST having been
seen changes what any further tuning would mean.
