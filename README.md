# EmberRisk

** wildfire risk intelligence pipeline — reproducible data engineering and machine learning, built with strict experimental discipline.**

EmberRisk predicts whether a given location (grid cell) will experience a fire within the next 7 days (`future_fire_7d`), using historical wildfire detections combined with weather and environmental data. The project covers the full lifecycle of a practical ML pipeline: data ingestion, validation, feature engineering, model comparison, threshold selection, and a final, one-time evaluation on a held-out TEST set.

A core engineering principle drives the project as much as the modeling result itself: a strict, code-enforced separation between model development and final evaluation. The TEST set was scored **exactly once**, after the model and decision threshold were fully locked using only the VALIDATION set — a rule enforced structurally in code (`ml/test_evaluation.py`), not just by convention.

## Data

EmberRisk uses two public environmental data sources:

- **NASA FIRMS** — wildfire detection data
- **NASA POWER** — weather and environmental data

The study area is represented using a canonical grid of 323 cells, with data structured into a cell-day format spanning January 2018 – December 2025. The prediction target is `future_fire_7d`: whether a fire occurs in that cell within the next 7 days. Rows near the end of the dataset where this future window isn't fully observable are excluded.

## Pipeline Architecture

```
Ingestion (NASA FIRMS + NASA POWER APIs)
     |
     v
Processing (bronze -> silver -> gold layers)
     |
     v
Cell-day scaffold + spatial-temporal feature engineering
     |
     v
Future-fire target generation (future_fire_7d)
     |
     v
Temporal Split (TRAIN / VALIDATION / TEST)
     |
     v
Baseline + Model Comparison (Logistic Regression -> Random Forest -> Gradient Boosting)
     |
     v
Threshold Selection + Calibration Analysis (VALIDATION only)
     |
     v
Locked Model (HistGradientBoostingClassifier @ threshold 0.70)
     |
     v
Final TEST Evaluation (one-time, frozen)
```

The processed dataset contains 943,806 cell-day rows before removing rows with unavailable future labels, resulting in a temporal split of TRAIN: 589,798 / VALIDATION: 117,895 / TEST: 233,852.

## Final Results

**Locked configuration:**

| Item | Value |
|---|---|
| Model | HistGradientBoostingClassifier |
| Threshold | 0.70 |
| Selection basis | VALIDATION only |
| TEST evaluation | Performed exactly once |

**TEST set performance:**

| Metric | Value |
|---|---|
| F1 Score | 0.7172 |
| Precision | 0.7001 |
| Recall | 0.7352 |
| ROC-AUC | 0.9311 |
| PR-AUC | 0.7985 |

The gap between VALIDATION and TEST performance was small (F1 dropped ~2.3% relative) and consistent with normal temporal generalization — not overfitting or data leakage.

Full results, confusion matrix, and a detailed limitations discussion: [`docs/final-results-and-limitations.md`](docs/final-results-and-limitations.md)

## Tech Stack

Python, pandas, NumPy, scikit-learn, pytest, Git/GitHub, Linux/WSL2

## Hardware Constraints

Designed to run on a resource-constrained local machine (8 GB RAM, dual-core CPU, no dedicated GPU). This intentionally ruled out deep learning, Spark, Kubernetes, and unnecessary cloud infrastructure — the pipeline is built to be practical and reproducible on modest hardware.

## Setup & Reproduction

```bash
git clone https://github.com/Obaydawan/emberrisk.git
cd emberrisk
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest -q
```

## Project Status

**Phases 1-7 complete:** data ingestion, processing, feature engineering, temporal split, baseline and model comparison, threshold/calibration analysis, model lock, and final one-time TEST evaluation. All 187 automated tests passing. The TEST result is frozen and fully documented in [`docs/final-results-and-limitations.md`](docs/final-results-and-limitations.md).

**Planned / not yet implemented:**
- Pipeline orchestration (Apache Airflow)
- Lightweight prediction-serving layer (batch scorer or API)
- Basic monitoring/drift-detection documentation

## Scope

EmberRisk is an independent wildfire-risk data engineering and ML portfolio project. The focus is on a complete, reproducible, and honestly-evaluated pipeline — built and run entirely on local, resource-constrained hardware.
