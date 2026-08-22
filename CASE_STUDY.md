# EmberRisk: An End-to-End Wildfire Risk Prediction Pipeline

**Data engineering + machine learning, built with production-grade discipline**

---

## The Problem

Wildfire risk prediction is a genuinely hard data problem: multiple public data sources with different update cadences and API constraints, spatial-temporal feature engineering across an 8-year period and 323 geographic cells, and a modeling task where getting the evaluation methodology wrong is easy — and where getting it wrong quietly produces numbers that look good but don't mean anything.

I built EmberRisk to predict whether a given location will experience a wildfire within the next 7 days, using NASA's public FIRMS (satellite fire detection) and POWER (weather/environmental) data. But the real project wasn't "train a model that predicts fires" — it was building the entire pipeline around that model the way a production system would need to: reproducibly, with rigorous evaluation discipline, and orchestrated so it isn't just a one-off notebook.

## What I Built

**A complete pipeline, not just a model:**

- **Ingestion** — resumable, idempotent clients for two different public APIs (NASA FIRMS, NASA POWER), each with different rate limits, chunking constraints, and data latency characteristics
- **Processing** — a medallion-style architecture turning raw satellite/weather data into a clean, validated cell-day feature dataset (943,806 rows), with automated data-quality checks that gate every downstream step
- **ML methodology** — strict TRAIN/VALIDATION/TEST separation with zero data leakage, model comparison (Logistic Regression → Random Forest → Gradient Boosting), threshold optimization, and a calibration analysis that was evaluated and *deliberately not acted on* because it didn't serve the actual product objective
- **A one-time, frozen final evaluation** — the test set was scored exactly once, enforced in code (not just by convention) via a lock-file mechanism that raises an error if anyone tries to re-run it
- **Orchestration** — the full pipeline runs as a scheduled Apache Airflow DAG (ingestion → processing → validation → model persistence → batch scoring), containerized with Docker Compose
- **A serving API** — a FastAPI service exposing the locked model for on-demand predictions, sharing the exact same scoring code path as the batch pipeline so the two can never silently disagree
- **187+ automated tests** covering every stage, from feature selection guards to lock-file enforcement to API input validation

## The Result

The final locked model (`HistGradientBoostingClassifier`, threshold 0.70) scored on a fully held-out test set:

| Metric | Value |
|---|---|
| F1 Score | 0.7172 |
| Precision | 0.7001 |
| Recall | 0.7352 |
| ROC-AUC | 0.9311 |
| PR-AUC | 0.7985 |

The gap between validation and test performance was small (~2.3% relative drop in F1) and consistent with normal generalization — not a red flag, and I documented exactly why rather than just reporting the headline number.

## What This Demonstrates

This project is a deliberate demonstration of skills that matter for real data engineering and ML work, not just modeling:

- **Reproducible pipelines** — anyone can clone the repo, install pinned dependencies, and run the exact same tests I ran
- **Evaluation discipline** — the difference between a model that looks good and a model whose performance claim is actually trustworthy
- **Orchestration** — scheduling, task dependencies, idempotent re-runs, and data-quality gates that block bad data from reaching the model
- **Honest documentation** — every limitation is written down, including two real environment issues I hit and solved (a dependency version mismatch between my local environment and the orchestration container, and NASA POWER's inherent data latency) rather than hidden
- **Serving infrastructure** — turning a trained model into something that can actually answer a request, not just sit in a notebook

## Services This Reflects

If you're looking for help with:
- **Data pipeline design and ingestion** from external/public APIs
- **ETL/ELT pipelines** with proper validation and data-quality gates
- **ML pipeline engineering** — reproducible training, evaluation, and model comparison workflows
- **Pipeline orchestration** with Apache Airflow (Docker-based, scheduled, idempotent)
- **Model-serving APIs** (FastAPI) for integrating ML predictions into other systems

— this project is a direct example of that work, built and documented to a standard I'd apply to client work.

**Full source, documentation, and results:** [github.com/Obaydawan/emberrisk](https://github.com/Obaydawan/emberrisk)
