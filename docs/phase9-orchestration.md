# EmberRisk Phase 9 — Pipeline Orchestration (Apache Airflow)

## Objective

Wrap EmberRisk's existing, already-locked pipeline (ingestion → processing
→ model persistence → batch scoring) in a scheduled, observable Airflow
DAG. This phase orchestrates *around* the Phase 6/7 locked model — it does
not retrain, retune, or reopen any modeling decision.

## Scope: fixed historical period, not a live pipeline

EmberRisk's modeling period is a fixed, locked constant
(`processing.MODELING_START = 2018-01-01`, `processing.MODELING_END =
2025-12-31`, `processing.EXPECTED_CELL_DAY_COUNT = 943,806`). This DAG
orchestrates that existing pipeline on a schedule — it does **not** ingest
or score genuinely new/live 2026+ data. Doing so would require
deliberately extending `MODELING_END` and the associated validation
constants, which is an explicit, separate future decision, not something
this DAG makes silently by running on a calendar schedule.

Because of this, the DAG is scheduled `@weekly` rather than daily: a daily
cadence would misrepresent what the pipeline actually does, since no new
real-world data enters the system between runs. What a repeated run
demonstrates instead is idempotency — ingestion tasks skip already-complete
work via the existing manifest system, and `processing.pipeline.run()`
deterministically rebuilds the same dataset from the same raw inputs.

## Architecture

```
ingest_firms  ─┐
               ├─→ run_processing_pipeline → validate_pipeline_output
ingest_power  ─┘                                      │
                                                        ▼
                                          train_and_save_model
                                                        │
                                                        ▼
                                                  score_batch
                                                        │
                                                        ▼
                                              data_quality_report
```

Every task calls existing, already-tested project code directly
(`ingestion.firms.ingest`, `ingestion.power.ingest`, `processing.pipeline`,
`ml.train_and_save_locked_model`, `ml.predict`). No pipeline or modeling
logic is reimplemented inside the DAG file — it is orchestration only.

### Task descriptions

- **`ingest_firms` / `ingest_power`** — call the existing manifest-based
  ingestion modules for the full locked historical range. Idempotent: a
  rerun against already-complete chunks mostly reports `skipped` rather
  than re-downloading.
- **`run_processing_pipeline`** — calls `processing.pipeline.run()`
  unmodified. This is a full rebuild of `cell_day_dataset.parquet` and
  the target tables from raw data every run, not an incremental merge.
  At this data scale (~60MB raw, dataset built in well under a minute),
  full-rebuild-on-schedule is simpler and more correct than incremental
  merge logic that isn't needed yet — a deliberate scale-appropriate
  choice, not an oversight.
- **`validate_pipeline_output`** — reads the validation report written by
  `processing.pipeline.run()` and fails the task loudly if any check
  did not pass. A real data-quality gate: `train_and_save_model` and
  everything downstream will not run on data that failed validation.
- **`train_and_save_model`** — fits the locked `GradientBoostingModel`
  (`HistGradientBoostingClassifier`, threshold 0.70) on TRAIN and persists
  it via `joblib` to `models/gradient_boosting_locked.joblib`. Introduces
  model persistence to the project (Phases 5-7 always fit fresh within the
  same run that used the model); see "Model persistence" below.
- **`score_batch`** — loads the persisted model and scores the full
  assembled feature table via the shared `select_features()` guard,
  writing timestamped predictions to `data/predictions/`.
- **`data_quality_report`** — logs row count and predicted positive rate
  for the run as a final sanity check.

## Model persistence (new in Phase 9)

No prior phase saved a model artifact — Phases 5, 6, and 7 each fit
`GradientBoostingModel` fresh, within the same script run that used it,
because each was a one-time analysis pass. A scheduled DAG that might run
repeatedly needs training and serving separated, so Phase 9 introduces
`ml/train_and_save_locked_model.py` (fits and saves) and `ml/predict.py`
(loads and scores). This persists the **same locked decision** — same
model class, same hyperparameters, same TRAIN split — it is not a new
model-selection decision.

## Infrastructure

- **Orchestrator**: Apache Airflow 2.10.4, via Docker Compose
  (`docker-compose.airflow.yml`), reusing the pattern from the author's
  prior TransactSafe project.
- **Executor**: `LocalExecutor`, not `CeleryExecutor`. This project
  targets an 8GB RAM / dual-core machine; Celery's additional
  broker/worker processes are unnecessary overhead at this scale.
  LocalExecutor still provides real parallel task execution on one
  machine, which is sufficient here.
- **Project code access**: the EmberRisk project directory is bind-mounted
  into the Airflow containers (`.:/opt/airflow/project`), so DAG tasks
  import `ingestion/`, `processing/`, and `ml/` directly — the same code
  that runs outside Docker, not a copy.

## Known environment discrepancy: scikit-learn version

The host `.venv` (used for all Phase 4-8 results, including the locked
Phase 7 TEST evaluation) runs `scikit-learn==1.9.0`, pinned in
`requirements.txt`. During initial Airflow setup, the container's base
image shipped incompatible pre-installed `numpy`/`pandas`/`scipy`
versions that conflicted with the pinned requirements, causing repeated
`train_and_save_model` failures (`ModuleNotFoundError`, then
`numpy`/`pandas`/`scipy` ABI conflicts). Getting the container to a
working state required installing a compatible combination directly
inside it: `pandas==2.1.4`, `numpy==1.26.4`, `scikit-learn==1.4.2`,
`scipy==1.12.0`.

**This means the model artifact produced by the Airflow container is
trained under `scikit-learn==1.4.2`, not the `1.9.0` used for every
reported result in this project** (Phase 5 comparison, Phase 6 threshold
sweep, Phase 7 TEST evaluation). `HistGradientBoostingClassifier` can
produce marginally different fitted output across sklearn minor versions
even with identical hyperparameters and training data, due to internal
implementation details (binning, floating-point accumulation order).

This is a real, acknowledged limitation, not a silently ignored one: the
**locked, reported EmberRisk results remain those computed on the host
venv under sklearn 1.9.0** (`docs/final-results-and-limitations.md`). The
containerized artifact demonstrates that the orchestration mechanics work
end-to-end; it is not being presented as a re-verification of the Phase 7
TEST numbers. A future improvement would pin the container's base image
or install step to exactly match `requirements.txt`, eliminating this
discrepancy — noted as follow-up work, not resolved in this phase.

## How to run locally

```bash
# Start Airflow (first run pulls the image and installs requirements.txt)
docker compose -f docker-compose.airflow.yml up airflow-init
docker compose -f docker-compose.airflow.yml up -d

# Airflow UI: http://localhost:8080 (user: admin / password: admin)
# Trigger the "emberrisk_pipeline" DAG manually from the UI, or:
docker exec --user airflow <scheduler-container-name> \
    airflow dags trigger emberrisk_pipeline

# Tear down
docker compose -f docker-compose.airflow.yml down
```

## What this phase does not do

- Does not retrain, retune, or re-select the locked model/threshold.
- Does not touch VALIDATION or TEST.
- Does not ingest or score live/current (2026+) data.
- Does not change any Phase 4-8 result or documentation.

## Status

All seven DAG tasks (`ingest_firms`, `ingest_power`,
`run_processing_pipeline`, `validate_pipeline_output`,
`train_and_save_model`, `score_batch`, `data_quality_report`) execute
successfully end-to-end. Phase 9 orchestration is functionally complete,
with the sklearn version discrepancy above documented as known follow-up
work rather than left unaddressed.
