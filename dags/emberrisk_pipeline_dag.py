"""
dags/emberrisk_pipeline_dag.py -- Phase 9: orchestrates the existing,
locked EmberRisk pipeline (ingestion -> processing -> model persistence ->
batch scoring) as a scheduled Airflow DAG.

IMPORTANT SCOPE NOTE (see docs/phase9-orchestration.md for full detail):
EmberRisk's modeling period is a fixed, locked historical window
(processing.MODELING_START = 2018-01-01, processing.MODELING_END =
2025-12-31). This DAG orchestrates that existing pipeline on a schedule --
it does NOT ingest or score genuinely new/live data, because doing so
would require deliberately extending MODELING_END and the associated
validation constants (EXPECTED_CELL_DAY_COUNT etc.), which is an explicit
future decision, not something this DAG makes silently.

Every task below calls existing, already-tested project code directly
(ingestion.firms.ingest, ingestion.power.ingest, processing.pipeline,
ml.train_and_save_locked_model, ml.predict) -- no pipeline logic is
reimplemented here. This file is orchestration only.

Because ingestion is manifest-based and idempotent (ingestion/common/
manifest.py), re-running ingest_firms / ingest_power against the same,
already-complete historical range is fast and safe -- it mostly reports
"skipped" rather than re-downloading. This is a deliberate demonstration
of idempotent pipeline design, not wasted work.
"""
from datetime import date

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "emberrisk",
    "retries": 1,
}


def _task_ingest_firms(**context):
    from processing import MODELING_START, MODELING_END
    from ingestion.firms.ingest import run_ingestion

    results = run_ingestion(
        start_d=MODELING_START.date(),
        end_d=MODELING_END.date(),
    )
    context["ti"].xcom_push(key="firms_ingest_results", value=results)
    return results


def _task_ingest_power(**context):
    from processing import MODELING_START, MODELING_END
    from ingestion.power.ingest import run_ingestion

    results = run_ingestion(
        start_d=MODELING_START.date(),
        end_d=MODELING_END.date(),
    )
    context["ti"].xcom_push(key="power_ingest_results", value=results)
    return results


def _task_run_processing_pipeline(**context):
    from processing.pipeline import run as run_processing

    dataset, targets, report = run_processing(out_dir="data/processed")
    context["ti"].xcom_push(key="n_dataset_rows", value=len(dataset))
    return {"n_dataset_rows": len(dataset)}


def _task_validate_pipeline_output(**context):
    """Fails loudly if any check in the Phase 3 validation report did not
    pass -- a real data-quality gate, not a pass-through step. The report
    is written by processing.pipeline.run() itself (already-tested Phase
    3 logic); this task only reads and enforces it."""
    import json
    from pathlib import Path

    report_path = Path("data/processed/validation_report.json")
    if not report_path.exists():
        raise FileNotFoundError(
            f"Expected validation report not found at {report_path} -- "
            f"did run_processing_pipeline complete successfully?"
        )

    with open(report_path) as f:
        report = json.load(f)

    failed_checks = {
        name: result for name, result in report.items()
        if not result.get("passed", False)
    }
    if failed_checks:
        raise RuntimeError(
            f"Phase 3 validation report contains {len(failed_checks)} failed "
            f"check(s), refusing to proceed to model training/scoring: "
            f"{failed_checks}"
        )

    return {"n_checks": len(report), "all_passed": True}


def _task_train_and_save_model(**context):
    from ml.train_and_save_locked_model import train_locked_model, save_model

    model, metadata = train_locked_model()
    model_path, metadata_path = save_model(model, metadata)
    context["ti"].xcom_push(key="model_path", value=str(model_path))
    return metadata


def _task_score_batch(**context):
    from ml.dataset import assemble_feature_label_table
    from ml.predict import load_locked_model, score_batch, write_predictions

    model, metadata = load_locked_model()

    # Score the full assembled feature/label table (id columns preserved).
    # This mirrors what Phase 7's TEST evaluation and the locked model were
    # built against -- same feature assembly path, not a parallel one.
    table, _assemble_report = assemble_feature_label_table()

    scored_df = score_batch(model, table, threshold=metadata["locked_threshold"])
    run_date = context["ds"]  # Airflow execution date, YYYY-MM-DD
    out_path = write_predictions(scored_df, run_date=run_date)

    context["ti"].xcom_push(key="predictions_path", value=str(out_path))
    context["ti"].xcom_push(key="n_scored_rows", value=len(scored_df))
    return {"n_scored_rows": len(scored_df), "predictions_path": str(out_path)}


def _task_data_quality_report(**context):
    """Lightweight final summary: row count, positive rate, output file
    existence check. Logged to Airflow task logs -- not a separate report
    file, since this is a sanity check on the run, not a new artifact."""
    from pathlib import Path

    import pandas as pd

    ti = context["ti"]
    predictions_path = ti.xcom_pull(key="predictions_path", task_ids="score_batch")
    n_scored_rows = ti.xcom_pull(key="n_scored_rows", task_ids="score_batch")

    path = Path(predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Expected predictions file not found at {path}")

    scored_df = pd.read_parquet(path)
    positive_rate = float(scored_df["predicted_positive"].mean()) if len(scored_df) else None

    summary = {
        "predictions_path": str(path),
        "n_scored_rows": n_scored_rows,
        "predicted_positive_rate": positive_rate,
    }
    print(f"EmberRisk DAG run summary: {summary}")
    return summary


with DAG(
    dag_id="emberrisk_pipeline",
    description=(
        "Orchestrates EmberRisk's locked, historical (2018-2025) pipeline: "
        "ingestion -> processing -> validation -> model persistence -> "
        "batch scoring. Does not process live/current data -- see "
        "docs/phase9-orchestration.md."
    ),
    default_args=default_args,
    # Not scheduled daily: this pipeline does not ingest new real-world
    # data (fixed historical modeling period), so a daily cadence would
    # misrepresent what the DAG actually does. Weekly is a reasonable,
    # honest default for demonstrating repeatable, idempotent execution.
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    tags=["emberrisk", "phase9"],
) as dag:

    ingest_firms = PythonOperator(
        task_id="ingest_firms",
        python_callable=_task_ingest_firms,
    )

    ingest_power = PythonOperator(
        task_id="ingest_power",
        python_callable=_task_ingest_power,
    )

    run_processing_pipeline = PythonOperator(
        task_id="run_processing_pipeline",
        python_callable=_task_run_processing_pipeline,
    )

    validate_pipeline_output = PythonOperator(
        task_id="validate_pipeline_output",
        python_callable=_task_validate_pipeline_output,
    )

    train_and_save_model = PythonOperator(
        task_id="train_and_save_model",
        python_callable=_task_train_and_save_model,
    )

    score_batch = PythonOperator(
        task_id="score_batch",
        python_callable=_task_score_batch,
    )

    data_quality_report = PythonOperator(
        task_id="data_quality_report",
        python_callable=_task_data_quality_report,
    )

    [ingest_firms, ingest_power] >> run_processing_pipeline
    run_processing_pipeline >> validate_pipeline_output
    validate_pipeline_output >> train_and_save_model
    train_and_save_model >> score_batch
    score_batch >> data_quality_report
