"""
api/main.py -- Phase 10: minimal prediction-serving API for the locked
EmberRisk model.

IMPORTANT SCOPE NOTE (see docs/phase10-serving-api.md for full detail):
EmberRisk's modeling period is a fixed, locked historical window
(2018-01-01 to 2025-12-31, per processing.MODELING_START/MODELING_END).
This API does NOT predict "today's" fire risk from live data -- it scores
whatever feature values are supplied in the request (real historical rows
via /predict/sample, or caller-supplied hypothetical values via /predict)
using the Phase 6/7 locked model. It is a serving-layer demonstration, not
a live risk tool.

The model is loaded ONCE at startup (see the lifespan handler below) and
held in memory -- not reloaded per request. All scoring goes through
ml.predict.score_batch(), the same function used by the Phase 9 Airflow
DAG's score_batch task, so the API and the batch pipeline can never
silently diverge in how they score a row.
"""
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from ml.dataset import FEATURE_COLUMNS, assemble_feature_label_table
from ml.predict import load_locked_model, score_batch

logger = logging.getLogger("emberrisk.api")

# Populated at startup by the lifespan handler below. Module-level so
# request handlers can read it without re-loading the model each call.
_model_state = {"model": None, "metadata": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading locked EmberRisk model...")
    model, metadata = load_locked_model()
    _model_state["model"] = model
    _model_state["metadata"] = metadata
    logger.info(
        "Model loaded: %s (threshold=%s, trained on %d rows)",
        metadata["model_name"], metadata["locked_threshold"], metadata["n_train_rows"],
    )
    yield
    _model_state["model"] = None
    _model_state["metadata"] = None


app = FastAPI(
    title="EmberRisk Prediction API",
    description=(
        "Serves predictions from EmberRisk's locked wildfire-risk model "
        "(HistGradientBoostingClassifier, threshold 0.70). Scores supplied "
        "feature values against a fixed, historical (2018-2025) modeling "
        "period -- see docs/phase10-serving-api.md for scope details."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model():
    if _model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return _model_state["model"], _model_state["metadata"]


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_model_state["model"] is not None,
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    _, metadata = _require_model()
    return ModelInfoResponse(**metadata)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    model, metadata = _require_model()

    n_rows = len(request.rows)
    if request.cell_ids is not None and len(request.cell_ids) != n_rows:
        raise HTTPException(
            status_code=422,
            detail=f"cell_ids length ({len(request.cell_ids)}) must match rows length ({n_rows}).",
        )
    if request.dates is not None and len(request.dates) != n_rows:
        raise HTTPException(
            status_code=422,
            detail=f"dates length ({len(request.dates)}) must match rows length ({n_rows}).",
        )

    feature_dicts = [row.model_dump() for row in request.rows]
    feature_df = pd.DataFrame(feature_dicts, columns=FEATURE_COLUMNS)

    if request.cell_ids is not None:
        feature_df["cell_id"] = request.cell_ids
    if request.dates is not None:
        feature_df["date"] = request.dates

    scored_df = score_batch(model, feature_df, threshold=metadata["locked_threshold"])

    predictions = [
        PredictionResult(
            cell_id=row.get("cell_id"),
            date=str(row["date"]) if "date" in row else None,
            predicted_probability=float(row["predicted_probability"]),
            predicted_positive=bool(row["predicted_positive"]),
        )
        for row in scored_df.to_dict(orient="records")
    ]

    return PredictionResponse(
        model_name=metadata["model_name"],
        locked_threshold=metadata["locked_threshold"],
        n_scored=len(predictions),
        predictions=predictions,
    )


@app.get("/predict/sample/{cell_id}/{date}", response_model=PredictionResponse)
def predict_sample(cell_id: str, date: str):
    """Convenience endpoint: scores a REAL historical (cell_id, date) row
    from the assembled feature table, for demoing without hand-building a
    feature payload. Only works for dates within the locked modeling
    period (2018-01-01 to 2025-12-31) -- this is historical replay, not a
    live prediction, per the module-level scope note above."""
    model, metadata = _require_model()

    table, _report = assemble_feature_label_table()
    match = table[(table["cell_id"] == cell_id) & (table["date"].astype(str) == date)]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No row found for cell_id={cell_id!r}, date={date!r}. "
                f"Note: only dates within the locked modeling period "
                f"(2018-01-01 to 2025-12-31) exist in this dataset."
            ),
        )

    scored_df = score_batch(model, match, threshold=metadata["locked_threshold"])
    row = scored_df.iloc[0].to_dict()

    return PredictionResponse(
        model_name=metadata["model_name"],
        locked_threshold=metadata["locked_threshold"],
        n_scored=1,
        predictions=[
            PredictionResult(
                cell_id=row.get("cell_id"),
                date=str(row.get("date")),
                predicted_probability=float(row["predicted_probability"]),
                predicted_positive=bool(row["predicted_positive"]),
            )
        ],
    )
