"""
api/schemas.py -- Pydantic request/response models for the EmberRisk
prediction API.

The feature field names and set are deliberately NOT hand-duplicated here
-- FeatureRow's fields are generated from ml.dataset.FEATURE_COLUMNS at
import time, so this schema can never silently drift out of sync with the
single source of truth already used by every model in the project
(ml.features.select_features).
"""
from typing import List, Optional

from pydantic import BaseModel, Field, create_model

from ml.dataset import FEATURE_COLUMNS

# Every feature is a required float. Built dynamically from
# FEATURE_COLUMNS rather than listed by hand, so adding/removing a feature
# column in ml/dataset.py is the only place that ever needs to change --
# this schema follows automatically instead of silently going stale.
FeatureRow = create_model(
    "FeatureRow",
    **{col: (float, Field(..., description=f"Feature: {col}")) for col in FEATURE_COLUMNS},
)


class PredictionRequest(BaseModel):
    """One or more feature rows to score. cell_id/date are optional
    identifiers, echoed back in the response but never passed to the
    model -- mirrors ml.predict.score_batch()'s id_columns handling."""
    rows: List[FeatureRow] = Field(..., min_length=1)
    cell_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional cell_id per row, same length as rows, echoed back only.",
    )
    dates: Optional[List[str]] = Field(
        default=None,
        description="Optional date per row, same length as rows, echoed back only.",
    )


class PredictionResult(BaseModel):
    cell_id: Optional[str] = None
    date: Optional[str] = None
    predicted_probability: float
    predicted_positive: bool


class PredictionResponse(BaseModel):
    model_name: str
    locked_threshold: float
    n_scored: int
    predictions: List[PredictionResult]


class ModelInfoResponse(BaseModel):
    model_name: str
    locked_threshold: float
    feature_columns: List[str]
    target_column: str
    n_train_rows: int
    split_validated: bool
    note: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
