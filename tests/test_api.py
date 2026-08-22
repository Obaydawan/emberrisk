"""
tests/test_api.py -- Phase 10: tests for the EmberRisk prediction API.

Uses FastAPI's TestClient, which runs the app's lifespan handler (loading
the real locked model artifact) for the duration of the test session --
these tests exercise the actual model, not a mock, same principle as the
rest of this project's "verify against real behavior" discipline.

Requires the model artifact to exist (models/gradient_boosting_locked.*),
i.e. `PYTHONPATH=. python -m ml.train_and_save_locked_model` must have
been run at least once before these tests will pass.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from ml.dataset import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _valid_feature_row():
    """A syntactically valid feature row (arbitrary values) covering every
    required column -- built from FEATURE_COLUMNS itself so this test
    can't silently go stale if a feature column is added/removed."""
    return {col: 0.0 for col in FEATURE_COLUMNS}


class TestHealth:
    def test_health_returns_ok_and_model_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True


class TestModelInfo:
    def test_model_info_matches_locked_config(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        body = response.json()
        assert body["model_name"] == "gradient_boosting"
        assert body["locked_threshold"] == 0.70
        assert set(body["feature_columns"]) == set(FEATURE_COLUMNS)
        assert body["target_column"] == "future_fire_7d"
        assert body["n_train_rows"] > 0
        assert body["split_validated"] is True


class TestPredict:
    def test_predict_single_row_returns_valid_prediction(self, client):
        payload = {"rows": [_valid_feature_row()]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["n_scored"] == 1
        assert len(body["predictions"]) == 1
        pred = body["predictions"][0]
        assert 0.0 <= pred["predicted_probability"] <= 1.0
        assert isinstance(pred["predicted_positive"], bool)

    def test_predict_multiple_rows(self, client):
        payload = {"rows": [_valid_feature_row(), _valid_feature_row(), _valid_feature_row()]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["n_scored"] == 3

    def test_predict_echoes_cell_id_and_date(self, client):
        payload = {
            "rows": [_valid_feature_row()],
            "cell_ids": ["test_cell_123"],
            "dates": ["2026-01-01"],
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        pred = response.json()["predictions"][0]
        assert pred["cell_id"] == "test_cell_123"
        assert pred["date"] == "2026-01-01"

    def test_predict_threshold_consistency(self, client):
        """predicted_positive must always match (probability >= locked
        threshold) -- catches any future refactor that accidentally
        decouples the two rather than deriving one from the other."""
        response = client.post("/predict", json={"rows": [_valid_feature_row()]})
        body = response.json()
        pred = body["predictions"][0]
        expected_positive = pred["predicted_probability"] >= body["locked_threshold"]
        assert pred["predicted_positive"] == expected_positive

    def test_predict_missing_feature_column_returns_422(self, client):
        incomplete_row = {"fire_count": 0.0}  # missing the other 14 required columns
        response = client.post("/predict", json={"rows": [incomplete_row]})
        assert response.status_code == 422

    def test_predict_empty_rows_returns_422(self, client):
        response = client.post("/predict", json={"rows": []})
        assert response.status_code == 422

    def test_predict_mismatched_cell_ids_length_returns_422(self, client):
        payload = {
            "rows": [_valid_feature_row(), _valid_feature_row()],
            "cell_ids": ["only_one_id"],
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestPredictSample:
    def test_predict_sample_known_row(self, client):
        """65_-183 on 2018-01-01 is a real row confirmed present in the
        assembled feature table (verified manually during Phase 10
        testing)."""
        response = client.get("/predict/sample/65_-183/2018-01-01")
        assert response.status_code == 200
        body = response.json()
        assert body["n_scored"] == 1
        pred = body["predictions"][0]
        assert pred["cell_id"] == "65_-183"
        assert 0.0 <= pred["predicted_probability"] <= 1.0

    def test_predict_sample_unknown_cell_returns_404(self, client):
        response = client.get("/predict/sample/nonexistent_cell_id/2018-01-01")
        assert response.status_code == 404

    def test_predict_sample_date_outside_modeling_period_returns_404(self, client):
        """2026 dates fall outside the locked modeling period
        (2018-01-01 to 2025-12-31), so no row exists to score -- this is
        the expected, documented behavior, not a bug."""
        response = client.get("/predict/sample/65_-183/2026-08-22")
        assert response.status_code == 404
