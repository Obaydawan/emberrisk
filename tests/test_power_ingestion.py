"""
Unit tests for the POWER ingestion client. All network calls are mocked --
these tests verify request construction, coverage validation, missingness
computation, manifest, and retry LOGIC only. They do not and cannot verify
real POWER API behavior -- that requires a live run against the real API,
done separately on the local machine.
"""

import json
from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import requests

from ingestion.common.manifest import IngestionManifest
from ingestion.common.grid import enumerate_grid_cells, cell_id_for, CA_BBOX, GRID_LAT_DEG, GRID_LON_DEG
from ingestion.power.client import (
    PowerClient, PowerApiError, PowerCoverageError,
    validate_coverage, compute_missingness,
)
from ingestion.power.ingest import run_ingestion, chunk_id_for


# ---------------------------------------------------------------------------
# Grid logic (shared with FIRMS, re-verified here since POWER depends on it)
# ---------------------------------------------------------------------------

def test_grid_cell_count_matches_phase1():
    cells = enumerate_grid_cells(CA_BBOX)
    assert len(cells) == 323  # confirmed Phase 1 finding


def test_grid_cell_ids_are_unique():
    cells = enumerate_grid_cells(CA_BBOX)
    ids = [c.cell_id for c in cells]
    assert len(ids) == len(set(ids))


def test_grid_centroid_within_cell_bounds():
    cells = enumerate_grid_cells(CA_BBOX)
    w, s, e, n = CA_BBOX
    for c in cells[:20]:  # spot-check a sample, not all 323
        assert w <= c.centroid_lon <= e
        assert s <= c.centroid_lat <= n


def test_cell_id_for_matches_enumeration():
    cells = enumerate_grid_cells(CA_BBOX)
    sample = cells[0]
    recomputed = cell_id_for(sample.centroid_lat, sample.centroid_lon, GRID_LAT_DEG, GRID_LON_DEG)
    assert recomputed == sample.cell_id


# ---------------------------------------------------------------------------
# Coverage validation
# ---------------------------------------------------------------------------

def test_validate_coverage_passes_for_exact_match():
    df = pd.DataFrame(index=pd.date_range("2018-01-01", "2018-01-05"))
    assert validate_coverage(df, date(2018, 1, 1), date(2018, 1, 5)) is True


def test_validate_coverage_raises_on_short_series():
    df = pd.DataFrame(index=pd.date_range("2018-01-01", "2018-01-03"))  # 3 rows, expected 5
    with pytest.raises(PowerCoverageError):
        validate_coverage(df, date(2018, 1, 1), date(2018, 1, 5))


def test_validate_coverage_raises_on_empty():
    df = pd.DataFrame()
    with pytest.raises(PowerCoverageError):
        validate_coverage(df, date(2018, 1, 1), date(2018, 1, 5))


# ---------------------------------------------------------------------------
# Missingness computation
# ---------------------------------------------------------------------------

def test_compute_missingness_all_present():
    df = pd.DataFrame({"T2M_MAX": [20.1, 21.3, 19.8], "RH2M": [55.0, 60.2, 58.1]})
    result = compute_missingness(df)
    assert result == {"T2M_MAX": 0.0, "RH2M": 0.0}


def test_compute_missingness_detects_fill_values():
    # POWER's documented fill/sentinel value is -999
    df = pd.DataFrame({"T2M_MAX": [20.1, -999.0, 19.8, -999.0]})
    result = compute_missingness(df)
    assert result["T2M_MAX"] == 50.0  # 2 of 4 rows are fill values


# ---------------------------------------------------------------------------
# Client request construction + retry/backoff logic
# ---------------------------------------------------------------------------

VALID_POWER_RESPONSE = {
    "properties": {
        "parameter": {
            "T2M_MAX": {"20180101": 18.2, "20180102": 19.1},
            "T2M_MIN": {"20180101": 5.4, "20180102": 6.0},
            "RH2M": {"20180101": 55.3, "20180102": 58.9},
            "PRECTOTCORR": {"20180101": 0.0, "20180102": 0.0},
            "WS2M": {"20180101": 2.1, "20180102": 1.9},
        }
    }
}


def _mock_response(json_body, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_body
    resp.raise_for_status.side_effect = (
        requests.exceptions.HTTPError(f"{status} error") if status >= 400 else None
    )
    return resp


def test_client_builds_correct_request_params():
    client = PowerClient(
        parameters=["T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M"],
        community="AG", max_retries=1, backoff_seconds=0,
    )
    with patch("ingestion.power.client.requests.get",
               return_value=_mock_response(VALID_POWER_RESPONSE)) as m:
        client.fetch_point_series(37.0, -120.0, date(2018, 1, 1), date(2018, 1, 2))

    _, kwargs = m.call_args
    sent_params = kwargs["params"]
    assert sent_params["parameters"] == "T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,WS2M"
    assert sent_params["community"] == "AG"
    assert sent_params["latitude"] == 37.0
    assert sent_params["longitude"] == -120.0
    assert sent_params["start"] == "20180101"
    assert sent_params["end"] == "20180102"
    assert sent_params["format"] == "JSON"


def test_client_success_returns_expected_shape():
    client = PowerClient(parameters=["T2M_MAX"], max_retries=1, backoff_seconds=0)
    with patch("ingestion.power.client.requests.get",
               return_value=_mock_response(VALID_POWER_RESPONSE)):
        df = client.fetch_point_series(37.0, -120.0, date(2018, 1, 1), date(2018, 1, 2))
    assert len(df) == 2
    assert "T2M_MAX" in df.columns


def test_client_retries_then_succeeds():
    client = PowerClient(parameters=["T2M_MAX"], max_retries=3, backoff_seconds=0)
    side_effects = [
        requests.exceptions.ConnectionError("network blip"),
        _mock_response(VALID_POWER_RESPONSE),
    ]
    with patch("ingestion.power.client.requests.get", side_effect=side_effects) as m:
        df = client.fetch_point_series(37.0, -120.0, date(2018, 1, 1), date(2018, 1, 2))
    assert m.call_count == 2
    assert len(df) == 2


def test_client_definitive_error_does_not_retry():
    client = PowerClient(parameters=["T2M_MAX"], max_retries=3, backoff_seconds=0)
    bad_response = {"error": "invalid parameters"}  # missing 'properties' entirely
    with patch("ingestion.power.client.requests.get",
               return_value=_mock_response(bad_response)) as m:
        with pytest.raises(PowerApiError):
            client.fetch_point_series(37.0, -120.0, date(2018, 1, 1), date(2018, 1, 2))
    assert m.call_count == 1  # no retry on a definitive API-level error


def test_client_exhausts_retries_and_raises():
    client = PowerClient(parameters=["T2M_MAX"], max_retries=3, backoff_seconds=0)
    with patch("ingestion.power.client.requests.get",
               side_effect=requests.exceptions.ConnectionError("down")) as m:
        with pytest.raises(PowerApiError):
            client.fetch_point_series(37.0, -120.0, date(2018, 1, 1), date(2018, 1, 2))
    assert m.call_count == 3


# ---------------------------------------------------------------------------
# End-to-end orchestration (mocked client) -- idempotent rerun + manifest
# ---------------------------------------------------------------------------

def test_run_ingestion_single_cell_completes_and_records_missingness(tmp_path, monkeypatch):
    raw_dir = tmp_path / "power_raw"
    call_log = []

    def fake_fetch(self, lat, lon, start_d, end_d):
        call_log.append((lat, lon))
        idx = pd.date_range(start_d, end_d)
        return pd.DataFrame({"T2M_MAX": [20.0] * len(idx)}, index=idx)

    monkeypatch.setattr(PowerClient, "fetch_point_series", fake_fetch)

    results = run_ingestion(
        date(2018, 1, 1), date(2018, 1, 5),
        raw_dir=raw_dir, max_cells=1,
    )
    assert results == {"completed": 1, "skipped": 0, "failed": 0}
    assert len(call_log) == 1

    with open(raw_dir / "_manifest.json") as f:
        manifest_data = json.load(f)
    entry = list(manifest_data.values())[0]
    assert entry["status"] == "complete"
    assert entry["row_count"] == 5
    assert "missingness_pct" in entry
    assert entry["missingness_pct"]["T2M_MAX"] == 0.0


def test_run_ingestion_idempotent_rerun(tmp_path, monkeypatch):
    raw_dir = tmp_path / "power_raw"
    call_log = []

    def fake_fetch(self, lat, lon, start_d, end_d):
        call_log.append((lat, lon))
        idx = pd.date_range(start_d, end_d)
        return pd.DataFrame({"T2M_MAX": [20.0] * len(idx)}, index=idx)

    monkeypatch.setattr(PowerClient, "fetch_point_series", fake_fetch)

    run_ingestion(date(2018, 1, 1), date(2018, 1, 5), raw_dir=raw_dir, max_cells=2)
    assert len(call_log) == 2

    results_2 = run_ingestion(date(2018, 1, 1), date(2018, 1, 5), raw_dir=raw_dir, max_cells=2)
    assert results_2 == {"completed": 0, "skipped": 2, "failed": 0}
    assert len(call_log) == 2  # unchanged -- no re-fetch


def test_run_ingestion_records_coverage_failure_without_crashing_run(tmp_path, monkeypatch):
    raw_dir = tmp_path / "power_raw"

    def fake_fetch_short_series(self, lat, lon, start_d, end_d):
        # Returns fewer rows than expected -- simulates a real coverage gap
        idx = pd.date_range(start_d, end_d)[:2]
        return pd.DataFrame({"T2M_MAX": [20.0, 21.0]}, index=idx)

    monkeypatch.setattr(PowerClient, "fetch_point_series", fake_fetch_short_series)

    results = run_ingestion(
        date(2018, 1, 1), date(2018, 1, 10),  # 10 days expected, only 2 returned
        raw_dir=raw_dir, max_cells=1,
    )
    assert results == {"completed": 0, "skipped": 0, "failed": 1}

    with open(raw_dir / "_manifest.json") as f:
        manifest_data = json.load(f)
    entry = list(manifest_data.values())[0]
    assert entry["status"] == "failed"


def test_run_ingestion_only_cell_id_filters_to_one_cell(tmp_path, monkeypatch):
    raw_dir = tmp_path / "power_raw"
    all_cells = enumerate_grid_cells(CA_BBOX)
    target = all_cells[5].cell_id

    def fake_fetch(self, lat, lon, start_d, end_d):
        idx = pd.date_range(start_d, end_d)
        return pd.DataFrame({"T2M_MAX": [20.0] * len(idx)}, index=idx)

    monkeypatch.setattr(PowerClient, "fetch_point_series", fake_fetch)

    results = run_ingestion(
        date(2018, 1, 1), date(2018, 1, 2),
        raw_dir=raw_dir, only_cell_id=target,
    )
    assert results == {"completed": 1, "skipped": 0, "failed": 0}


def test_run_ingestion_unknown_cell_id_raises(tmp_path):
    with pytest.raises(ValueError):
        run_ingestion(
            date(2018, 1, 1), date(2018, 1, 2),
            raw_dir=tmp_path / "power_raw", only_cell_id="not_a_real_cell",
        )
