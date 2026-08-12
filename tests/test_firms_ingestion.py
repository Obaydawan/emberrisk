"""
Unit tests for the FIRMS ingestion client. All network calls are mocked --
these tests verify chunking, manifest, and retry LOGIC only. They do not
and cannot verify real FIRMS API behavior (real schema, real confidence
values, real dates) -- that requires a live run against the real API with
a real MAP_KEY, done separately on the local machine.
"""

import json
from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import requests

from ingestion.common.manifest import IngestionManifest
from ingestion.firms.client import FirmsClient, FirmsApiError, MAX_DAY_RANGE
from ingestion.firms.ingest import generate_chunks, chunk_id_for, run_ingestion


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------

def test_generate_chunks_exact_multiple():
    chunks = list(generate_chunks(date(2018, 1, 1), date(2018, 1, 10)))
    assert len(chunks) == 2
    assert chunks[0] == (date(2018, 1, 1), date(2018, 1, 5), 5)
    assert chunks[1] == (date(2018, 1, 6), date(2018, 1, 10), 5)


def test_generate_chunks_remainder():
    chunks = list(generate_chunks(date(2018, 1, 1), date(2018, 1, 7)))
    assert len(chunks) == 2
    assert chunks[0] == (date(2018, 1, 1), date(2018, 1, 5), 5)
    assert chunks[1] == (date(2018, 1, 6), date(2018, 1, 7), 2)  # partial final chunk


def test_generate_chunks_single_day():
    chunks = list(generate_chunks(date(2018, 1, 1), date(2018, 1, 1)))
    assert chunks == [(date(2018, 1, 1), date(2018, 1, 1), 1)]


def test_generate_chunks_covers_full_range_no_gaps_no_overlap():
    start, end = date(2018, 1, 1), date(2018, 3, 17)  # arbitrary non-round range
    chunks = list(generate_chunks(start, end))
    covered_days = set()
    for chunk_start, chunk_end, day_range in chunks:
        assert (chunk_end - chunk_start).days + 1 == day_range
        assert day_range <= MAX_DAY_RANGE
        d = chunk_start
        while d <= chunk_end:
            assert d not in covered_days, f"date {d} covered by multiple chunks"
            covered_days.add(d)
            d += pd.Timedelta(days=1)
    expected_days = (end - start).days + 1
    assert len(covered_days) == expected_days


def test_generate_chunks_rejects_inverted_range():
    with pytest.raises(ValueError):
        list(generate_chunks(date(2018, 1, 10), date(2018, 1, 1)))


# ---------------------------------------------------------------------------
# Manifest logic
# ---------------------------------------------------------------------------

def test_manifest_starts_empty(tmp_path):
    m = IngestionManifest(tmp_path / "_manifest.json")
    assert m.status("2018-01-01_2018-01-05") == "pending"
    assert not m.is_complete("2018-01-01_2018-01-05")


def test_manifest_complete_lifecycle(tmp_path):
    m = IngestionManifest(tmp_path / "_manifest.json")
    cid = "2018-01-01_2018-01-05"

    m.mark_in_progress(cid)
    assert m.status(cid) == "in_progress"
    assert not m.is_complete(cid)

    m.mark_complete(cid, output_path="data/raw/firms/x.parquet", row_count=42)
    assert m.is_complete(cid)

    # persisted to disk correctly
    with open(tmp_path / "_manifest.json") as f:
        data = json.load(f)
    assert data[cid]["status"] == "complete"
    assert data[cid]["row_count"] == 42


def test_manifest_failed_is_not_complete(tmp_path):
    m = IngestionManifest(tmp_path / "_manifest.json")
    cid = "2018-01-01_2018-01-05"
    m.mark_failed(cid, error="boom")
    assert m.status(cid) == "failed"
    assert not m.is_complete(cid)


def test_manifest_survives_reload(tmp_path):
    path = tmp_path / "_manifest.json"
    m1 = IngestionManifest(path)
    m1.mark_complete("chunk_a", "out/a.parquet", row_count=10)

    m2 = IngestionManifest(path)  # simulate a fresh process loading the same file
    assert m2.is_complete("chunk_a")


# ---------------------------------------------------------------------------
# Client retry/backoff logic
# ---------------------------------------------------------------------------

VALID_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight\n"
    "38.5,-120.5,330.2,0.4,0.4,2018-01-02,1030,N,VIIRS,n,2.0,290.1,5.3,D\n"
)


def _mock_response(text, status=200):
    resp = MagicMock()
    resp.text = text
    resp.status_code = status
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status} error")
    else:
        resp.raise_for_status.side_effect = None
    return resp


def test_client_success_first_try():
    client = FirmsClient(map_key="fake_key", max_retries=3, backoff_seconds=0)
    with patch("ingestion.firms.client.requests.get", return_value=_mock_response(VALID_CSV)) as m:
        df = client.fetch_chunk((-124.5, 32.5, -114.0, 42.0), 5, date(2018, 1, 1))
    assert m.call_count == 1
    assert len(df) == 1
    assert df.iloc[0]["confidence"] == "n"


def test_client_retries_then_succeeds():
    client = FirmsClient(map_key="fake_key", max_retries=3, backoff_seconds=0)
    side_effects = [
        requests.exceptions.ConnectionError("network blip"),
        requests.exceptions.ConnectionError("network blip again"),
        _mock_response(VALID_CSV),
    ]
    with patch("ingestion.firms.client.requests.get", side_effect=side_effects) as m:
        df = client.fetch_chunk((-124.5, 32.5, -114.0, 42.0), 5, date(2018, 1, 1))
    assert m.call_count == 3  # two failures, then success
    assert len(df) == 1


def test_client_exhausts_retries_and_raises():
    client = FirmsClient(map_key="fake_key", max_retries=3, backoff_seconds=0)
    with patch("ingestion.firms.client.requests.get",
               side_effect=requests.exceptions.ConnectionError("down")) as m:
        with pytest.raises(FirmsApiError):
            client.fetch_chunk((-124.5, 32.5, -114.0, 42.0), 5, date(2018, 1, 1))
    assert m.call_count == 3  # exactly max_retries attempts, no more


def test_client_definitive_error_does_not_retry():
    client = FirmsClient(map_key="bad_key", max_retries=3, backoff_seconds=0)
    with patch("ingestion.firms.client.requests.get",
               return_value=_mock_response("Invalid MAP_KEY")) as m:
        with pytest.raises(FirmsApiError):
            client.fetch_chunk((-124.5, 32.5, -114.0, 42.0), 5, date(2018, 1, 1))
    assert m.call_count == 1  # no retry on a definitive API-level error


def test_client_rejects_day_range_over_limit():
    client = FirmsClient(map_key="fake_key")
    with pytest.raises(ValueError):
        client.fetch_chunk((-124.5, 32.5, -114.0, 42.0), 10, date(2018, 1, 1))


# ---------------------------------------------------------------------------
# End-to-end orchestration (mocked client) -- idempotent rerun behavior
# ---------------------------------------------------------------------------

def test_run_ingestion_idempotent_rerun(tmp_path, monkeypatch):
    raw_dir = tmp_path / "firms_raw"

    call_log = []

    def fake_fetch_chunk(self, bbox, day_range, start_date):
        call_log.append(start_date)
        return pd.read_csv(pd.io.common.StringIO(VALID_CSV))

    monkeypatch.setattr(FirmsClient, "fetch_chunk", fake_fetch_chunk)

    # First run: should fetch and complete 1 chunk (single day range)
    results_1 = run_ingestion(
        date(2018, 1, 1), date(2018, 1, 1),
        raw_dir=raw_dir, map_key="fake_key",
    )
    assert results_1 == {"completed": 1, "skipped": 0, "failed": 0}
    assert len(call_log) == 1
    assert (raw_dir / "_manifest.json").exists()

    # Second run over the SAME range: should skip, not re-fetch
    results_2 = run_ingestion(
        date(2018, 1, 1), date(2018, 1, 1),
        raw_dir=raw_dir, map_key="fake_key",
    )
    assert results_2 == {"completed": 0, "skipped": 1, "failed": 0}
    assert len(call_log) == 1  # unchanged -- no second API call happened
