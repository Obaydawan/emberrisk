"""
FirmsClient -- low-level FIRMS Area API wrapper.

Confirmed constraint (empirically verified in Phase 1, documented in
docs/phase2-ingestion-design.md): FIRMS's day_range parameter accepts a
maximum of 5 days per request. This client enforces that limit rather than
trusting the caller.
"""

import logging
import time
from io import StringIO

import pandas as pd
import requests

logger = logging.getLogger("emberrisk.ingestion.firms")

MAX_DAY_RANGE = 5


class FirmsApiError(Exception):
    """Raised for definitive API-level failures (bad key, bad params, no
    data). These are NOT retried, since retrying wouldn't change the
    outcome."""


class FirmsClient:
    BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

    def __init__(self, map_key, source="VIIRS_SNPP_SP", max_retries=3,
                 backoff_seconds=5, request_timeout=60):
        if not map_key:
            raise ValueError("FIRMS map_key is required")
        self.map_key = map_key
        self.source = source
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.request_timeout = request_timeout

    def _build_url(self, bbox, day_range, start_date):
        w, s, e, n = bbox
        area = f"{w},{s},{e},{n}"
        return f"{self.BASE_URL}/{self.map_key}/{self.source}/{area}/{day_range}/{start_date.isoformat()}"

    def fetch_chunk(self, bbox, day_range, start_date):
        """
        Fetch a single chunk (<=5 days). Retries transient network failures
        with exponential backoff. Raises FirmsApiError immediately (no
        retry) on a definitive API-level error, since a bad key or invalid
        request will fail identically on every retry.
        """
        if day_range > MAX_DAY_RANGE:
            raise ValueError(
                f"day_range={day_range} exceeds FIRMS's confirmed "
                f"{MAX_DAY_RANGE}-day limit"
            )

        url = self._build_url(bbox, day_range, start_date)
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "FIRMS request attempt %d/%d: %s + %dd, source=%s",
                    attempt, self.max_retries, start_date, day_range, self.source,
                )
                resp = requests.get(url, timeout=self.request_timeout)
                resp.raise_for_status()
                text = resp.text

                # FIRMS returns HTTP 200 with an error message as CSV text
                # on logical failures (bad key, no data) rather than a
                # non-200 HTTP status -- must check the body explicitly.
                if text.startswith("Invalid") or text.strip() == "":
                    raise FirmsApiError(
                        f"FIRMS API returned an error/empty response: {text[:300]!r}"
                    )

                df = pd.read_csv(StringIO(text))
                return df

            except FirmsApiError:
                raise  # definitive, no retry
            except (requests.exceptions.RequestException, pd.errors.ParserError) as e:
                last_exception = e
                logger.warning(
                    "Transient failure on attempt %d/%d: %s", attempt, self.max_retries, e
                )
                if attempt < self.max_retries:
                    sleep_time = self.backoff_seconds * (2 ** (attempt - 1))
                    logger.info("Retrying in %ds...", sleep_time)
                    time.sleep(sleep_time)

        raise FirmsApiError(
            f"FIRMS request failed after {self.max_retries} attempts: {last_exception}"
        )
