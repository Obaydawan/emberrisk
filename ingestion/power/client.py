"""
PowerClient -- low-level NASA POWER daily point API wrapper.

Per docs/phase2-ingestion-design.md section 10: POWER's daily point API
takes one (lat, lon) pair per call and returns a full date-range time
series in a SINGLE response -- unlike FIRMS, there is no day-range
chunking limit here. The natural "chunk" unit for POWER ingestion is
therefore one grid cell (see ingestion/power/ingest.py), not a date
sub-range.

Locked parameters (per docs/phase2-ingestion-design.md / Phase 1 script,
not to be silently changed): T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR, WS2M,
community=AG.
"""

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger("emberrisk.ingestion.power")

# POWER's documented fill/sentinel value for missing or unavailable data
MISSING_VALUE_THRESHOLD = -900  # values <= this are treated as missing/fill


class PowerApiError(Exception):
    """Definitive API-level failure (bad params, malformed response, point
    out of range, etc). NOT retried -- retrying wouldn't change the
    outcome."""


class PowerCoverageError(Exception):
    """The API call succeeded but the returned series does not cover the
    expected number of days for the requested range. Distinct from
    PowerApiError because the request itself succeeded -- this is a data
    quality problem, not a network/API problem."""


class PowerClient:
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

    def __init__(self, parameters, community="AG", max_retries=3,
                 backoff_seconds=5, request_timeout=60):
        if not parameters:
            raise ValueError("At least one POWER parameter is required")
        self.parameters = list(parameters)
        self.community = community
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.request_timeout = request_timeout

    def fetch_point_series(self, lat, lon, start_d, end_d):
        """Fetch the full daily time series for one point across
        [start_d, end_d] in a single call. Retries transient network
        failures with exponential backoff; raises PowerApiError immediately
        (no retry) on a definitive API-level error."""
        params = {
            "parameters": ",".join(self.parameters),
            "community": self.community,
            "latitude": lat,
            "longitude": lon,
            "start": start_d.strftime("%Y%m%d"),
            "end": end_d.strftime("%Y%m%d"),
            "format": "JSON",
        }

        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "POWER request attempt %d/%d: (%.4f, %.4f) %s..%s",
                    attempt, self.max_retries, lat, lon, start_d, end_d,
                )
                resp = requests.get(self.BASE_URL, params=params, timeout=self.request_timeout)
                resp.raise_for_status()
                data = resp.json()

                if "properties" not in data or "parameter" not in data.get("properties", {}):
                    # POWER returns HTTP 200 with an error structure/messages
                    # field for many logical failures rather than a non-200
                    # status -- must check response shape explicitly.
                    raise PowerApiError(
                        f"Unexpected POWER response shape (missing properties.parameter): "
                        f"{str(data)[:300]}"
                    )

                param_data = data["properties"]["parameter"]
                if not param_data:
                    raise PowerApiError("POWER response contained no parameter data")

                df = pd.DataFrame(param_data)
                df.index.name = "date"
                return df

            except PowerApiError:
                raise  # definitive, no retry
            except (requests.exceptions.RequestException, ValueError) as e:
                # ValueError covers resp.json() decode failures as well as
                # malformed-but-200 responses
                last_exception = e
                logger.warning(
                    "Transient failure on attempt %d/%d: %s", attempt, self.max_retries, e
                )
                if attempt < self.max_retries:
                    sleep_time = self.backoff_seconds * (2 ** (attempt - 1))
                    logger.info("Retrying in %ds...", sleep_time)
                    time.sleep(sleep_time)

        raise PowerApiError(
            f"POWER request failed after {self.max_retries} attempts: {last_exception}"
        )


def validate_coverage(df, start_d, end_d):
    """Raise PowerCoverageError if the returned series doesn't have exactly
    one row per day in [start_d, end_d]. This is the empirical POWER
    coverage check called for in docs/phase2-ingestion-design.md section 2 --
    it runs on every cell, not just the first one."""
    expected_days = (end_d - start_d).days + 1
    actual_days = len(df)
    if actual_days != expected_days:
        raise PowerCoverageError(
            f"Expected {expected_days} daily rows for {start_d}..{end_d}, "
            f"got {actual_days}"
        )
    return True


def compute_missingness(df, threshold=MISSING_VALUE_THRESHOLD):
    """Per-column percentage of sentinel/fill values (POWER uses -999 for
    missing data). Returned alongside every successfully ingested cell so
    missingness is visible per-cell rather than only in a separate sample
    check."""
    missing_pct = {}
    for col in df.columns:
        n_missing = (df[col] <= threshold).sum()
        missing_pct[col] = round(100 * n_missing / len(df), 4) if len(df) else None
    return missing_pct
