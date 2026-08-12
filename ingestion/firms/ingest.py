"""
FIRMS historical ingestion orchestrator -- Phase 2 first testable milestone.

Chunks a date range into <=5-day windows (FIRMS's confirmed limit), fetches
each via FirmsClient, writes one Parquet file per chunk, and tracks
completion in an IngestionManifest for resumability and idempotent reruns.

THIS FILE ONLY IMPLEMENTS THE INGESTION CLIENT. It does not run the full
2017-12-02 to 2025-12-31 backfill on its own -- the caller controls the
date range via CLI args, and the intended first real use is a tiny smoke
test (e.g. 2018-01-01 to 2018-01-05), not the full historical range.
"""

import argparse
import logging
import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from ingestion.common.manifest import IngestionManifest
from ingestion.firms.client import FirmsClient, FirmsApiError, MAX_DAY_RANGE

logger = logging.getLogger("emberrisk.ingestion.firms")

CA_BBOX = (-124.5, 32.5, -114.0, 42.0)
DEFAULT_SOURCE = "VIIRS_SNPP_SP"
RAW_DIR = Path("data/raw/firms")


def chunk_id_for(start_d, end_d):
    return f"{start_d.isoformat()}_{end_d.isoformat()}"


def generate_chunks(start_d, end_d, chunk_days=MAX_DAY_RANGE):
    """Yield (chunk_start, chunk_end, day_range) tuples covering
    [start_d, end_d] inclusive, each chunk at most chunk_days long."""
    if start_d > end_d:
        raise ValueError(f"start_d ({start_d}) must not be after end_d ({end_d})")
    cursor = start_d
    while cursor <= end_d:
        remaining = (end_d - cursor).days + 1
        day_range = min(chunk_days, remaining)
        chunk_end = cursor + timedelta(days=day_range - 1)
        yield cursor, chunk_end, day_range
        cursor += timedelta(days=day_range)


def run_ingestion(start_d, end_d, bbox=CA_BBOX, source=DEFAULT_SOURCE,
                   raw_dir=RAW_DIR, dry_run=False, map_key=None):
    if map_key is None:
        load_dotenv()
        map_key = os.getenv("FIRMS_MAP_KEY")
    if not map_key:
        raise SystemExit("FIRMS_MAP_KEY not set in environment (.env)")

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "_manifest.json"
    manifest = IngestionManifest(manifest_path)
    client = FirmsClient(map_key=map_key, source=source)

    chunks = list(generate_chunks(start_d, end_d))
    logger.info("Planned %d chunk(s) covering %s to %s", len(chunks), start_d, end_d)

    results = {"completed": 0, "skipped": 0, "failed": 0}

    for chunk_start, chunk_end, day_range in chunks:
        cid = chunk_id_for(chunk_start, chunk_end)

        if manifest.is_complete(cid):
            logger.info("Skipping already-complete chunk %s", cid)
            results["skipped"] += 1
            continue

        if dry_run:
            logger.info("[dry-run] Would fetch chunk %s (day_range=%d)", cid, day_range)
            continue

        manifest.mark_in_progress(cid)
        try:
            df = client.fetch_chunk(bbox, day_range, chunk_start)
            output_path = raw_dir / f"firms_{source}_{cid}.parquet"
            df.to_parquet(output_path, index=False)
            manifest.mark_complete(cid, output_path, row_count=len(df))
            logger.info("Chunk %s complete: %d rows -> %s", cid, len(df), output_path)
            results["completed"] += 1
        except FirmsApiError as e:
            manifest.mark_failed(cid, e)
            logger.error("Chunk %s failed: %s", cid, e)
            results["failed"] += 1

    logger.info("Ingestion run summary: %s", results)
    return results


def _parse_args():
    parser = argparse.ArgumentParser(
        description="EmberRisk FIRMS ingestion (Phase 2 testable milestone)"
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--dry-run", action="store_true",
                         help="Plan chunks without calling the API")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    run_ingestion(
        start_d=date.fromisoformat(args.start),
        end_d=date.fromisoformat(args.end),
        source=args.source,
        dry_run=args.dry_run,
    )
