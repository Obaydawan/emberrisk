"""
NASA POWER historical ingestion orchestrator -- Phase 2 milestone, mirrors
the FIRMS ingestion architecture (manifest-based resumability, idempotent
output, retry/backoff via PowerClient), adapted for POWER's different
request shape.

Key structural difference from FIRMS (per docs/phase2-ingestion-design.md
section 10): POWER has no day-range chunk limit, so the "chunk" unit here
is one GRID CELL fetched for the full requested date range in a single
call, not a date sub-range. For the full historical backfill this means
323 calls (one per canonical grid cell), not ~591 date-chunked calls.

THIS FILE DOES NOT RUN THAT BACKFILL. The caller controls date range and
cell selection via CLI args (--max-cells / --cell-id let you restrict to a
tiny smoke test), and the intended first real use is ONE cell over a short
period, not all 323 cells over 2018-2025.

Locked parameters/community/date-window/grid are NOT changed here -- see
docs/phase2-ingestion-design.md sections 3, 5, 10.
"""

import argparse
import logging
from datetime import date
from pathlib import Path

from ingestion.common.manifest import IngestionManifest
from ingestion.common.grid import enumerate_grid_cells, CA_BBOX
from ingestion.power.client import (
    PowerClient,
    PowerApiError,
    PowerCoverageError,
    validate_coverage,
    compute_missingness,
)

logger = logging.getLogger("emberrisk.ingestion.power")

# Locked per Phase 1 validation script / Phase 2 design -- not silently changed
DEFAULT_PARAMETERS = ["T2M_MAX", "T2M_MIN", "RH2M", "PRECTOTCORR", "WS2M"]
DEFAULT_COMMUNITY = "AG"
RAW_DIR = Path("data/raw/power")


def chunk_id_for(cell_id, start_d, end_d):
    return f"{cell_id}_{start_d.isoformat()}_{end_d.isoformat()}"


def run_ingestion(start_d, end_d, bbox=CA_BBOX, parameters=None,
                   community=DEFAULT_COMMUNITY, raw_dir=RAW_DIR,
                   dry_run=False, max_cells=None, only_cell_id=None,
                   client=None):
    parameters = list(parameters) if parameters else list(DEFAULT_PARAMETERS)

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "_manifest.json"
    manifest = IngestionManifest(manifest_path)

    if client is None:
        client = PowerClient(parameters=parameters, community=community)

    cells = enumerate_grid_cells(bbox)
    if only_cell_id is not None:
        cells = [c for c in cells if c.cell_id == only_cell_id]
        if not cells:
            raise ValueError(f"cell_id {only_cell_id!r} not found in canonical grid")
    if max_cells is not None:
        cells = cells[:max_cells]

    logger.info("Planned %d cell(s) covering %s to %s (params=%s, community=%s)",
                len(cells), start_d, end_d, parameters, community)

    results = {"completed": 0, "skipped": 0, "failed": 0}

    for cell in cells:
        cid = chunk_id_for(cell.cell_id, start_d, end_d)

        if manifest.is_complete(cid):
            logger.info("Skipping already-complete cell-chunk %s", cid)
            results["skipped"] += 1
            continue

        if dry_run:
            logger.info("[dry-run] Would fetch cell %s (centroid=%.4f,%.4f)",
                         cell.cell_id, cell.centroid_lat, cell.centroid_lon)
            continue

        manifest.mark_in_progress(cid)
        try:
            df = client.fetch_point_series(cell.centroid_lat, cell.centroid_lon, start_d, end_d)
            validate_coverage(df, start_d, end_d)
            missingness = compute_missingness(df)

            output_path = raw_dir / f"power_{cell.cell_id}_{start_d.isoformat()}_{end_d.isoformat()}.parquet"
            df.reset_index().to_parquet(output_path, index=False)

            manifest.mark_complete(
                cid, output_path, row_count=len(df),
                extra={"cell_id": cell.cell_id, "missingness_pct": missingness},
            )
            logger.info("Cell %s complete: %d rows -> %s, missingness=%s",
                         cell.cell_id, len(df), output_path, missingness)
            results["completed"] += 1

        except (PowerApiError, PowerCoverageError) as e:
            manifest.mark_failed(cid, e)
            logger.error("Cell %s failed: %s", cell.cell_id, e)
            results["failed"] += 1

    logger.info("Ingestion run summary: %s", results)
    return results


def _parse_args():
    parser = argparse.ArgumentParser(
        description="EmberRisk POWER ingestion (Phase 2 testable milestone)"
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--community", default=DEFAULT_COMMUNITY)
    parser.add_argument("--dry-run", action="store_true",
                         help="Plan cell list without calling the API")
    parser.add_argument("--max-cells", type=int, default=None,
                         help="TESTING ONLY: restrict to the first N grid cells")
    parser.add_argument("--cell-id", default=None,
                         help="TESTING ONLY: restrict to a single grid cell, e.g. '9_-199'")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    run_ingestion(
        start_d=date.fromisoformat(args.start),
        end_d=date.fromisoformat(args.end),
        community=args.community,
        dry_run=args.dry_run,
        max_cells=args.max_cells,
        only_cell_id=args.cell_id,
    )
