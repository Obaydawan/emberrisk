"""
EmberRisk canonical grid definition.

Per docs/phase2-ingestion-design.md section 5: this grid uses 0.5 deg
latitude x 0.625 deg longitude spacing to MATCH NASA POWER's native
meteorological resolution. It is the EmberRisk canonical grid, not a
claim of literal identity with POWER's internal grid boundaries.

This module is the single source of truth for grid cell IDs and centroids
so that FIRMS detection assignment and POWER cell sampling stay consistent
with each other and with the Phase 1 validation script's original logic
(same formulas, just factored out for reuse).
"""

from dataclasses import dataclass

CA_BBOX = (-124.5, 32.5, -114.0, 42.0)  # west, south, east, north
GRID_LAT_DEG = 0.5
GRID_LON_DEG = 0.625


@dataclass(frozen=True)
class GridCell:
    cell_id: str
    lat_index: int
    lon_index: int
    centroid_lat: float
    centroid_lon: float


def cell_id_for(lat, lon, grid_lat=GRID_LAT_DEG, grid_lon=GRID_LON_DEG):
    """Assign a raw (lat, lon) point to its grid cell ID. Used both to
    enumerate the canonical grid (via cell centroids) and to assign FIRMS
    detections to a cell."""
    lat_index = int(lat // grid_lat)
    lon_index = int(lon // grid_lon)
    return f"{lat_index}_{lon_index}"


def enumerate_grid_cells(bbox=CA_BBOX, grid_lat=GRID_LAT_DEG, grid_lon=GRID_LON_DEG):
    """Enumerate every cell in the bbox grid (323 cells for the locked CA
    bbox), regardless of whether it ever has a fire detection -- this is
    what lets downstream code treat zero-activity cells as valid negatives
    rather than silently omitting them (see Phase 1 validation script)."""
    w, s, e, n = bbox
    cells = []
    lat = s
    while lat < n:
        lon = w
        while lon < e:
            centroid_lat = lat + grid_lat / 2
            centroid_lon = lon + grid_lon / 2
            cid = cell_id_for(centroid_lat, centroid_lon, grid_lat, grid_lon)
            lat_index, lon_index = (int(p) for p in cid.split("_"))
            cells.append(GridCell(cid, lat_index, lon_index, centroid_lat, centroid_lon))
            lon += grid_lon
        lat += grid_lat
    return cells
