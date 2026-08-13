"""
Cell-day scaffold: the canonical (cell_id, date) framework every feature and
label attaches to. Exists so a cell with zero fire detections on a given
day still gets a row -- NO FIRE DETECTION != MISSING DATA.
"""
import pandas as pd

from ingestion.common.grid import enumerate_grid_cells, CA_BBOX


def build_scaffold(start, end, bbox=CA_BBOX):
    """Build a (cell_id, date) scaffold for [start, end] inclusive. Used
    with the warm-up-extended range for fire-feature computation, and with
    the modeling-period-only range for the final dataset shape check."""
    cells = enumerate_grid_cells(bbox)
    cell_ids = [c.cell_id for c in cells]
    dates = pd.date_range(start, end, freq="D")

    scaffold = pd.MultiIndex.from_product(
        [cell_ids, dates], names=["cell_id", "date"]
    ).to_frame(index=False)
    return scaffold
