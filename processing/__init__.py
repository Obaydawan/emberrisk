"""
Phase 3: standardization + canonical cell-day dataset construction.

Locked constants (per docs/phase2-ingestion-design.md, not redecided here):
"""
import pandas as pd

MODELING_START = pd.Timestamp("2018-01-01")
MODELING_END = pd.Timestamp("2025-12-31")

WARMUP_START = pd.Timestamp("2017-12-02")
WARMUP_END = pd.Timestamp("2017-12-31")

HORIZONS = (3, 7, 14)

EXPECTED_CELL_COUNT = 323
EXPECTED_CELL_DAY_COUNT = 943_806  # 323 cells x 2,922 modeling days
