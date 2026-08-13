"""
Phase 3 pipeline: raw FIRMS + POWER -> canonical cell-day dataset + targets.

Usage:
  python -m processing.pipeline --out-dir data/processed

Reads from data/raw/firms and data/raw/power (Phase 2 output), writes:
  {out_dir}/cell_day_dataset.parquet   -- features, modeling period only
  {out_dir}/targets_h{H}.parquet       -- one file per horizon (3/7/14)
  {out_dir}/validation_report.json     -- full data-quality check report
"""
import argparse
import json
import logging
from pathlib import Path

from processing import MODELING_START, MODELING_END, WARMUP_START, HORIZONS
from processing.grid_scaffold import build_scaffold
from processing.firms_standardize import load_raw_firms, standardize_firms, aggregate_to_cell_day
from processing.fire_features import fill_scaffold_with_fire_counts, compute_fire_rolling_features
from processing.power_standardize import load_raw_power, standardize_power
from processing.build_dataset import build_cell_day_dataset
from processing.targets import build_targets
from processing.validate import run_all_checks

logger = logging.getLogger("emberrisk.processing.pipeline")


def run(out_dir="data/processed"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw FIRMS...")
    raw_firms = load_raw_firms()
    logger.info("Raw FIRMS rows: %d", len(raw_firms))

    std_firms = standardize_firms(raw_firms)
    n_qualifying = int(std_firms["qualifying"].sum())
    logger.info("Qualifying FIRMS rows: %d", n_qualifying)

    firms_daily = aggregate_to_cell_day(std_firms)

    logger.info("Building warm-up-extended scaffold (%s .. %s)...", WARMUP_START.date(), MODELING_END.date())
    extended_scaffold = build_scaffold(WARMUP_START, MODELING_END)
    filled = fill_scaffold_with_fire_counts(extended_scaffold, firms_daily)

    logger.info("Computing rolling fire-history features...")
    fire_features = compute_fire_rolling_features(filled)

    logger.info("Loading raw POWER...")
    raw_power = load_raw_power()
    power_std = standardize_power(raw_power)

    logger.info("Joining fire features (trimmed to modeling period) with POWER...")
    dataset = build_cell_day_dataset(fire_features, power_std, MODELING_START, MODELING_END)

    dataset_path = out_dir / "cell_day_dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    logger.info("Wrote %s (%d rows)", dataset_path, len(dataset))

    logger.info("Building targets for horizons %s...", HORIZONS)
    fire_daily_for_targets = filled[["cell_id", "date", "fire_count"]]
    targets = build_targets(fire_daily_for_targets, MODELING_START, MODELING_END, HORIZONS)

    for h, tdf in targets.items():
        path = out_dir / f"targets_h{h}.parquet"
        tdf.to_parquet(path, index=False)
        logger.info("Wrote %s (%d rows, %d excluded at end-of-period boundary)",
                    path, len(tdf), tdf.attrs.get("n_excluded_end_of_period"))

    logger.info("Running validation checks...")
    report = run_all_checks(dataset, targets, power_df=power_std)
    report_path = out_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote %s", report_path)

    for name, result in report.items():
        status = "PASS" if result["passed"] else "FAIL"
        logger.info("[%s] %s: %s", status, name, result["detail"])

    return dataset, targets, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="EmberRisk Phase 3: cell-day dataset construction")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()
    run(out_dir=args.out_dir)
