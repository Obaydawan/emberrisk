"""
Joins warm-up-aware fire-history features (trimmed to the modeling period)
with standardized POWER weather to produce the canonical cell-day dataset.
"""
import pandas as pd


def build_cell_day_dataset(fire_features_df, power_df, modeling_start, modeling_end):
    fire_trimmed = fire_features_df[
        (fire_features_df["date"] >= modeling_start) & (fire_features_df["date"] <= modeling_end)
    ].copy()

    dataset = fire_trimmed.merge(power_df, on=["cell_id", "date"], how="left")
    return dataset
