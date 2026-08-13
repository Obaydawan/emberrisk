"""
Target generation: for each (cell_id, T), future_fire_{H}d = 1 if a
qualifying FIRMS detection occurs anywhere in [T+1, T+H] for that cell,
else 0.

END-OF-PERIOD BOUNDARY (critical, explicit, not silent):
  For T within the final H days of the modeling period, [T+1, T+H] extends
  past 2025-12-31, where V1 fire data doesn't exist. These rows get
  future_fire_{H}d = <NA> (pandas nullable int, not 0 or 1 -- a fabricated
  label would be worse than a missing one) rather than being silently
  dropped from the table. The count of such rows is attached to the
  returned DataFrame's .attrs for the caller to log/document, and it is the
  caller's responsibility to exclude null-labeled rows before training.
"""
import pandas as pd


def build_targets(fire_daily, modeling_start, modeling_end, horizons=(3, 7, 14)):
    """
    fire_daily: DataFrame [cell_id, date, fire_count] covering at least
    modeling_start..modeling_end (no lookahead beyond the modeling period
    is needed -- any row that would need it gets a null label instead).
    """
    df = fire_daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["has_fire"] = (df["fire_count"] > 0).astype(int)

    outputs = {}
    for h in horizons:
        usable_end = modeling_end - pd.Timedelta(days=h)
        parts = []
        for cell_id, g in df.groupby("cell_id", sort=False):
            g = g.sort_values("date").set_index("date")
            # trailing(T) = sum(has_fire[T-h+1 .. T]); shifting by -h gives
            # trailing(T+h) = sum(has_fire[T+1 .. T+h]) at position T --
            # i.e. exactly the forward-looking window, without any reversed
            # rolling trickery that's easy to get backwards.
            trailing = g["has_fire"].rolling(window=h, min_periods=h).sum()
            future = trailing.shift(-h)
            # IMPORTANT: (future > 0) alone would silently turn NaN into
            # False/0 (NaN > 0 evaluates to False in pandas/numpy), which is
            # exactly the fabricated-label bug this function must avoid.
            # Compute the boolean label, then explicitly re-null the
            # positions where future itself was unknown.
            null_mask = future.isna()
            label = (future > 0).astype("Int64")
            label[null_mask] = pd.NA
            parts.append(pd.DataFrame({
                "cell_id": cell_id,
                "date": g.index,
                f"future_fire_{h}d": label.values,
            }))

        horizon_df = pd.concat(parts, ignore_index=True)
        horizon_df = horizon_df[
            (horizon_df["date"] >= modeling_start) & (horizon_df["date"] <= modeling_end)
        ].reset_index(drop=True)

        n_excluded = int((horizon_df["date"] > usable_end).sum())
        horizon_df.attrs["n_excluded_end_of_period"] = n_excluded
        horizon_df.attrs["usable_end_date"] = str(usable_end.date())
        outputs[h] = horizon_df

    return outputs
