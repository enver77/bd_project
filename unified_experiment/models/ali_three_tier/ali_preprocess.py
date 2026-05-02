"""
Ali's Tier 0 -- Daily Feature Builder
Adapted to read from the unified hourly_data.csv produced by unified_preprocessing.

Original: read raw cleaned_asos_data.csv -> build hourly grid -> daily features
Now:      read unified hourly_data.csv (already grid-aligned, mask-aware)
          -> compute Ali's specific daily features (active hours, intensity, etc.)

Output: RESULTS_DIR/ali_preprocessed_daily.csv
"""

import os
import numpy as np
import pandas as pd

RESULTS_DIR = os.environ.get("BEDAS_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))

HOURLY_CSV = os.path.join(RESULTS_DIR, "hourly_data.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "ali_preprocessed_daily.csv")


def compute_active_mask(df: pd.DataFrame, threshold: float = 1.0) -> pd.Series:
    """Active hour = consumption >= 1.0 kWh (street light on)."""
    return df["kwh"] >= threshold


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build Ali's daily feature table."""
    active = compute_active_mask(df)
    df["active"] = active.astype(int)

    records = []
    for date, day_df in df.groupby(df.index.date):
        dt = pd.Timestamp(date)
        act = day_df[day_df["active"] == 1]["kwh"]
        all_kwh = day_df["kwh"]

        daily_active_kwh = act.sum() if len(act) > 0 else 0.0
        active_hours_count = len(act)
        mean_active_intensity = act.mean() if len(act) > 0 else 0.0
        p10_active = act.quantile(0.10) if len(act) > 0 else 0.0
        night_missing_frac = (all_kwh < 0.5).sum() / 24.0

        records.append({
            "date": dt,
            "daily_active_kwh": daily_active_kwh,
            "active_hours_count": active_hours_count,
            "mean_active_intensity": mean_active_intensity,
            "p10_active": p10_active,
            "night_missing_frac": night_missing_frac,
            "has_imputed_long": int(day_df.get("was_imputed", pd.Series(0, index=day_df.index)).max()),
        })

    daily = pd.DataFrame(records).set_index("date")
    daily.index = pd.DatetimeIndex(daily.index)

    # Rolling 28-day median baseline (no look-ahead via shift(1))
    daily["rolling_28d_median"] = (
        daily["daily_active_kwh"]
        .rolling(window=28, min_periods=7, center=False)
        .median()
        .shift(1)
    )
    daily["rolling_28d_median"] = daily["rolling_28d_median"].bfill()

    daily["daily_kwh_norm"] = daily["daily_active_kwh"] / daily["rolling_28d_median"].replace(0, np.nan)
    daily["daily_kwh_norm"] = daily["daily_kwh_norm"].fillna(0.0)

    # Lag features
    daily["delta_1d"] = daily["daily_active_kwh"].diff(1)
    daily["delta_7d"] = daily["daily_active_kwh"].diff(7)

    # Rolling 7-day std
    daily["rolling_7d_std"] = daily["daily_active_kwh"].rolling(7, min_periods=3).std()

    # Rolling 28-day z-score (no look-ahead)
    roll28_mean = daily["daily_active_kwh"].rolling(28, min_periods=7).mean().shift(1).bfill()
    roll28_std  = daily["daily_active_kwh"].rolling(28, min_periods=7).std().shift(1).bfill()
    daily["rolling_28d_zscore"] = (daily["daily_active_kwh"] - roll28_mean) / roll28_std.replace(0, np.nan)
    daily["rolling_28d_zscore"] = daily["rolling_28d_zscore"].fillna(0.0)

    # Cyclic date features
    daily["day_of_year"] = daily.index.day_of_year
    daily["month"] = daily.index.month
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)
    daily["day_of_year_sin"] = np.sin(2 * np.pi * daily["day_of_year"] / 365)
    daily["day_of_year_cos"] = np.cos(2 * np.pi * daily["day_of_year"] / 365)

    return daily


def main():
    print("=" * 70)
    print(" Ali's Tier 0 -- Daily Feature Builder (Unified Data)")
    print("=" * 70)

    print(f"Loading unified hourly data from {HOURLY_CSV} ...")
    hourly = pd.read_csv(HOURLY_CSV, parse_dates=["timestamp"])
    hourly = hourly.set_index("timestamp")

    # Rename to Ali's expected column name
    hourly["kwh"] = hourly["energy_kwh"].clip(lower=0.0)
    print(f"  Hourly rows: {len(hourly)}, range: {hourly.index.min()} -> {hourly.index.max()}")

    print("Building daily feature table ...")
    daily = build_daily_features(hourly)
    print(f"  Daily rows: {len(daily)}")

    daily.to_csv(OUT_CSV)
    print(f"Saved -> {OUT_CSV}")

    assert len(daily) >= 400, f"Expected >=400 days, got {len(daily)}"
    print("Sanity check passed")
    return daily


if __name__ == "__main__":
    main()
