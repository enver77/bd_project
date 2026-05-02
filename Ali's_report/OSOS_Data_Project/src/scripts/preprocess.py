"""
Tier 0 — Preprocessing
Converts raw hourly CSV to a clean daily feature table.

Output: data/processed/preprocessed_daily.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "processed" / "cleaned_asos_data.csv"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "preprocessed_daily.csv"


# ---------------------------------------------------------------------------
# Step 1 — Load raw data and build DatetimeIndex
# ---------------------------------------------------------------------------

def load_raw(path: Path) -> pd.DataFrame:
    from calendar import monthrange
    df = pd.read_csv(path)

    # Drop rows with out-of-range days (e.g. Feb 30, Apr 31) — all have Çekiş=0
    def _valid_date(row):
        try:
            m, y = int(row["Dönem"].split("/")[0]), int(row["Dönem"].split("/")[1])
            _, max_day = monthrange(y, m)
            return int(row["Gün"]) <= max_day
        except Exception:
            return False

    valid_mask = df.apply(_valid_date, axis=1)
    n_dropped = (~valid_mask).sum()
    if n_dropped:
        print(f"  Dropping {n_dropped} rows with invalid dates (padding rows)")
    df = df[valid_mask].copy()

    # Parse period + day + hour → timestamp (start of each hour)
    def _parse_row(row):
        month_str, year_str = row["Dönem"].split("/")
        day = int(row["Gün"])
        hour = int(str(row["Saat"]).split("-")[0])
        return pd.Timestamp(int(year_str), int(month_str), day, hour)

    df["timestamp"] = df.apply(_parse_row, axis=1)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    return df[["Çekiş"]].rename(columns={"Çekiş": "kwh"})


# ---------------------------------------------------------------------------
# Step 2 — Build complete hourly grid with gap interpolation
# ---------------------------------------------------------------------------

def build_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full_idx)

    # Mark imputed_long before filling
    gap_mask = df["kwh"].isna()
    # Find consecutive NaN runs
    gap_groups = (gap_mask != gap_mask.shift()).cumsum()
    gap_sizes = gap_mask.groupby(gap_groups).transform("sum")
    df["imputed_long"] = (gap_mask & (gap_sizes > 3)).astype(int)

    # Linear interpolation for short gaps (≤3 hours), leave long gaps NaN for now
    df["kwh"] = df["kwh"].interpolate(method="linear", limit=3)
    # Fill remaining NaN with 0 (long gaps — meter off)
    df["kwh"] = df["kwh"].fillna(0.0)
    df["kwh"] = df["kwh"].clip(lower=0.0)

    return df


# ---------------------------------------------------------------------------
# Step 3 — Empirical night/day mask (per day, seasonal)
# ---------------------------------------------------------------------------

def compute_active_mask(df: pd.DataFrame, threshold: float = 1.0) -> pd.Series:
    """Return boolean Series: True when hour is 'active' (street light on)."""
    return df["kwh"] >= threshold


# ---------------------------------------------------------------------------
# Step 4 — Build daily feature table
# ---------------------------------------------------------------------------

def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
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

        # night_missing_frac: fraction of hours < 0.5 kWh among expected night hours
        # Expected night hours: hours where median across all same-hour days > 1.0
        # (simplified: count hours that are NOT active as fraction of 24)
        night_missing_frac = (all_kwh < 0.5).sum() / 24.0

        records.append({
            "date": dt,
            "daily_active_kwh": daily_active_kwh,
            "active_hours_count": active_hours_count,
            "mean_active_intensity": mean_active_intensity,
            "p10_active": p10_active,
            "night_missing_frac": night_missing_frac,
            "has_imputed_long": day_df["imputed_long"].max(),
        })

    daily = pd.DataFrame(records).set_index("date")
    daily.index = pd.DatetimeIndex(daily.index)

    # Rolling 28-day median baseline
    daily["rolling_28d_median"] = (
        daily["daily_active_kwh"]
        .rolling(window=28, min_periods=7, center=False)
        .median()
        .shift(1)   # no look-ahead
    )
    # Forward-fill first window where rolling is NaN
    daily["rolling_28d_median"] = daily["rolling_28d_median"].bfill()

    daily["daily_kwh_norm"] = daily["daily_active_kwh"] / daily["rolling_28d_median"].replace(0, np.nan)
    daily["daily_kwh_norm"] = daily["daily_kwh_norm"].fillna(0.0)

    # Lag features
    daily["delta_1d"] = daily["daily_active_kwh"].diff(1)
    daily["delta_7d"] = daily["daily_active_kwh"].diff(7)

    # Rolling 7-day std
    daily["rolling_7d_std"] = (
        daily["daily_active_kwh"].rolling(7, min_periods=3).std()
    )

    # Rolling 28-day z-score
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading raw data …")
    hourly = load_raw(RAW_CSV)
    print(f"  Raw rows: {len(hourly)}, range: {hourly.index.min()} → {hourly.index.max()}")

    print("Building hourly grid …")
    hourly = build_hourly_grid(hourly)
    print(f"  Grid rows: {len(hourly)}")

    print("Building daily feature table …")
    daily = build_daily_features(hourly)
    print(f"  Daily rows: {len(daily)}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT_CSV)
    print(f"Saved → {OUT_CSV}")

    # Basic sanity check
    assert len(daily) >= 400, f"Expected ≥400 days, got {len(daily)}"
    print("✓ Sanity check passed")
    return daily


if __name__ == "__main__":
    main()
