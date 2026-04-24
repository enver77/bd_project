#!/usr/bin/env python3
"""
Module 4 – Forecast Residual Anomaly Detection (Mask-Aware)
=============================================================
Computes residuals only for present (non-missing) hours.
Daily score = mean absolute residual of PRESENT hours only.
Days with <18 present hours get score = NaN.

Outputs:
  - forecast_residuals.csv   : hourly residuals (present hours only)
  - daily_anomaly_scores.csv : per-day aggregated anomaly scores
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

Z_THRESHOLD = 2.5
MIN_PRESENT_HOURS = 18  # Need at least 75% present hours for reliable score


def load_master():
    df = pd.read_csv(os.path.join(DATA_DIR, "master_dataset.csv"), parse_dates=["Date"])
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    return df


def build_baseline_model(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline from non-missing, non-AllDayZero hours."""
    # Use only present hours from active days
    active = df[(df["is_missing"] == 0)].copy()
    day_sums = active.groupby("Date")["Consumption_kWh"].sum()
    active_dates = day_sums[day_sums > 0].index
    train = active[active["Date"].isin(active_dates)].copy()

    baseline = train.groupby(["Hour", "Month", "IsWeekend"])["Consumption_kWh"].agg(
        Expected_kWh="median",
        Baseline_Std="std"
    ).reset_index()

    hour_std = train.groupby("Hour")["Consumption_kWh"].std().reset_index()
    hour_std.columns = ["Hour", "Hour_Std"]
    baseline = baseline.merge(hour_std, on="Hour", how="left")
    baseline["Baseline_Std"] = baseline["Baseline_Std"].fillna(baseline["Hour_Std"])
    baseline.drop(columns=["Hour_Std"], inplace=True)
    return baseline


def main():
    print("=" * 70)
    print(" Module 4 – Forecast Residual Anomaly (Mask-Aware)")
    print("=" * 70)

    df = load_master()
    print(f"\n📂 Loaded: {len(df)} rows, {df['Date'].nunique()} days")

    baseline = build_baseline_model(df)
    print(f"✅ Baseline model: {len(baseline)} groups")

    df = df.merge(baseline, on=["Hour", "Month", "IsWeekend"], how="left")
    df["Expected_kWh"] = df["Expected_kWh"].fillna(0)
    df["Baseline_Std"] = df["Baseline_Std"].fillna(df["Baseline_Std"].median())

    # Residuals only for present hours
    df["Residual"] = np.where(df["is_missing"] == 0,
                               df["Consumption_kWh"] - df["Expected_kWh"],
                               np.nan)
    df["Abs_Residual"] = df["Residual"].abs()
    df["Z_Score"] = np.where(df["is_missing"] == 0,
                              df["Residual"] / (df["Baseline_Std"] + 1e-9),
                              np.nan)
    df["Hourly_Anomaly"] = (df["Z_Score"].abs() > Z_THRESHOLD) & (df["is_missing"] == 0)

    # Save hourly (only present hours have meaningful residuals)
    out_hourly = os.path.join(DATA_DIR, "forecast_residuals.csv")
    cols_hourly = ["Date", "Hour", "OSF_ID", "Consumption_kWh", "is_missing",
                   "Expected_kWh", "Residual", "Abs_Residual", "Z_Score", "Hourly_Anomaly"]
    df[cols_hourly].to_csv(out_hourly, index=False)

    # Daily aggregation (mask-aware: only from present hours)
    present = df[df["is_missing"] == 0]
    daily_scores = present.groupby("Date").agg(
        n_present=("Consumption_kWh", "count"),
        DailyTotal_kWh=("Consumption_kWh", "sum"),
        DailyExpected_kWh=("Expected_kWh", "sum"),
        DailyResidual=("Residual", "sum"),
        Mean_Abs_Residual=("Abs_Residual", "mean"),
        Max_Abs_ZScore=("Z_Score", lambda x: x.abs().max()),
        Anomaly_Hours=("Hourly_Anomaly", "sum"),
    ).reset_index()

    # Mask-aware scoring: valid only if enough present hours
    daily_scores["Daily_Anomaly_Score"] = np.where(
        daily_scores["n_present"] >= MIN_PRESENT_HOURS,
        daily_scores["Mean_Abs_Residual"] * (1 + daily_scores["Anomaly_Hours"] * 0.5),
        np.nan  # Not enough data for reliable score
    )

    daily_scores = daily_scores.sort_values("Daily_Anomaly_Score", ascending=False)

    out_daily = os.path.join(DATA_DIR, "daily_anomaly_scores.csv")
    daily_scores.to_csv(out_daily, index=False)

    # Summary
    scored = daily_scores[daily_scores["Daily_Anomaly_Score"].notna()]
    n_days_with_anomaly = (scored["Anomaly_Hours"] > 0).sum()
    n_unscored = daily_scores["Daily_Anomaly_Score"].isna().sum()

    print(f"\n📊 Results (Mask-Aware):")
    print(f"   Scored days: {len(scored)} (with ≥{MIN_PRESENT_HOURS} present hours)")
    print(f"   Unscored days (low data): {n_unscored}")
    print(f"   Days with ≥1 anomaly hour: {int(n_days_with_anomaly)}")

    print(f"\n📋 Top 15 scored days:")
    top = scored.head(15)
    print(top[["Date", "n_present", "DailyTotal_kWh", "Mean_Abs_Residual",
               "Anomaly_Hours", "Daily_Anomaly_Score"]].to_string(index=False))

    print(f"\n💾 Saved: {out_hourly}")
    print(f"💾 Saved: {out_daily}")
    print("=" * 70)


if __name__ == "__main__":
    main()
