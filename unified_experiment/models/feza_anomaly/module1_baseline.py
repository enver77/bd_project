#!/usr/bin/env python3
"""
Module 1 - Baseline Features (Mask-Aware)
==========================================
Computes daily features using only present (non-missing) hours.

Reads from: RESULTS_DIR/master_dataset.csv, RESULTS_DIR/missing_data_summary.csv
Writes to:  RESULTS_DIR/daily_features.csv, RESULTS_DIR/normal_profile.csv
"""

import os
import sys
import numpy as np
import pandas as pd

RESULTS_DIR = os.environ.get("BEDAS_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))

NIGHT_HOURS    = [22, 23, 0, 1, 2, 3, 4, 5]
DAY_HOURS      = [9, 10, 11, 12, 13, 14, 15, 16]
EVENING_HOURS  = [17, 18, 19, 20, 21]
MORNING_HOURS  = [6, 7, 8]


def load_data():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "master_dataset.csv"), parse_dates=["Date"])
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6])
    ms = pd.read_csv(os.path.join(RESULTS_DIR, "missing_data_summary.csv"), parse_dates=["Date"])
    return df, ms


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per day with mask-aware engineered features."""
    rows = []
    for dt, grp in df.groupby("Date"):
        grp_sorted = grp.sort_values("Hour")
        vals = grp_sorted.set_index("Hour")["Consumption_kWh"]
        mask = grp_sorted.set_index("Hour")["is_missing"]
        osf_id = grp["OSF_ID"].iloc[0]

        n_present = int((mask == 0).sum())
        n_missing = int((mask == 1).sum())
        missing_ratio = round(n_missing / 24, 4)

        present_vals = vals[mask == 0]
        daily_total = float(present_vals.sum()) if len(present_vals) > 0 else 0.0

        def segment_mean(hours):
            seg = [(vals.get(h, np.nan), mask.get(h, 1)) for h in hours]
            present = [v for v, m in seg if m == 0 and not np.isnan(v)]
            return float(np.mean(present)) if present else np.nan

        night_mean = segment_mean(NIGHT_HOURS)
        day_mean = segment_mean(DAY_HOURS)
        evening_mean = segment_mean(EVENING_HOURS)
        morning_mean = segment_mean(MORNING_HOURS)

        eps = 1e-6
        day_night_ratio = night_mean / (day_mean + eps) if not np.isnan(night_mean) and not np.isnan(day_mean) else np.nan

        def safe_get(h):
            if mask.get(h, 1) == 0:
                return vals.get(h, np.nan)
            return np.nan

        v16, v17 = safe_get(16), safe_get(17)
        ramp_up = v17 - v16 if not np.isnan(v16) and not np.isnan(v17) else np.nan

        v06, v07 = safe_get(6), safe_get(7)
        ramp_down = v06 - v07 if not np.isnan(v06) and not np.isnan(v07) else np.nan

        transition_sharpness = (ramp_up if not np.isnan(ramp_up) else 0) + \
                                (ramp_down if not np.isnan(ramp_down) else 0)

        if len(present_vals) > 0:
            peak_hour = present_vals.idxmax()
            peak_val = float(present_vals.max())
            min_hour = present_vals.idxmin()
            min_val = float(present_vals.min())
        else:
            peak_hour = min_hour = np.nan
            peak_val = min_val = 0.0

        hourly_std = float(present_vals.std()) if len(present_vals) > 1 else np.nan

        night_present = [vals.get(h, np.nan) for h in NIGHT_HOURS if mask.get(h, 1) == 0]
        night_std = float(np.std(night_present)) if len(night_present) > 1 else np.nan

        day_nz = sum(1 for h in DAY_HOURS if mask.get(h, 1) == 0 and vals.get(h, 0) > 0.05)
        all_day_zero = (daily_total == 0.0 and n_present > 0)

        dow = pd.Timestamp(dt).dayofweek
        month = pd.Timestamp(dt).month
        year = pd.Timestamp(dt).year

        rows.append({
            "Date": dt, "OSF_ID": osf_id, "Year": year, "Month": month,
            "DayOfWeek": dow, "IsWeekend": dow in [5, 6],
            "DailyTotal": round(daily_total, 4),
            "DayMean": round(day_mean, 4) if not np.isnan(day_mean) else np.nan,
            "NightMean": round(night_mean, 4) if not np.isnan(night_mean) else np.nan,
            "EveningMean": round(evening_mean, 4) if not np.isnan(evening_mean) else np.nan,
            "MorningMean": round(morning_mean, 4) if not np.isnan(morning_mean) else np.nan,
            "DayNightRatio": round(day_night_ratio, 4) if not np.isnan(day_night_ratio) else np.nan,
            "TransitionSharpness": round(transition_sharpness, 4),
            "RampUp_16to17": round(ramp_up, 4) if not np.isnan(ramp_up) else np.nan,
            "RampDown_06to07": round(ramp_down, 4) if not np.isnan(ramp_down) else np.nan,
            "PeakHour": peak_hour, "PeakValue": round(peak_val, 4),
            "MinHour": min_hour, "MinValue": round(min_val, 4),
            "HourlyStd": round(hourly_std, 4) if not np.isnan(hourly_std) else np.nan,
            "NightStd": round(night_std, 4) if not np.isnan(night_std) else np.nan,
            "DayNonZeroCount": int(day_nz), "AllDayZero": all_day_zero,
            "n_present": n_present, "missing_ratio": missing_ratio,
        })

    return pd.DataFrame(rows)


def build_normal_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average 24-hour curve from complete days with non-zero consumption."""
    complete = df.groupby("Date").filter(lambda g: g["is_missing"].sum() == 0)
    day_totals = complete.groupby("Date")["Consumption_kWh"].sum()
    non_zero_dates = day_totals[day_totals > 0].index
    normal = complete[complete["Date"].isin(non_zero_dates)]
    profile = normal.groupby("Hour")["Consumption_kWh"].agg(["mean", "std", "median"]).reset_index()
    profile.columns = ["Hour", "Mean_kWh", "Std_kWh", "Median_kWh"]
    return profile


def main():
    print("=" * 70)
    print(" Module 1 - Baseline Features (Mask-Aware)")
    print("=" * 70)

    df, ms = load_data()
    print(f"\nLoaded master dataset: {len(df)} rows, {df['Date'].nunique()} days")

    daily = build_daily_features(df)
    daily = daily.merge(ms[["Date", "data_quality_flag"]], on="Date", how="left")
    daily["data_quality_flag"] = daily["data_quality_flag"].fillna("complete")

    out_daily = os.path.join(RESULTS_DIR, "daily_features.csv")
    daily.to_csv(out_daily, index=False)
    print(f"\nDaily features: {len(daily)} rows, {len(daily.columns)} columns")

    profile = build_normal_profile(df)
    out_profile = os.path.join(RESULTS_DIR, "normal_profile.csv")
    profile.to_csv(out_profile, index=False)
    print(f"Normal 24h profile saved")

    active = daily[(~daily["AllDayZero"]) & (daily["data_quality_flag"] != "data_quality_low")]
    print(f"\nFeature Statistics (active + quality days, n={len(active)}):")
    stats_cols = ["DailyTotal", "NightMean", "DayMean", "EveningMean",
                  "DayNightRatio", "TransitionSharpness", "HourlyStd"]
    print(active[stats_cols].describe().round(3).to_string())

    print(f"\nSaved: {out_daily}")
    print(f"Saved: {out_profile}")
    print("=" * 70)


if __name__ == "__main__":
    main()
