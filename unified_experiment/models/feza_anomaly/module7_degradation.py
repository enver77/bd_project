#!/usr/bin/env python3
"""
Module 7 - Degradation Risk & Predictive Fault Detection (Mask-Aware)
======================================================================
Reads from: RESULTS_DIR/daily_features.csv, anomaly_flags.csv, master_dataset.csv
Writes to:  RESULTS_DIR/predictive_risk_scores.csv, pre_failure_profile.csv
"""

import os
import numpy as np
import pandas as pd

RESULTS_DIR = os.environ.get("BEDAS_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))


def rolling_slope(series, window):
    def _slope(vals):
        vals = vals.dropna()
        if len(vals) < window // 2:
            return np.nan
        x = np.arange(len(vals))
        y = vals.values
        n = len(x)
        sx = x.sum(); sy = y.sum()
        sxy = (x * y).sum(); sxx = (x * x).sum()
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            return 0.0
        return (n * sxy - sx * sy) / denom
    return series.rolling(window, min_periods=window//2).apply(_slope, raw=False)


def main():
    print("=" * 70)
    print(" Module 7 - Degradation Risk (Mask-Aware)")
    print("=" * 70)

    daily = pd.read_csv(os.path.join(RESULTS_DIR, "daily_features.csv"), parse_dates=["Date"])
    daily = daily.sort_values("Date").reset_index(drop=True)
    flags = pd.read_csv(os.path.join(RESULTS_DIR, "anomaly_flags.csv"), parse_dates=["Date"])

    active_mask = (~daily["AllDayZero"]) & (daily["data_quality_flag"] != "data_quality_low")
    df = daily[active_mask].copy().reset_index(drop=True)
    print(f"\nActive quality days: {len(df)}")

    night_median_global = df["NightMean"].median()
    df["NightEfficiency"] = df["NightMean"] / (night_median_global + 1e-9)
    print(f"   Global night median: {night_median_global:.3f} kWh")

    for window in [14, 30]:
        df[f"NightSlope_{window}d"] = rolling_slope(df["NightMean"], window)
        df[f"DailySlope_{window}d"] = rolling_slope(df["DailyTotal"], window)

    df["Night_Volatility_14d"] = df["NightMean"].rolling(14, min_periods=7).std()
    df["Night_Volatility_30d"] = df["NightMean"].rolling(30, min_periods=14).std()

    flag_cols = [c for c in flags.columns if c.startswith("Flag_")]
    flags_active = flags[flags["Date"].isin(df["Date"])].sort_values("Date").reset_index(drop=True)
    if len(flag_cols) > 0:
        flags_active["FlagSum"] = flags_active[flag_cols].sum(axis=1)
        df = df.merge(flags_active[["Date", "FlagSum"]], on="Date", how="left")
        df["FlagSum"] = df["FlagSum"].fillna(0)
        df["FlagAccum_14d"] = df["FlagSum"].rolling(14, min_periods=1).sum()
    else:
        df["FlagSum"] = 0
        df["FlagAccum_14d"] = 0

    def compute_degradation(row):
        score = 0.0
        if row["NightEfficiency"] < 0.85: score += 25
        elif row["NightEfficiency"] < 0.92: score += 15
        if not np.isnan(row.get("NightSlope_14d", np.nan)):
            if row["NightSlope_14d"] < -0.02: score += 20
            elif row["NightSlope_14d"] < -0.01: score += 10
        if not np.isnan(row.get("NightSlope_30d", np.nan)):
            if row["NightSlope_30d"] < -0.015: score += 15
        if not np.isnan(row.get("Night_Volatility_14d", np.nan)):
            if row["Night_Volatility_14d"] > 0.4: score += 10
        if row["FlagAccum_14d"] >= 5: score += 15
        elif row["FlagAccum_14d"] >= 3: score += 8
        return min(score, 100)

    df["DegradationScore"] = df.apply(compute_degradation, axis=1)

    def risk_level(score):
        if score >= 60: return "HIGH_RISK"
        if score >= 35: return "MEDIUM_RISK"
        if score >= 15: return "LOW_RISK"
        return "NORMAL"

    df["RiskLevel"] = df["DegradationScore"].apply(risk_level)

    high_risk = df[df["RiskLevel"].isin(["HIGH_RISK", "MEDIUM_RISK"])]
    if len(high_risk) > 5:
        pf_dates = high_risk["Date"].head(30).values
        master = pd.read_csv(os.path.join(RESULTS_DIR, "master_dataset.csv"), parse_dates=["Date"])
        master_pf = master[(master["Date"].isin(pf_dates)) & (master["is_missing"] == 0)]
        pf_profile = master_pf.groupby("Hour")["Consumption_kWh"].agg(
            ["mean", "std", "median"]).reset_index()
        pf_profile.columns = ["Hour", "PreFail_Mean", "PreFail_Std", "PreFail_Median"]
        out_pf = os.path.join(RESULTS_DIR, "pre_failure_profile.csv")
        pf_profile.to_csv(out_pf, index=False)
        print(f"\nPre-failure profile extracted from {len(pf_dates)} days")
    else:
        print("\nNot enough high/medium risk days for pre-failure profile")

    out_cols = ["Date", "OSF_ID", "NightMean", "DailyTotal", "NightEfficiency",
                "NightSlope_14d", "NightSlope_30d", "DailySlope_14d", "DailySlope_30d",
                "Night_Volatility_14d", "Night_Volatility_30d", "FlagAccum_14d",
                "DegradationScore", "RiskLevel", "missing_ratio", "data_quality_flag"]
    out_file = os.path.join(RESULTS_DIR, "predictive_risk_scores.csv")
    df[out_cols].to_csv(out_file, index=False)

    print(f"\nRisk Distribution ({len(df)} quality days):")
    for level in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK", "NORMAL"]:
        n = (df["RiskLevel"] == level).sum()
        if n > 0:
            print(f"   {level:20s}: {n:4d} days")

    print(f"\nSaved: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
