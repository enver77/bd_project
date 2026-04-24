#!/usr/bin/env python3
"""
Module 3B – Filtered Change Point Detection (Mask-Aware)
=========================================================
CUSUM change point detection on mask-aware features.
Excludes data_quality_low days from the analysis.

Outputs:
  - changepoints_filtered.csv
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

MIN_CHANGE_PCT = 15.0
MIN_PERSISTENCE = 7


def cusum_changepoints(series: np.ndarray, threshold: float = 4.0, drift: float = 0.5,
                       min_segment: int = 14) -> list:
    n = len(series)
    if n < 2 * min_segment:
        return []
    mean_val = np.mean(series)
    std_val = np.std(series)
    if std_val < 1e-9:
        return []
    normalized = (series - mean_val) / std_val
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    changepoints = []
    for i in range(1, n):
        s_pos[i] = max(0, s_pos[i-1] + normalized[i] - drift)
        s_neg[i] = max(0, s_neg[i-1] - normalized[i] - drift)
        if s_pos[i] > threshold or s_neg[i] > threshold:
            if not changepoints or (i - changepoints[-1]) >= min_segment:
                changepoints.append(i)
                s_pos[i] = 0
                s_neg[i] = 0
    return changepoints


def segment_analysis(dates, values, cp_indices):
    results = []
    all_breaks = [0] + cp_indices + [len(values)]
    for i in range(len(cp_indices)):
        cp_idx = cp_indices[i]
        before_start = all_breaks[i]
        after_end = all_breaks[i + 2] if i + 2 < len(all_breaks) else len(values)
        before = values[before_start:cp_idx]
        after = values[cp_idx:after_end]
        before_mean = np.mean(before) if len(before) > 0 else np.nan
        after_mean = np.mean(after) if len(after) > 0 else np.nan
        change_pct = ((after_mean - before_mean) / (before_mean + 1e-9)) * 100
        duration = after_end - cp_idx

        if len(after) >= MIN_PERSISTENCE:
            first_week = after[:MIN_PERSISTENCE]
            first_week_mean = np.mean(first_week)
            persistent = abs(first_week_mean - before_mean) > abs(before_mean * 0.05)
        else:
            persistent = False

        results.append({
            "ChangePoint_Date": dates[cp_idx],
            "ChangePoint_Index": cp_idx,
            "Before_Mean": round(before_mean, 3),
            "After_Mean": round(after_mean, 3),
            "Change_kWh": round(after_mean - before_mean, 3),
            "Change_Pct": round(change_pct, 1),
            "Before_Days": len(before),
            "After_Days": len(after),
            "Duration_Days": duration,
            "Persistent": persistent,
        })
    return results


def main():
    print("=" * 70)
    print(" Module 3B – Filtered Change Point Detection (Mask-Aware)")
    print("=" * 70)

    daily = pd.read_csv(os.path.join(DATA_DIR, "daily_features.csv"), parse_dates=["Date"])
    daily = daily.sort_values("Date").reset_index(drop=True)

    # Exclude data_quality_low AND AllDayZero days from CUSUM
    active = daily[(~daily["AllDayZero"]) &
                   (daily["data_quality_flag"] != "data_quality_low")].reset_index(drop=True)
    dates = active["Date"].values
    print(f"\n📂 Active quality days for CUSUM: {len(active)}")

    all_results = []
    for series_name in ["DailyTotal", "NightMean", "EveningMean"]:
        vals = active[series_name].dropna().values
        if len(vals) < 28:
            continue
        cps = cusum_changepoints(vals, threshold=4.0, drift=0.5, min_segment=14)
        if cps:
            results = segment_analysis(dates, vals, cps)
            for r in results:
                r["Series"] = series_name
                all_results.append(r)

    raw_df = pd.DataFrame(all_results)
    print(f"   Raw change points: {len(raw_df)}")

    # Filter
    if len(raw_df) > 0:
        filtered = raw_df[
            (raw_df["Change_Pct"].abs() >= MIN_CHANGE_PCT) &
            (raw_df["Duration_Days"] >= MIN_PERSISTENCE)
        ].copy()
        filtered = filtered.sort_values("ChangePoint_Date").reset_index(drop=True)
    else:
        filtered = pd.DataFrame()

    print(f"   Filtered: {len(filtered)}")

    out = os.path.join(DATA_DIR, "changepoints_filtered.csv")
    filtered.to_csv(out, index=False)

    if len(filtered) > 0:
        print(f"\n📋 Filtered Change Points:")
        for _, r in filtered.iterrows():
            p = "✅ Persistent" if r["Persistent"] else "⚡ Transient"
            print(f"   📍 {pd.Timestamp(r['ChangePoint_Date']).date()} [{r['Series']:15s}] | "
                  f"{r['Before_Mean']:.2f} → {r['After_Mean']:.2f} kWh ({r['Change_Pct']:+.1f}%) "
                  f"| {r['Duration_Days']}d | {p}")

    print(f"\n💾 Saved: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
