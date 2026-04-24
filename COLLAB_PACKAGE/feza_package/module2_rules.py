#!/usr/bin/env python3
"""
Module 2 – Rule-Based Anomaly Flags (Mask-Aware)
==================================================
Fast, interpretable anomaly flags. All checks skip missing hours.

Rules:
  1. AllDayZero      – all PRESENT hours sum = 0 (not because of missing data)
  2. DaytimeNonZero  – present daytime hours with consumption > threshold
  3. NightDrop       – NightMean (present hours) significantly below rolling avg
  4. ScheduleShift   – ramp-up/ramp-down shifted (only checks present hours)

Outputs:
  - anomaly_flags.csv : per-day flags (mask-aware)
  - anomaly_top20.csv : top 20 inspection list
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Thresholds
DAYTIME_THRESHOLD_KWH  = 0.2
DAYTIME_HOUR_LIMIT     = 2
NIGHT_DROP_Z           = -2.0
ROLLING_WINDOW         = 30
RAMPUP_NORMAL_HOUR     = 17
RAMPDOWN_NORMAL_HOUR   = 7
SCHEDULE_SHIFT_HOURS   = 2


def load_data():
    master = pd.read_csv(os.path.join(DATA_DIR, "master_dataset.csv"), parse_dates=["Date"])
    daily  = pd.read_csv(os.path.join(DATA_DIR, "daily_features.csv"), parse_dates=["Date"])
    return master, daily


def flag_all_day_zero(daily: pd.DataFrame) -> pd.Series:
    """Flag only if all PRESENT hours are zero (not due to all-missing)."""
    return (daily["DailyTotal"] == 0.0) & (daily["n_present"] > 0)


def flag_daytime_nonzero(master: pd.DataFrame) -> pd.DataFrame:
    """Flag days where multiple present daytime hours have meaningful consumption."""
    daytime = master[(master["Hour"].between(9, 16)) & (master["is_missing"] == 0)].copy()
    counts = daytime[daytime["Consumption_kWh"] > DAYTIME_THRESHOLD_KWH] \
        .groupby("Date").size().reset_index(name="DaytimeHighCount")
    counts["DaytimeNonZero"] = counts["DaytimeHighCount"] >= DAYTIME_HOUR_LIMIT
    return counts[["Date", "DaytimeHighCount", "DaytimeNonZero"]]


def flag_night_drop(daily: pd.DataFrame) -> pd.DataFrame:
    """Flag days where NightMean (from present night hours) is below rolling avg.
    Skip days with insufficient present night hours (NightMean is NaN)."""
    df = daily[["Date", "NightMean", "data_quality_flag"]].copy()
    df = df.sort_values("Date")

    # Use only quality days for rolling stats
    df["NightMean_Roll"] = df["NightMean"].rolling(ROLLING_WINDOW, min_periods=7, center=True).mean()
    df["NightMean_RollStd"] = df["NightMean"].rolling(ROLLING_WINDOW, min_periods=7, center=True).std()
    df["NightDrop_Z"] = (df["NightMean"] - df["NightMean_Roll"]) / (df["NightMean_RollStd"] + 1e-9)
    df["NightDrop"] = (df["NightDrop_Z"] < NIGHT_DROP_Z) & df["NightMean"].notna()
    return df[["Date", "NightDrop_Z", "NightDrop"]]


def flag_schedule_shift(master: pd.DataFrame) -> pd.DataFrame:
    """Detect shifted ramp-up/ramp-down using only present hours."""
    results = []
    for dt, grp in master.groupby("Date"):
        present = grp[grp["is_missing"] == 0].sort_values("Hour").set_index("Hour")["Consumption_kWh"]
        if present.empty or present.sum() == 0:
            results.append({"Date": dt, "EveningOnHour": np.nan,
                            "MorningOffHour": np.nan, "ScheduleShift": False})
            continue

        # Evening ramp-up: first PRESENT hour >= 15 where consumption jumps
        evening_on = np.nan
        for h in range(15, 22):
            if h in present.index and (h + 1) in present.index:
                if present[h] < 1.0 and present[h + 1] > 1.0:
                    evening_on = h + 1
                    break
        if np.isnan(evening_on) and 17 in present.index and present[17] > 1.0:
            evening_on = 17

        # Morning ramp-down
        morning_off = np.nan
        for h in range(5, 10):
            if h in present.index and (h + 1) in present.index:
                if present[h] > 1.0 and present[h + 1] < 1.0:
                    morning_off = h + 1
                    break
        if np.isnan(morning_off) and 7 in present.index and present[7] < 0.5:
            morning_off = 7

        shift = False
        if not np.isnan(evening_on) and abs(evening_on - RAMPUP_NORMAL_HOUR) >= SCHEDULE_SHIFT_HOURS:
            shift = True
        if not np.isnan(morning_off) and abs(morning_off - RAMPDOWN_NORMAL_HOUR) >= SCHEDULE_SHIFT_HOURS:
            shift = True

        results.append({
            "Date": dt,
            "EveningOnHour": evening_on,
            "MorningOffHour": morning_off,
            "ScheduleShift": shift,
        })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print(" Module 2 – Rule-Based Anomaly Flags (Mask-Aware)")
    print("=" * 70)

    master, daily = load_data()

    # 1. AllDayZero (mask-aware)
    daily["Flag_AllDayZero"] = flag_all_day_zero(daily)

    # 2. DaytimeNonZero (mask-aware)
    dt_nz = flag_daytime_nonzero(master)
    daily = daily.merge(dt_nz[["Date", "DaytimeHighCount", "DaytimeNonZero"]], on="Date", how="left")
    daily["DaytimeHighCount"] = daily["DaytimeHighCount"].fillna(0).astype(int)
    daily["Flag_DaytimeNonZero"] = daily["DaytimeNonZero"].fillna(False)
    daily.drop(columns=["DaytimeNonZero"], inplace=True)

    # 3. NightDrop (mask-aware)
    nd = flag_night_drop(daily)
    daily = daily.merge(nd[["Date", "NightDrop_Z", "NightDrop"]], on="Date", how="left")
    daily["Flag_NightDrop"] = daily["NightDrop"].fillna(False)
    daily.drop(columns=["NightDrop"], inplace=True)

    # 4. ScheduleShift (mask-aware)
    ss = flag_schedule_shift(master)
    daily = daily.merge(ss[["Date", "EveningOnHour", "MorningOffHour", "ScheduleShift"]], on="Date", how="left")
    daily["Flag_ScheduleShift"] = daily["ScheduleShift"].fillna(False)
    daily.drop(columns=["ScheduleShift"], inplace=True)

    # Count flags per day
    flag_cols = ["Flag_AllDayZero", "Flag_DaytimeNonZero", "Flag_NightDrop", "Flag_ScheduleShift"]
    daily["FlagCount"] = daily[flag_cols].sum(axis=1)

    daily["AnomalyTypes"] = daily.apply(
        lambda r: ", ".join([c.replace("Flag_", "") for c in flag_cols if r[c]]),
        axis=1
    )

    # Save
    out_flags = os.path.join(DATA_DIR, "anomaly_flags.csv")
    daily.to_csv(out_flags, index=False)

    # Top 20 (exclude data_quality_low days from inspection list)
    flagged = daily[(daily["FlagCount"] > 0) & (daily["data_quality_flag"] != "data_quality_low")]
    flagged = flagged.sort_values(["FlagCount", "Date"], ascending=[False, True])
    top20 = flagged.head(20)
    out_top20 = os.path.join(DATA_DIR, "anomaly_top20.csv")
    top_cols = ["Date", "OSF_ID", "DailyTotal", "NightMean", "DayMean",
                "FlagCount", "AnomalyTypes", "NightDrop_Z", "EveningOnHour",
                "MorningOffHour", "missing_ratio", "data_quality_flag"]
    top20[top_cols].to_csv(out_top20, index=False)

    # Summary
    print(f"\n📊 Anomaly Flag Summary ({len(daily)} days):")
    for col in flag_cols:
        n = daily[col].sum()
        print(f"   {col:30s}: {int(n):4d} days ({n/len(daily)*100:.1f}%)")
    print(f"   {'Any flag':30s}: {int((daily['FlagCount']>0).sum()):4d} days")

    q_ok = daily[daily["data_quality_flag"] != "data_quality_low"]
    print(f"\n   Flags on quality days only: {int((q_ok['FlagCount']>0).sum())} days")

    print(f"\n💾 Saved: {out_flags}")
    print(f"💾 Saved: {out_top20}")
    print("=" * 70)


if __name__ == "__main__":
    main()
