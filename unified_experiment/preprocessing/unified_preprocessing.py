#!/usr/bin/env python3
"""
Unified Preprocessing Pipeline
================================
Single source of truth for ALL team member approaches.

Pipeline:
  1. Feza's data_pipeline.py  ->  master_dataset.csv  (mask-aware, with is_missing)
  2. Imputation step          ->  master_imputed.csv   (interpolated, cutoff-safe)
  3. Hourly flat export       ->  hourly_data.csv      (timestamp-indexed, ML-ready)
  4. Failure labeling         ->  hourly_data_labeled.csv (with failure_next_24h target)

Every downstream model reads from hourly_data_labeled.csv (or master_imputed.csv
for Feza's daily-aggregation pipeline). No model does its own Excel parsing.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add preprocessing dir to path so we can import data_pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_pipeline


def impute_master(master: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing hours using time-based interpolation.

    Strategy (Feza-style, controlled):
      1. Only impute within the valid date range (up to CUTOFF)
      2. Reindex to full hourly coverage within that range
      3. Use time-based interpolation (same method Sena used, but cutoff-safe)
      4. Keep the is_missing column so downstream can still use the mask

    Returns: imputed DataFrame with same columns + 'was_imputed' flag
    """
    master = master.copy()
    master["Date"] = pd.to_datetime(master["Date"])

    # Build a proper timestamp column
    master["timestamp"] = master["Date"] + pd.to_timedelta(master["Hour"], unit="h")
    master = master.sort_values("timestamp").reset_index(drop=True)

    # Reindex to full hourly range within cutoff
    full_index = pd.date_range(
        start=master["timestamp"].min(),
        end=master["timestamp"].max(),
        freq="h"
    )

    # Set timestamp as index for reindex
    master = master.set_index("timestamp")

    # Before reindex, remember which rows existed
    existing_timestamps = set(master.index)

    master = master.reindex(full_index)
    master.index.name = "timestamp"

    # Mark newly created rows (gaps in original data)
    master["was_imputed"] = 0
    new_rows = ~master.index.isin(existing_timestamps)
    master.loc[new_rows, "was_imputed"] = 1
    # Also mark originally-missing rows as imputed
    master.loc[master["is_missing"] == 1, "was_imputed"] = 1

    # Fill metadata columns for new rows
    master["Hour"] = master.index.hour
    master["Date"] = master.index.normalize()
    master["Month"] = master.index.month
    master["Year"] = master.index.year

    # Forward-fill OSF_ID and missingness_cause for new rows
    master["OSF_ID"] = master["OSF_ID"].ffill().bfill()
    master["missingness_cause"] = master["missingness_cause"].fillna("")
    master["is_missing"] = master["is_missing"].fillna(1).astype(int)

    # Interpolate consumption using time-based method
    master["Consumption_kWh"] = master["Consumption_kWh"].interpolate(method="time")
    # Fill any remaining edge NaNs with nearest
    master["Consumption_kWh"] = master["Consumption_kWh"].fillna(method="bfill").fillna(method="ffill").fillna(0)

    master = master.reset_index()
    print(f"   Imputed {int(master['was_imputed'].sum())} hours "
          f"({int(new_rows.sum())} new rows + "
          f"{int(master['was_imputed'].sum()) - int(new_rows.sum())} originally missing)")

    return master


def create_hourly_flat(master_imputed: pd.DataFrame) -> pd.DataFrame:
    """
    Convert master_imputed to a flat hourly format with engineered time features.
    This is the format Enver's and Sena's models expect.

    Columns: timestamp, energy_kwh, hour, day_of_week, is_weekend, is_night,
             month, year, day, is_missing, was_imputed, OSF_ID
    """
    df = master_imputed.copy()

    hourly = pd.DataFrame({
        "timestamp": df["timestamp"],
        "energy_kwh": df["Consumption_kWh"],
        "hour": df["Hour"].astype(int),
        "day": df["timestamp"].dt.day,
        "month": df["timestamp"].dt.month,
        "year": df["timestamp"].dt.year,
        "day_of_week": df["timestamp"].dt.dayofweek,
        "is_weekend": df["timestamp"].dt.dayofweek.isin([5, 6]).astype(int),
        "is_night": df["Hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int),
        "is_missing": df["is_missing"],
        "was_imputed": df["was_imputed"],
        "OSF_ID": df["OSF_ID"],
    })

    hourly = hourly.sort_values("timestamp").reset_index(drop=True)
    return hourly


def add_failure_labels(hourly: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """
    Add failure labels from BEDAS outage reports.
    Same logic as enver_reports/failure_labeling.py.

    Labels:
      - failure_now:      1 if a failure overlaps with that hour
      - failure_next_24h: 1 if a failure starts within the next 24 hours
      - failure_next_48h: 1 if a failure starts within the next 48 hours
    """
    # Try to load the failure report
    failure_file = os.path.join(data_dir,
        "rpt-308_modem_kesinti_raporu_(butun_kesintiler)_bPZj6.xlsx")

    if not os.path.exists(failure_file):
        print(f"   WARNING: Failure report not found at {failure_file}")
        print("   Skipping failure labeling.")
        hourly["failure_now"] = 0
        hourly["failure_next_24h"] = 0
        hourly["failure_next_48h"] = 0
        return hourly

    import openpyxl
    failures = pd.read_excel(failure_file)
    failures["start"] = pd.to_datetime(failures.iloc[:, 8], dayfirst=True)
    failures["end"] = pd.to_datetime(failures.iloc[:, 9], dayfirst=True)
    failures["duration_sec"] = failures.iloc[:, 10]

    print(f"   Loaded {len(failures)} failure events")

    ts = hourly["timestamp"].values
    hour_end = ts + np.timedelta64(1, "h")
    failure_starts = failures["start"].values

    # failure_now: hour overlaps with failure window
    failure_now = np.zeros(len(hourly), dtype=int)
    for _, f in failures.iterrows():
        mask = (f["start"] < hour_end) & (f["end"] > ts)
        failure_now[mask] = 1

    # failure_next_24h / 48h
    failure_next_24h = np.zeros(len(hourly), dtype=int)
    failure_next_48h = np.zeros(len(hourly), dtype=int)
    for fs in failure_starts:
        diff_hours = (fs - ts) / np.timedelta64(1, "h")
        failure_next_24h[(diff_hours > 0) & (diff_hours <= 24)] = 1
        failure_next_48h[(diff_hours > 0) & (diff_hours <= 48)] = 1

    hourly["failure_now"] = failure_now
    hourly["failure_next_24h"] = failure_next_24h
    hourly["failure_next_48h"] = failure_next_48h

    print(f"   failure_now == 1:      {failure_now.sum()} hours ({100*failure_now.mean():.2f}%)")
    print(f"   failure_next_24h == 1: {failure_next_24h.sum()} hours ({100*failure_next_24h.mean():.2f}%)")

    return hourly


def run(data_dir: str, output_dir: str):
    """
    Run the full unified preprocessing pipeline.

    Parameters
    ----------
    data_dir   : path to folder containing raw osf_*.xlsx and rpt-*.xlsx files
    output_dir : path to write all output CSVs
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(" UNIFIED PREPROCESSING PIPELINE")
    print("=" * 70)

    # ── Step 1: Feza's data pipeline ──
    print("\n[Step 1/4] Running Feza's data pipeline...")
    master, outage_df, missing_summary = data_pipeline.run_pipeline(
        data_dir=data_dir,
        output_dir=output_dir
    )

    # ── Step 2: Imputation ──
    print("\n[Step 2/4] Imputing missing values...")
    master_imputed = impute_master(master)
    master_imputed.to_csv(os.path.join(output_dir, "master_imputed.csv"), index=False)
    print(f"   Saved: master_imputed.csv ({len(master_imputed)} rows)")

    # ── Step 3: Flat hourly export ──
    print("\n[Step 3/4] Creating hourly flat dataset...")
    hourly = create_hourly_flat(master_imputed)
    hourly.to_csv(os.path.join(output_dir, "hourly_data.csv"), index=False)
    print(f"   Saved: hourly_data.csv ({len(hourly)} rows)")

    # ── Step 4: Failure labeling ──
    print("\n[Step 4/4] Adding failure labels...")
    hourly_labeled = add_failure_labels(hourly, data_dir)
    hourly_labeled.to_csv(os.path.join(output_dir, "hourly_data_labeled.csv"), index=False)
    print(f"   Saved: hourly_data_labeled.csv")

    # ── Summary ──
    print("\n" + "=" * 70)
    print(" PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"   Date range : {hourly['timestamp'].min()} -> {hourly['timestamp'].max()}")
    print(f"   Total rows : {len(hourly)}")
    print(f"   Missing    : {int(hourly['is_missing'].sum())} hours (kept as flag)")
    print(f"   Imputed    : {int(hourly['was_imputed'].sum())} hours")
    print(f"\n   Output dir : {output_dir}")
    print(f"   Files:")
    print(f"     - master_dataset.csv        (raw with mask)")
    print(f"     - master_imputed.csv         (imputed with mask)")
    print(f"     - hourly_data.csv            (flat hourly)")
    print(f"     - hourly_data_labeled.csv    (flat hourly + failure labels)")
    print(f"     - missing_data_summary.csv   (per-day quality)")
    print(f"     - outage_events.csv          (BEDAS outages)")
    print(f"     - data_quality_report.csv    (summary metrics)")
    print("=" * 70)

    return hourly_labeled


if __name__ == "__main__":
    # Default: look for data in ../data/, output to ../results/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    data_dir = os.path.join(project_dir, "data")
    output_dir = os.path.join(project_dir, "results")

    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please create unified_experiment/data/ and copy osf_*.xlsx + rpt-*.xlsx there.")
        sys.exit(1)

    run(data_dir, output_dir)
