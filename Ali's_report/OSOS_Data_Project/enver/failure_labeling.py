"""
Failure Labeling for Street Lighting Data
==========================================
Reads cekis_data.csv (hourly energy consumption) and the modem failure report,
then adds binary target labels:
  - failure_now:    1 if a failure overlaps with that hour
  - failure_next_24h: 1 if a failure starts within the next 24 hours (for prediction)
  - failure_next_48h: 1 if a failure starts within the next 48 hours

Output: cekis_data_labeled.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent

# ── Load data ────────────────────────────────────────────────────────────────
cekis = pd.read_csv(DATA_DIR / "cekis_data.csv", parse_dates=["timestamp"])
failures = pd.read_excel(
    DATA_DIR / "rpt-308_modem_kesinti_raporu_(butun_kesintiler)_bPZj6.xlsx"
)

# Parse failure timestamps (DD/MM/YYYY HH:MM:SS)
failures["start"] = pd.to_datetime(failures.iloc[:, 8], dayfirst=True)
failures["end"] = pd.to_datetime(failures.iloc[:, 9], dayfirst=True)
failures["duration_sec"] = failures.iloc[:, 10]

print(f"Cekis data : {len(cekis)} rows, {cekis['timestamp'].min()} to {cekis['timestamp'].max()}")
print(f"Failures   : {len(failures)} events")

# ── Label 1: failure_now (hour overlaps with a failure window) ───────────────
# For each cekis row, the "hour window" is [timestamp, timestamp + 1h)
cekis["hour_end"] = cekis["timestamp"] + pd.Timedelta(hours=1)

failure_now = np.zeros(len(cekis), dtype=int)
for _, f in failures.iterrows():
    # overlap: failure starts before hour ends AND failure ends after hour starts
    mask = (f["start"] < cekis["hour_end"]) & (f["end"] > cekis["timestamp"])
    failure_now[mask] = 1

cekis["failure_now"] = failure_now

# ── Label 2 & 3: failure in next 24h / 48h (for predictive models) ──────────
failure_starts = failures["start"].values  # numpy datetime64 array

failure_next_24h = np.zeros(len(cekis), dtype=int)
failure_next_48h = np.zeros(len(cekis), dtype=int)

ts = cekis["timestamp"].values
for fs in failure_starts:
    diff_hours = (fs - ts) / np.timedelta64(1, "h")
    failure_next_24h[(diff_hours > 0) & (diff_hours <= 24)] = 1
    failure_next_48h[(diff_hours > 0) & (diff_hours <= 48)] = 1

cekis["failure_next_24h"] = failure_next_24h
cekis["failure_next_48h"] = failure_next_48h

# ── Also add failure metadata columns ────────────────────────────────────────
# For rows where failure_now==1, add the duration of the overlapping failure
cekis["failure_duration_sec"] = 0
for _, f in failures.iterrows():
    mask = (f["start"] < cekis["hour_end"]) & (f["end"] > cekis["timestamp"])
    cekis.loc[mask, "failure_duration_sec"] = f["duration_sec"]

# ── Clean up and save ────────────────────────────────────────────────────────
cekis.drop(columns=["hour_end"], inplace=True)

out_path = DATA_DIR / "cekis_data_labeled.csv"
cekis.to_csv(out_path, index=False)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Output: {out_path}")
print(f"Total rows        : {len(cekis)}")
print(f"failure_now == 1  : {cekis['failure_now'].sum()} ({100*cekis['failure_now'].mean():.2f}%)")
print(f"failure_next_24h  : {cekis['failure_next_24h'].sum()} ({100*cekis['failure_next_24h'].mean():.2f}%)")
print(f"failure_next_48h  : {cekis['failure_next_48h'].sum()} ({100*cekis['failure_next_48h'].mean():.2f}%)")
print(f"\nColumns: {cekis.columns.tolist()}")
print(f"\nLabel distribution (failure_now):")
print(cekis["failure_now"].value_counts())
