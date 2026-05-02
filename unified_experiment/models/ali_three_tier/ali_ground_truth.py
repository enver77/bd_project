"""
Ali's Ground Truth Parser
Converts rpt-300 meter outages and rpt-301 modem outages -> daily label CSV.

Reads from: DATA_DIR/rpt-300_*.xlsx, DATA_DIR/rpt-301_*.xlsx
Writes to:  RESULTS_DIR/ali_ground_truth_labels.csv
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.environ.get("BEDAS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
RESULTS_DIR = os.environ.get("BEDAS_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))

RPT300 = os.path.join(DATA_DIR, "rpt-300_sayac_kesinti_raporu_(kesinti_bazli)_0XldY.xlsx")
RPT301 = os.path.join(DATA_DIR, "rpt-301_modem_kesinti_raporu_(kesinti_bazli)_f2Umn.xlsx")
OUT_CSV = os.path.join(RESULTS_DIR, "ali_ground_truth_labels.csv")
DAILY_CSV = os.path.join(RESULTS_DIR, "ali_preprocessed_daily.csv")


def parse_rpt300():
    df = pd.read_excel(RPT300)
    # Try Turkish column names; fall back to positional indexing
    if "Başlangıç Tarihi" in df.columns:
        df["start"] = pd.to_datetime(df["Başlangıç Tarihi"], dayfirst=True)
        df["end"] = pd.to_datetime(df["Bitiş Tarihi"], dayfirst=True)
        df["duration_min"] = df["Kesinti Süre (dk)"]
    else:
        df["start"] = pd.to_datetime(df.iloc[:, 9], dayfirst=True)
        df["end"] = pd.to_datetime(df.iloc[:, 10], dayfirst=True)
        df["duration_min"] = df.iloc[:, 11]
    return df[["start", "end", "duration_min"]].copy()


def parse_rpt301():
    df = pd.read_excel(RPT301)
    if "Başlangıç Tarihi" in df.columns:
        df["start"] = pd.to_datetime(df["Başlangıç Tarihi"], dayfirst=True)
        df["end"] = pd.to_datetime(df["Bitiş Tarihi"], dayfirst=True)
        df["duration_sec"] = df["Kesinti Süre (sn)"]
    else:
        df["start"] = pd.to_datetime(df.iloc[:, 8], dayfirst=True)
        df["end"] = pd.to_datetime(df.iloc[:, 9], dayfirst=True)
        df["duration_sec"] = df.iloc[:, 10]
    return df[["start", "end", "duration_sec"]].copy()


def outages_to_daily_labels(outages, date_index):
    labels = pd.Series(0, index=date_index, name="outage")
    for _, row in outages.iterrows():
        if pd.isna(row["start"]) or pd.isna(row["end"]):
            continue
        start_day = row["start"].normalize()
        end_day = row["end"].normalize()
        mask = (date_index >= start_day) & (date_index <= end_day)
        labels[mask] = 1
    return labels


def main():
    if os.path.exists(DAILY_CSV):
        daily = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)
        date_index = daily.index
    else:
        # Fall back to date range from rpt-300
        r300 = parse_rpt300()
        start = r300["start"].min().normalize()
        end = r300["end"].max().normalize() + pd.Timedelta(days=1)
        date_index = pd.date_range(start, end, freq="D")

    print("Parsing rpt-300 (meter outages) ...")
    if os.path.exists(RPT300):
        rpt300 = parse_rpt300()
        print(f"  Found {len(rpt300)} meter outage events")
    else:
        print(f"  WARNING: {RPT300} not found")
        rpt300 = pd.DataFrame(columns=["start", "end", "duration_min"])

    print("Parsing rpt-301 (modem outages) ...")
    if os.path.exists(RPT301):
        rpt301 = parse_rpt301()
        print(f"  Found {len(rpt301)} modem outage events")
    else:
        print(f"  WARNING: {RPT301} not found")
        rpt301 = pd.DataFrame(columns=["start", "end", "duration_sec"])

    meter_labels = outages_to_daily_labels(rpt300, date_index)
    modem_labels = outages_to_daily_labels(rpt301, date_index)

    if len(rpt300) > 0:
        rpt300["date"] = rpt300["start"].dt.normalize()
        duration_map = rpt300.groupby("date")["duration_min"].sum()
    else:
        duration_map = pd.Series(dtype=float)

    gt = pd.DataFrame({
        "outage": meter_labels,
        "modem_outage": modem_labels,
    })
    gt["outage_duration_min"] = gt.index.map(duration_map).fillna(0).astype(int)

    gt["outage_confidence"] = "none"
    gt.loc[gt["outage"] == 1, "outage_confidence"] = "possible"
    gt.loc[(gt["outage"] == 1) & (gt["outage_duration_min"] >= 60), "outage_confidence"] = "definite"

    gt.to_csv(OUT_CSV)
    print(f"\nSaved -> {OUT_CSV}")

    n_outage_days = int(gt["outage"].sum())
    n_definite = int((gt["outage_confidence"] == "definite").sum())
    print(f"Total outage days: {n_outage_days}  (definite: {n_definite})")
    print(f"Ground truth outage dates:")
    for d in gt[gt["outage"] == 1].index:
        dur = gt.loc[d, "outage_duration_min"]
        print(f"  {d.date()}  ({dur} min)")

    return gt


if __name__ == "__main__":
    main()
