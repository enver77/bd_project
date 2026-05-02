"""
Ground Truth Parser
Converts rpt-300 meter outages and rpt-301 modem outages → daily label CSV.

Output: data/processed/ground_truth_labels.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RPT300 = PROJECT_ROOT / "Bakım_Kayıtları" / "rpt-300_sayac_kesinti_raporu_(kesinti_bazli)_0XldY.xlsx"
RPT301 = PROJECT_ROOT / "Bakım_Kayıtları" / "rpt-301_modem_kesinti_raporu_(kesinti_bazli)_f2Umn.xlsx"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "ground_truth_labels.csv"
PREPROCESSED = PROJECT_ROOT / "data" / "processed" / "preprocessed_daily.csv"


def parse_rpt300() -> pd.DataFrame:
    df = pd.read_excel(RPT300)
    df["start"] = pd.to_datetime(df["Başlangıç Tarihi"], dayfirst=True)
    df["end"]   = pd.to_datetime(df["Bitiş Tarihi"],     dayfirst=True)
    df["duration_min"] = df["Kesinti Süre (dk)"]
    return df[["start", "end", "duration_min"]].copy()


def parse_rpt301() -> pd.DataFrame:
    df = pd.read_excel(RPT301)
    df["start"] = pd.to_datetime(df["Başlangıç Tarihi"], dayfirst=True)
    df["end"]   = pd.to_datetime(df["Bitiş Tarihi"],     dayfirst=True)
    df["duration_sec"] = df["Kesinti Süre (sn)"]
    return df[["start", "end", "duration_sec"]].copy()


def outages_to_daily_labels(outages: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.Series:
    """Flag any calendar day touched by an outage interval."""
    labels = pd.Series(0, index=date_index, name="outage")
    for _, row in outages.iterrows():
        start_day = row["start"].normalize()
        end_day   = row["end"].normalize()
        mask = (date_index >= start_day) & (date_index <= end_day)
        labels[mask] = 1
    return labels


def main():
    # Load date index from preprocessed daily
    if PREPROCESSED.exists():
        daily = pd.read_csv(PREPROCESSED, index_col=0, parse_dates=True)
        date_index = daily.index
    else:
        # Fall back: generate manually from rpt300 timestamps
        r300 = parse_rpt300()
        start = r300["start"].min().normalize()
        end   = r300["end"].max().normalize() + pd.Timedelta(days=1)
        date_index = pd.date_range(start, end, freq="D")

    print("Parsing rpt-300 (meter outages) …")
    rpt300 = parse_rpt300()
    print(f"  Found {len(rpt300)} meter outage events")
    for _, row in rpt300.iterrows():
        print(f"    {row['start']} → {row['end']}  ({row['duration_min']} min)")

    print("Parsing rpt-301 (modem outages) …")
    rpt301 = parse_rpt301()
    print(f"  Found {len(rpt301)} modem outage events")

    # Build label columns
    meter_labels = outages_to_daily_labels(rpt300, date_index)
    modem_labels = outages_to_daily_labels(rpt301, date_index)

    # Duration info per outage day
    rpt300["date"] = rpt300["start"].dt.normalize()
    duration_map = rpt300.groupby("date")["duration_min"].sum()

    gt = pd.DataFrame({
        "outage": meter_labels,
        "modem_outage": modem_labels,
    })
    gt["outage_duration_min"] = gt.index.map(duration_map).fillna(0).astype(int)

    # Annotate confidence: long outages (≥60 min) are "definite"
    gt["outage_confidence"] = "none"
    gt.loc[gt["outage"] == 1, "outage_confidence"] = "possible"
    gt.loc[(gt["outage"] == 1) & (gt["outage_duration_min"] >= 60), "outage_confidence"] = "definite"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    gt.to_csv(OUT_CSV)
    print(f"\nSaved → {OUT_CSV}")

    n_outage_days = gt["outage"].sum()
    n_definite    = (gt["outage_confidence"] == "definite").sum()
    print(f"Total outage days: {n_outage_days}  (definite: {n_definite})")
    print(f"Ground truth outage dates:")
    for d in gt[gt["outage"] == 1].index:
        dur = gt.loc[d, "outage_duration_min"]
        print(f"  {d.date()}  ({dur} min)")

    return gt


if __name__ == "__main__":
    main()
