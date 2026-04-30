#!/usr/bin/env python3
"""
Module 6 v2 - Combined Score & Maintenance Output (Mask-Aware)
===============================================================
Reads from: RESULTS_DIR/anomaly_flags.csv, daily_anomaly_scores.csv,
            gaf_anomaly_scores.csv, changepoints_filtered.csv
Writes to:  RESULTS_DIR/combined_scores_v2.csv, maintenance_list_v2.csv,
            data_issues_list.csv, maintenance_output.csv
"""

import os
import numpy as np
import pandas as pd

RESULTS_DIR = os.environ.get("BEDAS_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))

WEIGHTS = {
    "rule_flags":  0.30,
    "forecast":    0.30,
    "gaf":         0.20,
    "changepoint": 0.20,
}


def normalize_0_1(series):
    mn, mx = series.min(), series.max()
    if mx - mn < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def main():
    print("=" * 70)
    print(" Module 6 v2 - Combined Score (Mask-Aware)")
    print("=" * 70)

    flags = pd.read_csv(os.path.join(RESULTS_DIR, "anomaly_flags.csv"), parse_dates=["Date"])
    forecast = pd.read_csv(os.path.join(RESULTS_DIR, "daily_anomaly_scores.csv"), parse_dates=["Date"])
    gaf = pd.read_csv(os.path.join(RESULTS_DIR, "gaf_anomaly_scores.csv"), parse_dates=["Date"])

    cp_file = os.path.join(RESULTS_DIR, "changepoints_filtered.csv")
    if os.path.exists(cp_file):
        cp_df = pd.read_csv(cp_file)
    else:
        cp_df = pd.DataFrame(columns=["ChangePoint_Date"])

    adz_mask = flags["Flag_AllDayZero"] == True
    dql_mask = flags["data_quality_flag"] == "data_quality_low"

    data_issues = flags[adz_mask | dql_mask][["Date", "OSF_ID", "DailyTotal",
                                               "missing_ratio", "data_quality_flag"]].copy()
    data_issues["Category"] = data_issues.apply(
        lambda r: "CRITICAL_DATA_ISSUE" if r["DailyTotal"] == 0.0 and r["missing_ratio"] < 1.0
                  else "DATA_QUALITY_LOW", axis=1)
    data_issues["Reason"] = data_issues.apply(
        lambda r: ("AllDayZero - All present hours show zero consumption"
                   if r["Category"] == "CRITICAL_DATA_ISSUE"
                   else f"Insufficient data - {r['missing_ratio']*100:.0f}% hours missing"), axis=1)

    out_data_issues = os.path.join(RESULTS_DIR, "data_issues_list.csv")
    data_issues.to_csv(out_data_issues, index=False)
    print(f"\nData Issues: {len(data_issues)} days")

    operational = flags[~adz_mask & ~dql_mask].copy()
    print(f"Operational days for scoring: {len(operational)}")

    op_flags = ["Flag_DaytimeNonZero", "Flag_NightDrop", "Flag_ScheduleShift"]
    operational["OpFlagCount"] = operational[op_flags].sum(axis=1).astype(float)
    operational.loc[operational["Flag_NightDrop"] == True, "OpFlagCount"] += 1.0
    operational["RuleScore"] = normalize_0_1(operational["OpFlagCount"])

    fcast = forecast[forecast["Daily_Anomaly_Score"].notna()].copy()
    fcast = fcast[~fcast["Date"].isin(data_issues["Date"])]
    fcast["ForecastScore"] = normalize_0_1(fcast["Daily_Anomaly_Score"])
    operational = operational.merge(
        fcast[["Date", "ForecastScore", "Anomaly_Hours", "Mean_Abs_Residual"]],
        on="Date", how="left")

    gaf_op = gaf[~gaf["Date"].isin(data_issues["Date"])].copy()
    gaf_op["GAFScore"] = normalize_0_1(gaf_op["GAF_Anomaly_Score"])
    operational = operational.merge(gaf_op[["Date", "GAFScore"]], on="Date", how="left")

    if len(cp_df) > 0:
        cp_dates = pd.to_datetime(cp_df["ChangePoint_Date"].unique())
    else:
        cp_dates = pd.DatetimeIndex([])

    cp_prox = []
    for dt in operational["Date"].values:
        if len(cp_dates) == 0:
            cp_prox.append({"Date": dt, "CP_Proximity_Days": 999})
            continue
        diffs = np.abs((cp_dates - dt).days)
        cp_prox.append({"Date": dt, "CP_Proximity_Days": int(diffs.min())})
    cp_prox_df = pd.DataFrame(cp_prox)
    cp_prox_df["CPScore"] = normalize_0_1(1.0 / (cp_prox_df["CP_Proximity_Days"] + 1))
    operational = operational.merge(cp_prox_df[["Date", "CPScore", "CP_Proximity_Days"]],
                                   on="Date", how="left")

    for col in ["RuleScore", "ForecastScore", "GAFScore", "CPScore"]:
        operational[col] = operational[col].fillna(0)

    operational["CombinedScore"] = (
        WEIGHTS["rule_flags"]  * operational["RuleScore"] +
        WEIGHTS["forecast"]    * operational["ForecastScore"] +
        WEIGHTS["gaf"]         * operational["GAFScore"] +
        WEIGHTS["changepoint"] * operational["CPScore"]
    )

    operational["AnomalyTypes"] = operational.apply(
        lambda r: ", ".join([c.replace("Flag_", "") for c in op_flags if r[c]]), axis=1)

    def classify(row):
        if row["CombinedScore"] > 0.6: return "CRITICAL_OPERATIONAL"
        if row["CombinedScore"] > 0.4: return "HIGH"
        if row["CombinedScore"] > 0.25: return "MEDIUM"
        if row["CombinedScore"] > 0.12: return "LOW"
        return "NORMAL"

    operational["Priority"] = operational.apply(classify, axis=1)
    operational = operational.sort_values("Date")

    out_combined = os.path.join(RESULTS_DIR, "combined_scores_v2.csv")
    operational.to_csv(out_combined, index=False)

    maint = operational[operational["Priority"] != "NORMAL"].sort_values("CombinedScore", ascending=False)
    maint_cols = ["Date", "OSF_ID", "Priority", "CombinedScore", "AnomalyTypes",
                  "RuleScore", "ForecastScore", "GAFScore", "CPScore",
                  "Anomaly_Hours", "CP_Proximity_Days", "missing_ratio", "data_quality_flag"]
    out_maint = os.path.join(RESULTS_DIR, "maintenance_list_v2.csv")
    maint[maint_cols].to_csv(out_maint, index=False)

    # Maintenance output (final)
    all_dates = flags[["Date", "OSF_ID", "DailyTotal", "missing_ratio", "data_quality_flag"]].copy()
    all_dates = all_dates.merge(
        operational[["Date", "CombinedScore", "Priority", "AnomalyTypes"]],
        on="Date", how="left")

    all_dates.loc[all_dates["Priority"].isna() & adz_mask.values, "Priority"] = "DATA_ISSUE"
    all_dates.loc[all_dates["Priority"].isna() & dql_mask.values, "Priority"] = "DATA_QUALITY_LOW"
    all_dates.loc[all_dates["Priority"].isna(), "Priority"] = "NORMAL"
    all_dates["CombinedScore"] = all_dates["CombinedScore"].fillna(0)
    all_dates["AnomalyTypes"] = all_dates["AnomalyTypes"].fillna("")

    def make_notes(row):
        notes = []
        if row["data_quality_flag"] == "data_quality_low":
            notes.append(f"Low data quality ({row['missing_ratio']*100:.0f}% missing)")
        elif row["missing_ratio"] > 0:
            notes.append(f"Partial missing ({row['missing_ratio']*100:.0f}% missing, scores mask-aware)")
        if row["AnomalyTypes"]:
            notes.append(f"Flags: {row['AnomalyTypes']}")
        return "; ".join(notes) if notes else "Normal operation"

    all_dates["Notes"] = all_dates.apply(make_notes, axis=1)
    maint_output = all_dates.rename(columns={"CombinedScore": "anomaly_score"})
    maint_output = maint_output[["Date", "anomaly_score", "missing_ratio",
                                  "data_quality_flag", "Priority", "AnomalyTypes", "Notes"]]
    maint_output = maint_output.sort_values("Date")

    out_maint_output = os.path.join(RESULTS_DIR, "maintenance_output.csv")
    maint_output.to_csv(out_maint_output, index=False)

    print(f"\nScore Distribution ({len(operational)} operational days):")
    for p in ["CRITICAL_OPERATIONAL", "HIGH", "MEDIUM", "LOW", "NORMAL"]:
        n = (operational["Priority"] == p).sum()
        if n > 0:
            print(f"   {p:25s}: {n:4d} days")

    print(f"\nTop 20 Maintenance List:")
    pd.set_option("display.width", 220)
    print(maint[maint_cols].head(20).to_string(index=False))

    print(f"\nSaved: {out_combined}")
    print(f"Saved: {out_maint}")
    print(f"Saved: {out_data_issues}")
    print(f"Saved: {out_maint_output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
