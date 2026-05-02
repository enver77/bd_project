import pandas as pd
import numpy as np
import os

# 1. Hourly IF: from anomaly_merged_results.csv
df_if = pd.read_csv("C:/Users/enver/OneDrive - Kadir Has University/bedas_project/raporlar26/anomaly_merged_results.csv")
df_if["Date"] = pd.to_datetime(df_if["timestamp"]).dt.date.astype(str)
if_flagged_days = set(df_if[df_if["if_anomaly_A"] == 1]["Date"])

# 2. Three-Tier: from anomaly_report.csv
df_tt = pd.read_csv("Ali's_report/OSOS_Data_Project/outputs/anomaly_report.csv")
tt_flagged_days = set(df_tt[df_tt["confidence"].isin(["moderate", "high"])]["date"])

# 3. Supervised GBDT: predictive_risk_scores.csv
df_gbdt = pd.read_csv("feza_reports_old/predictive_risk_scores.csv")
gbdt_flagged_days = set(df_gbdt[df_gbdt["RiskLevel"] == "HIGH_RISK"]["Date"])

# 4. GAF: gaf_anomaly_scores.csv
df_gaf = pd.read_csv("feza_reports_old/gaf_anomaly_scores.csv")
df_gaf = df_gaf.sort_values("GAF_Anomaly_Score", ascending=False).reset_index(drop=True)
top_10_days = set(df_gaf.head(10)["Date"])
top_5_pct_days = set(df_gaf.head(int(len(df_gaf) * 0.05))["Date"])

gaf_flagged_days = top_5_pct_days

all_dates = sorted(list(set(df_if["Date"]) | set(df_tt["date"]) | set(df_gbdt["Date"]) | set(df_gaf["Date"])))

# create matrix
df_matrix = pd.DataFrame({"Date": all_dates})
df_matrix["IF"] = df_matrix["Date"].isin(if_flagged_days).astype(int)
df_matrix["TT"] = df_matrix["Date"].isin(tt_flagged_days).astype(int)
df_matrix["GBDT"] = df_matrix["Date"].isin(gbdt_flagged_days).astype(int)
df_matrix["GAF"] = df_matrix["Date"].isin(gaf_flagged_days).astype(int)

df_matrix["Flag_Count"] = df_matrix["IF"] + df_matrix["TT"] + df_matrix["GBDT"] + df_matrix["GAF"]

# Save matrix locally
df_matrix.to_csv("cross_method_daily_flags.csv", index=False)

def check_coverage(flagged_dates, start, end=None):
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end else start_dt
    mask = (pd.to_datetime(list(flagged_dates)) >= start_dt) & (pd.to_datetime(list(flagged_dates)) <= end_dt)
    return any(mask)

events = {
    "Dec 2025 degradation": ("2025-12-01", "2025-12-31"),
    "Mar 2025 sudden drops": ("2025-03-01", "2025-03-31"),
    "12 May 2025 nighttime outage": ("2025-05-12", "2025-05-12"),
}

rules = {
    "Union ($\\ge 1$ method)": df_matrix[df_matrix["Flag_Count"] >= 1]["Date"].tolist(),
    "Agreement ($\\ge 2$ methods)": df_matrix[df_matrix["Flag_Count"] >= 2]["Date"].tolist(),
    "Strong Agreement ($\\ge 3$ methods)": df_matrix[df_matrix["Flag_Count"] >= 3]["Date"].tolist(),
    "All-method (4 methods)": df_matrix[df_matrix["Flag_Count"] == 4]["Date"].tolist()
}

summary_data = []
for rule_name, rule_dates in rules.items():
    row = {"Rule": rule_name, "Flagged Days": len(rule_dates)}
    for ev_name, (ev_start, ev_end) in events.items():
        row[ev_name] = "Yes" if check_coverage(rule_dates, ev_start, ev_end) else "No"
    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv("agreement_burden_summary.csv", index=False)

print(df_summary.to_string())
