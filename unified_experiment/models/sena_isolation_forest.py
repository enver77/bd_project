#!/usr/bin/env python3
"""
Sena's Approach - Isolation Forest Anomaly Detection
=====================================================
Adapted to use unified preprocessing output.

Produces the SAME outputs and graph types as Sena's original notebooks:
  1. Anomaly bar chart per month
  2. Day-of-month x month heatmap
  3. Hour-of-day x month heatmap
  4. Seasonal pie chart
  5. Time series with anomaly scatter (test set)
  6. Hourly anomaly distribution bar chart
  7. Monthly anomaly distribution bar chart

Reads from: RESULTS_DIR/hourly_data_labeled.csv
Writes to:  RESULTS_DIR/sena_*.csv and RESULTS_DIR/sena_*.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = os.environ.get("BEDAS_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def main():
    print("=" * 70)
    print(" Sena's Approach - Isolation Forest (Unified Data)")
    print("=" * 70)

    # ── Load unified data ──
    df = pd.read_csv(os.path.join(RESULTS_DIR, "hourly_data_labeled.csv"),
                     parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"\nLoaded: {len(df)} rows")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    # ══════════════════════════════════════════════════════════════════════
    # Part 1: Full-dataset Isolation Forest (same as Sena's 14_ay notebook)
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Part 1: Full-Dataset Isolation Forest ──")

    model_full = IsolationForest(contamination=0.03, random_state=42)
    df["is_anomaly"] = model_full.fit_predict(df[["hour", "energy_kwh"]])

    anomalies = df[df["is_anomaly"] == -1]
    print(f"Total anomalies detected: {len(anomalies)}")

    # ── Graph 1: Anomaly bar chart per month ──
    df["month_label"] = df["timestamp"].dt.strftime("%b %Y")
    anomaly_dist = anomalies.groupby(
        anomalies["timestamp"].dt.to_period("M")).size()

    month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

    clean_labels = [f"{month_map[p.month]} {p.year}" for p in anomaly_dist.index]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(clean_labels, anomaly_dist.values, color="red",
                   edgecolor="black", alpha=0.8)
    plt.title("Number of Anomalies (Failure Risks) per Month",
              fontsize=15, fontweight="bold")
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Count of Anomalies", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5,
                 int(yval), ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sena_anomaly_per_month.png"), dpi=150)
    plt.close()

    # ── Graph 2: Day-of-month x month heatmap ──
    anomalies_copy = anomalies.copy()
    anomalies_copy["month_str"] = anomalies_copy["timestamp"].dt.strftime("%b %Y")
    anomalies_copy["day_of_month"] = anomalies_copy["timestamp"].dt.day

    heatmap_df = anomalies_copy.groupby(["month_str", "day_of_month"]).size().unstack(fill_value=0)

    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlOrRd",
                cbar_kws={"label": "Anomaly Count"})
    plt.title("System-Wide Anomaly Calendar: Distribution by Day and Month",
              fontsize=18, fontweight="bold")
    plt.xlabel("Day of the Month", fontsize=14)
    plt.ylabel("Month", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sena_anomaly_heatmap_day.png"), dpi=150)
    plt.close()

    # ── Graph 3: Hour-of-day x month heatmap ──
    hourly_heatmap = anomalies_copy.groupby(["month_str", "hour"]).size().unstack(fill_value=0)
    all_hours = list(range(24))
    hourly_heatmap = hourly_heatmap.reindex(columns=all_hours, fill_value=0)

    plt.figure(figsize=(20, 10))
    sns.heatmap(hourly_heatmap, annot=True, fmt="d", cmap="YlOrRd",
                cbar_kws={"label": "Anomaly Count"})
    plt.title("24-Hour Anomaly Distribution Across 14 Months",
              fontsize=16, fontweight="bold")
    plt.xlabel("Hour of the Day (0-23)", fontsize=12)
    plt.ylabel("Month", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sena_anomaly_heatmap_hour.png"), dpi=150)
    plt.close()

    # ── Graph 4: Seasonal pie chart ──
    def get_season(ts):
        m = ts.month
        if m in [12, 1, 2]: return "Winter"
        if m in [3, 4, 5]: return "Spring"
        if m in [6, 7, 8]: return "Summer"
        return "Autumn"

    df["season"] = df["timestamp"].apply(get_season)
    season_anomalies = df[df["is_anomaly"] == -1].groupby("season").size()

    plt.figure(figsize=(10, 8))
    colors = ["#5DADE2", "#F39C12", "#58D68D", "#EC7063"]
    season_order = ["Winter", "Summer", "Spring", "Autumn"]
    season_vals = [season_anomalies.get(s, 0) for s in season_order]

    plt.pie(season_vals, labels=season_order, autopct="%1.1f%%",
            startangle=140, colors=colors, explode=(0.1, 0, 0, 0), shadow=True)
    plt.title("Seasonal Distribution of Anomalies (Failure Risks)",
              fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sena_seasonal_pie.png"), dpi=150)
    plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # Part 2: Train/Test Isolation Forest (same as Sena's train-test notebook)
    # ══════════════════════════════════════════════════════════════════════
    print("\n── Part 2: Train/Test Isolation Forest ──")

    # Feature engineering (same as Sena)
    ts_df = df[["timestamp", "energy_kwh"]].copy()
    ts_df = ts_df.rename(columns={"energy_kwh": "consumption"})
    ts_df = ts_df.set_index("timestamp")

    ts_df["hour"] = ts_df.index.hour
    ts_df["dayofweek"] = ts_df.index.dayofweek
    ts_df["month"] = ts_df.index.month
    ts_df["hour_sin"] = np.sin(2 * np.pi * ts_df["hour"] / 24)
    ts_df["hour_cos"] = np.cos(2 * np.pi * ts_df["hour"] / 24)
    ts_df["lag24"] = ts_df["consumption"].shift(24)
    ts_df = ts_df.dropna().copy()

    # Train/test split (same as Sena: before/after Dec 2025)
    train = ts_df[ts_df.index < "2025-12-01"].copy()
    test = ts_df[ts_df.index >= "2025-12-01"].copy()

    print(f"Train: {len(train)} rows | Test: {len(test)} rows")

    features = ["consumption", "hour_sin", "hour_cos", "dayofweek", "lag24"]
    X_train = train[features]
    X_test = test[features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_tt = IsolationForest(contamination=0.03, n_estimators=200, random_state=42)
    model_tt.fit(X_train_scaled)

    test["pred"] = model_tt.predict(X_test_scaled)
    test["anomaly"] = (test["pred"] == -1).astype(int)

    print(f"Test anomaly count: {test['anomaly'].sum()}")
    print(f"Test anomaly rate: {test['anomaly'].mean():.4f}")

    # ── Graph 5: Time series with anomaly scatter ──
    plt.figure(figsize=(16, 6))
    plt.plot(test.index, test["consumption"], label="Consumption")
    plt.scatter(test.index[test["anomaly"] == 1],
                test.loc[test["anomaly"] == 1, "consumption"],
                color="red", s=20, label="Anomaly")
    plt.title("Isolation Forest Anomalies - Test Set")
    plt.xlabel("Time")
    plt.ylabel("Consumption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sena_test_anomalies.png"), dpi=150)
    plt.close()

    # ── Graph 6: Hourly anomaly distribution ──
    hourly_anomalies = test.groupby(test.index.hour)["anomaly"].sum()
    plt.figure(figsize=(10, 4))
    hourly_anomalies.plot(kind="bar")
    plt.title("Hourly Distribution of Anomalies - Test Set")
    plt.xlabel("Hour")
    plt.ylabel("Anomaly Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sena_hourly_anomalies.png"), dpi=150)
    plt.close()

    # ── Graph 7: Monthly anomaly distribution ──
    monthly_anomalies = test.groupby(test.index.month)["anomaly"].sum()
    plt.figure(figsize=(8, 4))
    monthly_anomalies.plot(kind="bar")
    plt.title("Monthly Distribution of Anomalies - Test Set")
    plt.xlabel("Month")
    plt.ylabel("Anomaly Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sena_monthly_anomalies.png"), dpi=150)
    plt.close()

    # ── Save results ──
    test.to_csv(os.path.join(RESULTS_DIR, "sena_test_anomaly_results.csv"))
    train.to_csv(os.path.join(RESULTS_DIR, "sena_train_anomaly_results.csv"))

    # Full-dataset anomaly results
    anomaly_summary = df[["timestamp", "energy_kwh", "hour", "is_anomaly"]].copy()
    anomaly_summary.to_csv(os.path.join(RESULTS_DIR, "sena_full_anomaly_results.csv"), index=False)

    print(f"\nSaved all Sena outputs to: {RESULTS_DIR}/sena_*")
    print("=" * 70)


if __name__ == "__main__":
    main()
