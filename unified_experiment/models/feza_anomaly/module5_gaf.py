#!/usr/bin/env python3
"""
Module 5 - GAF-Based Unsupervised Anomaly Detection (Mask-Aware)
==================================================================
Reads from: RESULTS_DIR/master_dataset.csv, daily_features.csv, missing_data_summary.csv
Writes to:  RESULTS_DIR/gaf_anomaly_scores.csv, gaf_top10_anomalies.csv,
            imputation_log.csv, master_imputed.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RESULTS_DIR = os.environ.get("BEDAS_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))


def timeseries_to_gaf(series_24h):
    vmin, vmax = series_24h.min(), series_24h.max()
    if vmax - vmin < 1e-9:
        scaled = np.zeros_like(series_24h)
    else:
        scaled = 2 * (series_24h - vmin) / (vmax - vmin) - 1
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    gaf = np.cos(np.add.outer(phi, phi))
    return gaf


def build_monthly_medians(master):
    clean_days = master.groupby("Date").filter(lambda g: g["is_missing"].sum() == 0)
    if len(clean_days) == 0:
        return {}
    clean_days = clean_days.copy()
    clean_days["YM"] = clean_days["Date"].dt.to_period("M")
    medians = {}
    for (ym, hour), grp in clean_days.groupby(["YM", "Hour"]):
        medians[(ym.year, ym.month, hour)] = grp["Consumption_kWh"].median()
    return medians


def impute_for_gaf(master, medians):
    imputation_log = []
    result = master.copy()

    for idx in result.index:
        row = result.loc[idx]
        if row["is_missing"] == 1:
            dt = row["Date"]
            hour = int(row["Hour"])
            year = dt.year
            month = dt.month
            key = (year, month, hour)
            median_val = medians.get(key, 0.0)
            result.at[idx, "Consumption_kWh"] = median_val
            imputation_log.append({
                "Date": dt, "Hour": hour,
                "Imputed_Value": round(median_val, 4),
                "Method": "monthly_hourly_median_for_gaf",
            })

    return result, imputation_log


class SimpleAutoencoder:
    """576 -> 64 -> 16 -> 64 -> 576 feedforward autoencoder."""
    def __init__(self, input_dim=576, hidden1=64, hidden2=16, lr=0.001):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = np.random.randn(hidden2, hidden1) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(hidden1)
        self.W4 = np.random.randn(hidden1, input_dim) * np.sqrt(2.0 / hidden1)
        self.b4 = np.zeros(input_dim)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = self._relu(self.z3)
        self.z4 = self.a3 @ self.W4 + self.b4
        return self.z4

    def train_step(self, X):
        batch = X.shape[0]
        output = self.forward(X)
        loss = np.mean((output - X) ** 2)
        dz4 = 2 * (output - X) / batch
        dW4 = self.a3.T @ dz4;  db4 = dz4.sum(axis=0)
        da3 = dz4 @ self.W4.T
        dz3 = da3 * self._relu_deriv(self.z3)
        dW3 = self.a2.T @ dz3;  db3 = dz3.sum(axis=0)
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self._relu_deriv(self.z2)
        dW2 = self.a1.T @ dz2;  db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_deriv(self.z1)
        dW1 = X.T @ dz1;  db1 = dz1.sum(axis=0)
        self.W4 -= self.lr * dW4;  self.b4 -= self.lr * db4
        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1
        return loss

    def reconstruction_error(self, X):
        output = self.forward(X)
        return np.mean((output - X) ** 2, axis=1)


def main():
    print("=" * 70)
    print(" Module 5 - GAF Autoencoder (Mask-Aware)")
    print("=" * 70)

    master = pd.read_csv(os.path.join(RESULTS_DIR, "master_dataset.csv"), parse_dates=["Date"])
    daily = pd.read_csv(os.path.join(RESULTS_DIR, "daily_features.csv"), parse_dates=["Date"])
    missing_summary = pd.read_csv(os.path.join(RESULTS_DIR, "missing_data_summary.csv"), parse_dates=["Date"])

    dq_low_dates = set(missing_summary[missing_summary["data_quality_flag"] == "data_quality_low"]["Date"])
    partial_dates = set(missing_summary[missing_summary["data_quality_flag"] == "partial_missing"]["Date"])
    n_present_lookup = dict(zip(missing_summary["Date"], missing_summary["n_present"]))

    print(f"\nQuality summary:")
    print(f"   Total days: {master['Date'].nunique()}")
    print(f"   data_quality_low (excluded): {len(dq_low_dates)}")
    print(f"   partial_missing (imputed for GAF): {len(partial_dates)}")

    print("\nImputing missing hours for GAF input...")
    medians = build_monthly_medians(master)
    master_imputed, imp_log = impute_for_gaf(master, medians)

    print(f"   Imputed {len(imp_log)} hours")
    if imp_log:
        pd.DataFrame(imp_log).to_csv(os.path.join(RESULTS_DIR, "imputation_log.csv"), index=False)

    profiles = {}
    masks = {}
    for dt, grp in master_imputed.groupby("Date"):
        if dt in dq_low_dates:
            continue
        grp_sorted = grp.sort_values("Hour")
        if len(grp_sorted) == 24:
            vals = grp_sorted["Consumption_kWh"].values
            is_missing = grp_sorted["is_missing"].values
            if not np.any(np.isnan(vals)):
                profiles[dt] = vals
                masks[dt] = (1 - is_missing).astype(float)

    dates = sorted(profiles.keys())
    print(f"\n{len(dates)} days with 24h profiles (after excluding low-quality)")

    print("Generating GAF images...")
    gafs = {dt: timeseries_to_gaf(profiles[dt]) for dt in dates}
    gaf_matrix = np.array([gafs[dt].flatten() for dt in dates])

    zero_days = set(daily[daily["AllDayZero"]]["Date"].values)
    train_mask = np.array([dt not in zero_days and dt not in partial_dates for dt in dates])
    train_data = gaf_matrix[train_mask]
    print(f"   Training on {train_mask.sum()} complete normal days")

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    all_scaled = scaler.transform(gaf_matrix)

    print("Training autoencoder...")
    np.random.seed(42)
    ae = SimpleAutoencoder(input_dim=576, hidden1=64, hidden2=16, lr=0.0005)

    n_epochs = 200
    batch_size = min(64, len(train_scaled))
    for epoch in range(n_epochs):
        indices = np.random.permutation(len(train_scaled))
        epoch_loss = 0
        n_batches = 0
        for start in range(0, len(train_scaled), batch_size):
            batch = train_scaled[indices[start:start+batch_size]]
            loss = ae.train_step(batch)
            epoch_loss += loss
            n_batches += 1
        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch+1}/{n_epochs} -- Loss: {epoch_loss/n_batches:.6f}")

    print("Computing mask-aware reconstruction errors...")
    raw_errors = ae.reconstruction_error(all_scaled)

    results_rows = []
    for i, dt in enumerate(dates):
        n_present = n_present_lookup.get(dt, 24)
        presence_ratio = n_present / 24.0
        raw_err = raw_errors[i]
        mask_aware_err = raw_err * presence_ratio

        results_rows.append({
            "Date": dt,
            "GAF_ReconError_Raw": raw_err,
            "n_present": n_present,
            "presence_ratio": round(presence_ratio, 4),
            "GAF_ReconError_MaskAware": mask_aware_err,
        })

    results = pd.DataFrame(results_rows)
    err_min = results["GAF_ReconError_MaskAware"].min()
    err_max = results["GAF_ReconError_MaskAware"].max()
    results["GAF_Anomaly_Score"] = (results["GAF_ReconError_MaskAware"] - err_min) / (err_max - err_min + 1e-9)
    results = results.sort_values("GAF_Anomaly_Score", ascending=False)

    out_all = os.path.join(RESULTS_DIR, "gaf_anomaly_scores.csv")
    results.to_csv(out_all, index=False)

    top10 = results.head(10)
    out_top10 = os.path.join(RESULTS_DIR, "gaf_top10_anomalies.csv")
    top10.to_csv(out_top10, index=False)

    print(f"\nTop 10 Most Anomalous Days (mask-aware GAF score):")
    print(top10[["Date", "GAF_Anomaly_Score", "n_present",
                  "GAF_ReconError_Raw"]].to_string(index=False))

    print(f"\nSaved: {out_all}")
    print(f"Saved: {out_top10}")
    print("=" * 70)


if __name__ == "__main__":
    main()
