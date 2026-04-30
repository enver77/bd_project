#!/usr/bin/env python3
"""
Experiment: Enver's Merged Data Strategy on Feza's Anomaly Pipeline
=====================================================================
Feeds cekis + akim-gerilim merged features into Feza's GAF autoencoder
to see if electrical-domain features improve unsupervised anomaly detection.

Strategy:
  Model A: GAF autoencoder on energy consumption only (Feza's original)
  Model B: GAF autoencoder on energy + akim-gerilim features

Both use Feza's preprocessing (mask-aware, cutoff-safe).

Reads from: RESULTS_DIR/master_dataset.csv, missing_data_summary.csv,
            DATA_DIR/akim-gerilim_raporu_Vkx3H.xlsx
Writes to:  RESULTS_DIR/merged_gaf_comparison.csv, merged_gaf_*.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

RESULTS_DIR = os.environ.get("BEDAS_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
DATA_DIR = os.environ.get("BEDAS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def timeseries_to_gaf(series):
    """Convert a time series to a Gramian Angular Summation Field."""
    vmin, vmax = series.min(), series.max()
    if vmax - vmin < 1e-9:
        scaled = np.zeros_like(series)
    else:
        scaled = 2 * (series - vmin) / (vmax - vmin) - 1
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    return np.cos(np.add.outer(phi, phi))


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


class SimpleAutoencoder:
    """Configurable feedforward autoencoder."""
    def __init__(self, input_dim, hidden1=64, hidden2=16, lr=0.0005):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = np.random.randn(hidden2, hidden1) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(hidden1)
        self.W4 = np.random.randn(hidden1, input_dim) * np.sqrt(2.0 / hidden1)
        self.b4 = np.zeros(input_dim)

    def _relu(self, x): return np.maximum(0, x)
    def _relu_d(self, x): return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1; self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.a2 = self._relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3; self.a3 = self._relu(self.z3)
        self.z4 = self.a3 @ self.W4 + self.b4
        return self.z4

    def train_step(self, X):
        batch = X.shape[0]; output = self.forward(X)
        loss = np.mean((output - X) ** 2)
        dz4 = 2 * (output - X) / batch
        dW4 = self.a3.T @ dz4; db4 = dz4.sum(0)
        dz3 = (dz4 @ self.W4.T) * self._relu_d(self.z3)
        dW3 = self.a2.T @ dz3; db3 = dz3.sum(0)
        dz2 = (dz3 @ self.W3.T) * self._relu_d(self.z2)
        dW2 = self.a1.T @ dz2; db2 = dz2.sum(0)
        dz1 = (dz2 @ self.W2.T) * self._relu_d(self.z1)
        dW1 = X.T @ dz1; db1 = dz1.sum(0)
        self.W4 -= self.lr*dW4; self.b4 -= self.lr*db4
        self.W3 -= self.lr*dW3; self.b3 -= self.lr*db3
        self.W2 -= self.lr*dW2; self.b2 -= self.lr*db2
        self.W1 -= self.lr*dW1; self.b1 -= self.lr*db1
        return loss

    def recon_error(self, X):
        return np.mean((self.forward(X) - X) ** 2, axis=1)


def train_and_score(data_matrix, train_mask, dates, n_present_lookup, label, seed=42):
    """Train autoencoder and return mask-aware scores."""
    train_data = data_matrix[train_mask]
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    all_scaled = scaler.transform(data_matrix)

    np.random.seed(seed)
    input_dim = data_matrix.shape[1]
    h1 = min(128, max(32, input_dim // 4))
    h2 = min(32, max(8, input_dim // 16))
    ae = SimpleAutoencoder(input_dim=input_dim, hidden1=h1, hidden2=h2, lr=0.0005)

    n_epochs = 200
    batch_size = min(64, len(train_scaled))
    for epoch in range(n_epochs):
        indices = np.random.permutation(len(train_scaled))
        for start in range(0, len(train_scaled), batch_size):
            ae.train_step(train_scaled[indices[start:start+batch_size]])
        if (epoch + 1) % 50 == 0:
            loss = np.mean(ae.recon_error(train_scaled))
            print(f"   [{label}] Epoch {epoch+1}/{n_epochs} -- Loss: {loss:.6f}")

    raw_errors = ae.recon_error(all_scaled)
    scores = []
    for i, dt in enumerate(dates):
        n_present = n_present_lookup.get(dt, 24)
        presence_ratio = n_present / 24.0
        scores.append(raw_errors[i] * presence_ratio)

    scores = np.array(scores)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return scores_norm, raw_errors


def main():
    print("=" * 70)
    print(" Experiment: Merged Data on Feza's GAF Pipeline")
    print("=" * 70)

    master = pd.read_csv(os.path.join(RESULTS_DIR, "master_dataset.csv"), parse_dates=["Date"])
    missing_summary = pd.read_csv(os.path.join(RESULTS_DIR, "missing_data_summary.csv"), parse_dates=["Date"])

    dq_low_dates = set(missing_summary[missing_summary["data_quality_flag"] == "data_quality_low"]["Date"])
    n_present_lookup = dict(zip(missing_summary["Date"], missing_summary["n_present"]))

    # Impute for GAF
    medians = build_monthly_medians(master)
    master_imp = master.copy()
    for idx in master_imp.index:
        if master_imp.loc[idx, "is_missing"] == 1:
            dt = master_imp.loc[idx, "Date"]
            hour = int(master_imp.loc[idx, "Hour"])
            key = (dt.year, dt.month, hour)
            master_imp.at[idx, "Consumption_kWh"] = medians.get(key, 0.0)

    # Build 24h profiles (excluding low-quality days)
    profiles = {}
    for dt, grp in master_imp.groupby("Date"):
        if dt in dq_low_dates: continue
        grp_sorted = grp.sort_values("Hour")
        if len(grp_sorted) == 24:
            vals = grp_sorted["Consumption_kWh"].values
            if not np.any(np.isnan(vals)):
                profiles[dt] = vals

    dates = sorted(profiles.keys())
    print(f"\n{len(dates)} days with valid 24h profiles")

    # ── Model A: Energy-only GAF (Feza's original) ──
    print("\n-- Model A: Energy-only GAF --")
    gaf_A = np.array([timeseries_to_gaf(profiles[dt]).flatten() for dt in dates])

    daily = pd.read_csv(os.path.join(RESULTS_DIR, "daily_features.csv"), parse_dates=["Date"])
    zero_days = set(daily[daily["AllDayZero"]]["Date"].values)
    partial_dates = set(missing_summary[missing_summary["data_quality_flag"] == "partial_missing"]["Date"])
    train_mask = np.array([dt not in zero_days and dt not in partial_dates for dt in dates])

    scores_A, raw_A = train_and_score(gaf_A, train_mask, dates, n_present_lookup, "Model A")

    # ── Model B: Energy + Akim-Gerilim GAF ──
    akim_file = os.path.join(DATA_DIR, "akim-gerilim_raporu_Vkx3H.xlsx")
    if not os.path.exists(akim_file):
        print(f"\nWARNING: {akim_file} not found. Skipping Model B.")
        scores_B = None
    else:
        print("\n-- Model B: Energy + Akim-Gerilim GAF --")
        akim = pd.read_excel(akim_file)
        akim_ts_col = akim.columns[0]
        akim[akim_ts_col] = pd.to_datetime(akim[akim_ts_col])
        akim["timestamp_hour"] = akim[akim_ts_col].dt.floor("h")
        akim_hourly = akim.drop(columns=[akim_ts_col]).groupby("timestamp_hour").mean().reset_index()
        akim_hourly.rename(columns={"timestamp_hour": "timestamp"}, inplace=True)

        # Build daily akim profiles (aggregate hourly to 24h)
        akim_hourly["Date"] = akim_hourly["timestamp"].dt.normalize()
        akim_hourly["Hour"] = akim_hourly["timestamp"].dt.hour
        akim_numeric = akim_hourly.select_dtypes(include=[np.number])
        akim_numeric["Date"] = akim_hourly["Date"]
        akim_numeric["Hour"] = akim_hourly["Hour"]

        n_akim_features = len([c for c in akim_numeric.columns if c not in ["Date", "Hour"]])
        print(f"   Akim-gerilim features per hour: {n_akim_features}")

        # Build combined profiles: energy GAF + akim flat vector
        gaf_B_list = []
        valid_dates_B = []
        for dt in dates:
            dt_ts = pd.Timestamp(dt)
            day_akim = akim_numeric[akim_numeric["Date"] == dt_ts]
            if len(day_akim) < 12:  # need at least 12 hours of akim data
                continue

            # Reindex to 24 hours, interpolate
            day_akim = day_akim.set_index("Hour").drop(columns=["Date"])
            day_akim = day_akim.reindex(range(24)).interpolate().fillna(0)

            energy_gaf = timeseries_to_gaf(profiles[dt]).flatten()  # 576
            akim_flat = day_akim.values.flatten()  # 24 * n_akim_features

            combined = np.concatenate([energy_gaf, akim_flat])
            gaf_B_list.append(combined)
            valid_dates_B.append(dt)

        if len(gaf_B_list) > 0:
            gaf_B = np.array(gaf_B_list)
            train_mask_B = np.array([dt not in zero_days and dt not in partial_dates
                                     for dt in valid_dates_B])
            n_present_B = {dt: n_present_lookup.get(dt, 24) for dt in valid_dates_B}

            print(f"   {len(valid_dates_B)} days with both energy + akim profiles")
            scores_B, raw_B = train_and_score(gaf_B, train_mask_B, valid_dates_B, n_present_B, "Model B")
        else:
            print("   No days with sufficient akim data. Skipping Model B.")
            scores_B = None

    # ── Comparison ──
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    results_A = pd.DataFrame({"Date": dates, "Score_A": scores_A, "Raw_A": raw_A})

    if scores_B is not None:
        results_B = pd.DataFrame({"Date": valid_dates_B, "Score_B": scores_B, "Raw_B": raw_B})
        comparison = results_A.merge(results_B, on="Date", how="inner")
        comparison = comparison.sort_values("Score_A", ascending=False)

        # Top anomalies comparison
        print("\nTop 15 anomaly days (Model A score):")
        top = comparison.head(15)
        print(top[["Date", "Score_A", "Score_B"]].to_string(index=False))

        # Correlation
        corr = comparison["Score_A"].corr(comparison["Score_B"])
        print(f"\nScore correlation (A vs B): {corr:.4f}")

        # Rank agreement at different thresholds
        for pct in [5, 10, 15]:
            n = max(1, int(len(comparison) * pct / 100))
            top_A = set(comparison.nlargest(n, "Score_A")["Date"])
            top_B = set(comparison.nlargest(n, "Score_B")["Date"])
            overlap = len(top_A & top_B)
            print(f"   Top-{pct}% overlap: {overlap}/{n} ({100*overlap/n:.0f}%)")

        comparison.to_csv(os.path.join(RESULTS_DIR, "merged_gaf_comparison.csv"), index=False)

        # ── Plot: Score comparison ──
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(comparison["Score_A"], comparison["Score_B"],
                       alpha=0.5, s=10)
        axes[0].plot([0, 1], [0, 1], "r--", alpha=0.5)
        axes[0].set_xlabel("Model A Score (Energy Only)")
        axes[0].set_ylabel("Model B Score (Energy + Akim)")
        axes[0].set_title(f"Anomaly Score Comparison (r={corr:.3f})")

        axes[1].hist(comparison["Score_A"], bins=30, alpha=0.5, label="Model A")
        axes[1].hist(comparison["Score_B"], bins=30, alpha=0.5, label="Model B")
        axes[1].set_xlabel("Anomaly Score")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Score Distribution")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "merged_gaf_comparison.png"), dpi=150)
        plt.close()
    else:
        results_A.to_csv(os.path.join(RESULTS_DIR, "merged_gaf_comparison.csv"), index=False)
        print("Only Model A results available (no akim-gerilim data).")

    print(f"\nSaved to: {RESULTS_DIR}/merged_gaf_*")
    print("=" * 70)


if __name__ == "__main__":
    main()
