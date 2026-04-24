#!/usr/bin/env python3
"""
Module 5-M -- GAF-Based Anomaly Detection with Merged Data (Energy + Electrical)
==================================================================================
Extends Feza's Module 5 (GAF autoencoder) by adding current, voltage, and power
factor channels from akim-gerilim data.

Approach:
  - Uses Feza's mask-aware preprocessing (master_imputed.csv)
  - Merges akim-gerilim (current/voltage/pf) hourly data
  - Builds multi-channel GAF: one GASF image per feature channel
    * Channel 1: Energy consumption (original)
    * Channel 2: Average current (R/S/T mean)
    * Channel 3: Average voltage (R/S/T mean)
    * Channel 4: Current imbalance (std across R/S/T)
  - Concatenates flattened GAFs -> larger autoencoder input
  - Mask-aware scoring identical to Module 5

Comparison:
  - Model A: Energy-only GAF (replicates Module 5 baseline)
  - Model B: Energy + Electrical GAF (4-channel)

Outputs:
  - gaf_merged_scores.csv       : per-day anomaly scores (both models)
  - gaf_merged_top20.csv        : top 20 anomalous days comparison
  - gaf_merged_comparison.csv   : detection rates against known failures
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Akim-gerilim file -- same one used in raporlar26 pipeline
RAPORLAR_DIR = os.path.join(DATA_DIR, "..", "..", "..", "..", "raporlar26")
AKIM_FILE = os.path.join(RAPORLAR_DIR, "akim-gerilim_raporu_Vkx3H.xlsx")
FAILURE_FILE = os.path.join(RAPORLAR_DIR,
                            "rpt-308_modem_kesinti_raporu_(butun_kesintiler)_bPZj6.xlsx")


# ---------------------------------------------------------------------------
# GAF Transformation (identical to module5_gaf.py)
# ---------------------------------------------------------------------------
def timeseries_to_gaf(series_24h: np.ndarray) -> np.ndarray:
    """Convert a 24-value time series to a 24x24 Gramian Angular Summation Field."""
    vmin, vmax = series_24h.min(), series_24h.max()
    if vmax - vmin < 1e-9:
        scaled = np.zeros_like(series_24h)
    else:
        scaled = 2 * (series_24h - vmin) / (vmax - vmin) - 1
    scaled = np.clip(scaled, -1, 1)
    phi = np.arccos(scaled)
    gaf = np.cos(np.add.outer(phi, phi))
    return gaf


# ---------------------------------------------------------------------------
# Simple Autoencoder (same architecture as module5, but flexible input_dim)
# ---------------------------------------------------------------------------
class SimpleAutoencoder:
    """Flexible feedforward autoencoder: input_dim -> hidden1 -> hidden2 -> hidden1 -> input_dim."""
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
        dW4 = self.a3.T @ dz4
        db4 = dz4.sum(axis=0)
        da3 = dz4 @ self.W4.T
        dz3 = da3 * self._relu_deriv(self.z3)
        dW3 = self.a2.T @ dz3
        db3 = dz3.sum(axis=0)
        da2 = dz3 @ self.W3.T
        dz2 = da2 * self._relu_deriv(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._relu_deriv(self.z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        self.W4 -= self.lr * dW4;  self.b4 -= self.lr * db4
        self.W3 -= self.lr * dW3;  self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1
        return loss

    def reconstruction_error(self, X):
        output = self.forward(X)
        return np.mean((output - X) ** 2, axis=1)


# ---------------------------------------------------------------------------
# Train and score helper
# ---------------------------------------------------------------------------
def train_and_score(gaf_matrix, train_mask, input_dim, hidden1, hidden2,
                    lr=0.0005, n_epochs=200, batch_size=64, label=""):
    """Train autoencoder on training subset, return reconstruction errors for all days."""
    train_data = gaf_matrix[train_mask]
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    all_scaled = scaler.transform(gaf_matrix)

    np.random.seed(42)
    ae = SimpleAutoencoder(input_dim=input_dim, hidden1=hidden1,
                           hidden2=hidden2, lr=lr)
    bs = min(batch_size, len(train_scaled))

    for epoch in range(n_epochs):
        indices = np.random.permutation(len(train_scaled))
        epoch_loss = 0
        n_batches = 0
        for start in range(0, len(train_scaled), bs):
            batch = train_scaled[indices[start:start + bs]]
            loss = ae.train_step(batch)
            epoch_loss += loss
            n_batches += 1
        if (epoch + 1) % 50 == 0:
            avg = epoch_loss / n_batches
            print(f"   [{label}] Epoch {epoch+1}/{n_epochs} -- Loss: {avg:.6f}")

    errors = ae.reconstruction_error(all_scaled)
    return errors


# ---------------------------------------------------------------------------
# Failure evaluation (same logic as anomaly_detection_merged.py)
# ---------------------------------------------------------------------------
def evaluate_against_failures(dates, scores, threshold_pct, failures_df, label=""):
    """Evaluate anomaly scores against known failures."""
    threshold = np.percentile(scores, 100 - threshold_pct)
    is_anomaly = scores >= threshold

    date_score = pd.DataFrame({"Date": dates, "score": scores, "anomaly": is_anomaly})
    date_score["Date"] = pd.to_datetime(date_score["Date"])

    detected = 0
    for _, f in failures_df.iterrows():
        f_date = f["start"].normalize()
        mask = (
            (date_score["Date"] >= f_date - pd.Timedelta(days=1))
            & (date_score["Date"] <= f_date + pd.Timedelta(days=1))
        )
        nearby = date_score[mask]
        if nearby["anomaly"].any():
            detected += 1

    total_anomalies = int(is_anomaly.sum())
    total_failures = len(failures_df)
    det_rate = 100 * detected / total_failures if total_failures > 0 else 0

    print(f"\n  [{label}] top {threshold_pct}% flagged")
    print(f"  Failures detected: {detected}/{total_failures} ({det_rate:.1f}%)")
    print(f"  Total anomaly flags: {total_anomalies}")

    return detected, total_anomalies, det_rate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" Module 5-M -- GAF Autoencoder with Merged Data")
    print("=" * 70)

    # ====================================================================
    # 1. Load Feza's preprocessed data
    # ====================================================================
    master_path = os.path.join(DATA_DIR, "master_imputed.csv")
    if not os.path.exists(master_path):
        print(f"ERROR: {master_path} not found.")
        print("Run module5_gaf.py first to generate master_imputed.csv")
        sys.exit(1)

    master = pd.read_csv(master_path, parse_dates=["Date"])
    print(f"\nLoaded master_imputed.csv: {len(master)} rows")
    print(f"Date range: {master['Date'].min().date()} to {master['Date'].max().date()}")

    # Load missing data summary for quality flags
    missing_path = os.path.join(DATA_DIR, "missing_data_summary.csv")
    if os.path.exists(missing_path):
        missing_summary = pd.read_csv(missing_path, parse_dates=["Date"])
        dq_low_dates = set(missing_summary[
            missing_summary["data_quality_flag"] == "data_quality_low"]["Date"])
        partial_dates = set(missing_summary[
            missing_summary["data_quality_flag"] == "partial_missing"]["Date"])
        n_present_lookup = dict(zip(missing_summary["Date"],
                                    missing_summary["n_present"]))
    else:
        print("WARNING: missing_data_summary.csv not found, skipping quality flags")
        dq_low_dates = set()
        partial_dates = set()
        n_present_lookup = {}

    # Load daily features for AllDayZero flag
    daily_path = os.path.join(DATA_DIR, "daily_features.csv")
    if os.path.exists(daily_path):
        daily = pd.read_csv(daily_path, parse_dates=["Date"])
        zero_days = set(daily[daily["AllDayZero"]]["Date"].values)
    else:
        print("WARNING: daily_features.csv not found")
        zero_days = set()

    # ====================================================================
    # 2. Load and merge akim-gerilim data
    # ====================================================================
    print(f"\nLoading akim-gerilim data from: {AKIM_FILE}")
    if not os.path.exists(AKIM_FILE):
        print(f"ERROR: {AKIM_FILE} not found.")
        print("Make sure the raporlar26 folder contains akim-gerilim_raporu_Vkx3H.xlsx")
        sys.exit(1)

    akim = pd.read_excel(AKIM_FILE)
    akim_ts_col = akim.columns[0]
    akim[akim_ts_col] = pd.to_datetime(akim[akim_ts_col])
    akim["timestamp"] = akim[akim_ts_col].dt.floor("h")
    akim_hourly = akim.drop(columns=[akim_ts_col]).groupby("timestamp").mean().reset_index()

    # Rename columns (same logic as anomaly_detection_merged.py)
    col_map = {}
    for c in akim_hourly.columns:
        if c == "timestamp":
            continue
        col_map[c] = (
            c.replace("Ak\u0131m", "current")
            .replace("Gerilim", "voltage")
            .replace("Cosf", "pf")
            .replace("Frekans", "freq")
            .replace("N\u00f6tr", "neutral")
            .replace(" ", "_")
        )
    akim_hourly.rename(columns=col_map, inplace=True)

    # Drop mostly-NaN columns (neutral current, frequency are >99% NaN)
    akim_cols_keep = [c for c in akim_hourly.columns
                      if c != "timestamp" and akim_hourly[c].isna().mean() < 0.5]
    akim_hourly = akim_hourly[["timestamp"] + akim_cols_keep]

    print(f"Akim-gerilim: {len(akim_hourly)} hourly rows, columns: {akim_cols_keep}")

    # Create timestamp in master for merging
    master["timestamp"] = pd.to_datetime(
        master["Date"].dt.strftime("%Y-%m-%d") + " " +
        master["Hour"].astype(str).str.zfill(2) + ":00:00"
    )

    # Merge
    merged = pd.merge(master, akim_hourly, on="timestamp", how="inner")
    merged = merged.sort_values(["Date", "Hour"]).reset_index(drop=True)
    print(f"Merged dataset: {len(merged)} rows ({merged['Date'].nunique()} days)")

    # Compute derived electrical features
    if "R_current" in merged.columns:
        merged["avg_current"] = merged[["R_current", "S_current", "T_current"]].mean(axis=1)
        merged["avg_voltage"] = merged[["R_voltage", "S_voltage", "T_voltage"]].mean(axis=1)
        merged["current_imbalance"] = merged[["R_current", "S_current", "T_current"]].std(axis=1)
    else:
        print("ERROR: Expected R_current, S_current, T_current columns not found")
        print(f"Available columns: {list(merged.columns)}")
        sys.exit(1)

    # Fill NaN in electrical columns
    for col in ["avg_current", "avg_voltage", "current_imbalance"]:
        merged[col] = merged[col].ffill().fillna(0)

    # ====================================================================
    # 3. Load failure ground truth
    # ====================================================================
    failures = None
    if os.path.exists(FAILURE_FILE):
        failures_raw = pd.read_excel(FAILURE_FILE)
        failures = pd.DataFrame()
        failures["start"] = pd.to_datetime(failures_raw.iloc[:, 8], dayfirst=True)
        failures["end"] = pd.to_datetime(failures_raw.iloc[:, 9], dayfirst=True)
        failures["duration_sec"] = failures_raw.iloc[:, 10]
        failures = failures.dropna(subset=["start"])
        print(f"Loaded {len(failures)} failure events for evaluation")
    else:
        print(f"WARNING: {FAILURE_FILE} not found, skipping failure evaluation")

    # ====================================================================
    # 4. Build 24h profiles for each channel
    # ====================================================================
    print("\nBuilding 24h multi-channel profiles...")

    energy_profiles = {}
    current_profiles = {}
    voltage_profiles = {}
    imbalance_profiles = {}
    masks = {}

    for dt, grp in merged.groupby("Date"):
        if dt in dq_low_dates:
            continue
        grp_sorted = grp.sort_values("Hour")
        if len(grp_sorted) != 24:
            continue

        energy_vals = grp_sorted["Consumption_kWh"].values
        current_vals = grp_sorted["avg_current"].values
        voltage_vals = grp_sorted["avg_voltage"].values
        imb_vals = grp_sorted["current_imbalance"].values
        is_missing = grp_sorted["is_missing"].values

        if np.any(np.isnan(energy_vals)):
            continue

        energy_profiles[dt] = energy_vals
        current_profiles[dt] = current_vals
        voltage_profiles[dt] = voltage_vals
        imbalance_profiles[dt] = imb_vals
        masks[dt] = (1 - is_missing).astype(float)

    dates = sorted(energy_profiles.keys())
    print(f"{len(dates)} days with complete 24h merged profiles")

    if len(dates) == 0:
        print("ERROR: No valid days found. Check data alignment.")
        sys.exit(1)

    # ====================================================================
    # 5. Generate GAF images
    # ====================================================================
    print("\nGenerating GAF images...")

    # Model A: Energy-only GAF (576-dim, same as Module 5)
    gaf_energy = np.array([timeseries_to_gaf(energy_profiles[dt]).flatten()
                           for dt in dates])
    print(f"  Model A (energy-only): {gaf_energy.shape}")

    # Model B: Multi-channel GAF (4 channels x 576 = 2304-dim)
    gaf_merged_list = []
    for dt in dates:
        g_energy = timeseries_to_gaf(energy_profiles[dt]).flatten()
        g_current = timeseries_to_gaf(current_profiles[dt]).flatten()
        g_voltage = timeseries_to_gaf(voltage_profiles[dt]).flatten()
        g_imbalance = timeseries_to_gaf(imbalance_profiles[dt]).flatten()
        gaf_merged_list.append(np.concatenate([g_energy, g_current,
                                               g_voltage, g_imbalance]))
    gaf_merged = np.array(gaf_merged_list)
    print(f"  Model B (4-channel merged): {gaf_merged.shape}")

    # Training mask: complete, non-zero, non-partial days
    train_mask = np.array([
        dt not in zero_days and dt not in partial_dates
        for dt in dates
    ])
    print(f"  Training days: {train_mask.sum()} (excluding zero/partial/low-quality)")

    # ====================================================================
    # 6. Train autoencoders and score
    # ====================================================================
    print("\n" + "=" * 70)
    print("MODEL A: Energy-Only GAF (baseline, replicates Module 5)")
    print("=" * 70)
    errors_A = train_and_score(
        gaf_energy, train_mask,
        input_dim=576, hidden1=64, hidden2=16,
        lr=0.0005, n_epochs=200, label="Model A"
    )

    print("\n" + "=" * 70)
    print("MODEL B: Multi-Channel GAF (Energy + Current + Voltage + Imbalance)")
    print("=" * 70)
    errors_B = train_and_score(
        gaf_merged, train_mask,
        input_dim=2304, hidden1=128, hidden2=32,
        lr=0.0003, n_epochs=300, label="Model B"
    )

    # ====================================================================
    # 7. Mask-aware scoring
    # ====================================================================
    print("\nComputing mask-aware scores...")
    results = []
    for i, dt in enumerate(dates):
        n_present = n_present_lookup.get(dt, 24)
        presence_ratio = n_present / 24.0

        raw_A = errors_A[i]
        raw_B = errors_B[i]
        masked_A = raw_A * presence_ratio
        masked_B = raw_B * presence_ratio

        results.append({
            "Date": dt,
            "ReconError_A_Raw": raw_A,
            "ReconError_B_Raw": raw_B,
            "n_present": n_present,
            "presence_ratio": round(presence_ratio, 4),
            "ReconError_A_MaskAware": masked_A,
            "ReconError_B_MaskAware": masked_B,
        })

    results_df = pd.DataFrame(results)

    # Normalize to 0-1
    for col in ["ReconError_A_MaskAware", "ReconError_B_MaskAware"]:
        mn, mx = results_df[col].min(), results_df[col].max()
        results_df[col.replace("ReconError", "Score")] = \
            (results_df[col] - mn) / (mx - mn + 1e-9)

    results_df = results_df.sort_values("Score_B_MaskAware", ascending=False)

    # ====================================================================
    # 8. Save results
    # ====================================================================
    out_scores = os.path.join(DATA_DIR, "gaf_merged_scores.csv")
    results_df.to_csv(out_scores, index=False)

    top20 = results_df.head(20)
    out_top20 = os.path.join(DATA_DIR, "gaf_merged_top20.csv")
    top20.to_csv(out_top20, index=False)

    print(f"\nTop 10 anomalous days (Model B - merged GAF):")
    print(top20[["Date", "Score_B_MaskAware", "Score_A_MaskAware",
                 "n_present"]].head(10).to_string(index=False))

    # ====================================================================
    # 9. Evaluate against failures (if available)
    # ====================================================================
    if failures is not None and len(failures) > 0:
        print("\n" + "=" * 70)
        print("FAILURE DETECTION COMPARISON")
        print("=" * 70)

        dates_arr = np.array(dates)
        scores_A = results_df.set_index("Date").loc[dates, "Score_A_MaskAware"].values
        scores_B = results_df.set_index("Date").loc[dates, "Score_B_MaskAware"].values

        comparison_rows = []
        for pct in [5, 10, 15, 20]:
            det_A, flags_A, rate_A = evaluate_against_failures(
                dates_arr, scores_A, pct, failures, f"Model A top-{pct}%")
            det_B, flags_B, rate_B = evaluate_against_failures(
                dates_arr, scores_B, pct, failures, f"Model B top-{pct}%")
            comparison_rows.append({
                "Top_Pct": pct,
                "Model_A_Detected": det_A,
                "Model_A_Rate": round(rate_A, 1),
                "Model_A_Flags": flags_A,
                "Model_B_Detected": det_B,
                "Model_B_Rate": round(rate_B, 1),
                "Model_B_Flags": flags_B,
            })

        comp_df = pd.DataFrame(comparison_rows)
        out_comp = os.path.join(DATA_DIR, "gaf_merged_comparison.csv")
        comp_df.to_csv(out_comp, index=False)

        print(f"\n  Comparison summary:")
        print(comp_df.to_string(index=False))
        print(f"\n  Saved: {out_comp}")

    # ====================================================================
    # 10. Per-failure detail (which model catches which)
    # ====================================================================
    if failures is not None and len(failures) > 0:
        print("\n" + "=" * 70)
        print("PER-FAILURE DETECTION (top 10% threshold)")
        print("=" * 70)

        threshold_A = np.percentile(scores_A, 90)
        threshold_B = np.percentile(scores_B, 90)

        date_df = pd.DataFrame({
            "Date": pd.to_datetime(dates),
            "anom_A": scores_A >= threshold_A,
            "anom_B": scores_B >= threshold_B,
        })

        for _, f in failures.iterrows():
            f_date = f["start"].normalize()
            mask = (
                (date_df["Date"] >= f_date - pd.Timedelta(days=1))
                & (date_df["Date"] <= f_date + pd.Timedelta(days=1))
            )
            nearby = date_df[mask]
            det_A = "A" if nearby["anom_A"].any() else "."
            det_B = "B" if nearby["anom_B"].any() else "."
            dur = f["duration_sec"]
            print(f"  {f['start'].date()}  dur={dur:>6.0f}s  [{det_A}{det_B}]")

        print("\n  Legend: A=energy-only GAF, B=merged GAF, .=not detected")

    print(f"\nSaved: {out_scores}")
    print(f"Saved: {out_top20}")
    print("=" * 70)
    print(" Module 5-M complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
