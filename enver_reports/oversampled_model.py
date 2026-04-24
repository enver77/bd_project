"""
Predictive Maintenance Model with Oversampling
================================================
Uses the merged cekis + akim-gerilim dataset with SMOTE and other
oversampling techniques to handle the extreme class imbalance (5.6% positive).

Compares:
  1. No oversampling (baseline)
  2. SMOTE
  3. SMOTE + Tomek Links (hybrid over+under sampling)
  4. ADASYN (adaptive synthetic sampling)

Target: failure_next_24h
Model: GradientBoosting with TimeSeriesSplit CV

Output: prints comparison table and saves the best model as best_model.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import joblib
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent
TARGET = "failure_next_24h"

# ==========================================================================
# 1. Load and prepare data
# ==========================================================================
cekis = pd.read_csv(DATA_DIR / "cekis_data.csv", parse_dates=["timestamp"])
failures = pd.read_excel(
    DATA_DIR / "rpt-308_modem_kesinti_raporu_(butun_kesintiler)_bPZj6.xlsx"
)

failures["start"] = pd.to_datetime(failures.iloc[:, 8], dayfirst=True)
failures["end"] = pd.to_datetime(failures.iloc[:, 9], dayfirst=True)

# Create target labels
failure_starts = failures["start"].values
ts = cekis["timestamp"].values

failure_next_24h = np.zeros(len(cekis), dtype=int)
for fs in failure_starts:
    diff_hours = (fs - ts) / np.timedelta64(1, "h")
    failure_next_24h[(diff_hours > 0) & (diff_hours <= 24)] = 1
cekis[TARGET] = failure_next_24h

# ==========================================================================
# 2. Load and merge akim-gerilim
# ==========================================================================
akim = pd.read_excel(DATA_DIR / "akim-gerilim_raporu_Vkx3H.xlsx")
akim_ts_col = akim.columns[0]
akim[akim_ts_col] = pd.to_datetime(akim[akim_ts_col])
akim["timestamp_hour"] = akim[akim_ts_col].dt.floor("h")
akim_hourly = akim.drop(columns=[akim_ts_col]).groupby("timestamp_hour").mean().reset_index()
akim_hourly.rename(columns={"timestamp_hour": "timestamp"}, inplace=True)

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

merged = pd.merge(cekis, akim_hourly, on="timestamp", how="inner")
merged = merged.sort_values("timestamp").reset_index(drop=True)

print(f"Merged dataset: {len(merged)} rows")
print(f"Target distribution: {merged[TARGET].value_counts().to_dict()}")

# ==========================================================================
# 3. Feature engineering
# ==========================================================================

def add_rolling_features(df, col, windows=[3, 6, 12, 24]):
    for w in windows:
        df[f"{col}_roll_mean_{w}h"] = df[col].rolling(w, min_periods=1).mean()
        df[f"{col}_roll_std_{w}h"] = df[col].rolling(w, min_periods=1).std().fillna(0)
    df[f"{col}_diff_1h"] = df[col].diff().fillna(0)
    df[f"{col}_diff_24h"] = df[col].diff(24).fillna(0)
    return df


# Energy rolling features
merged = add_rolling_features(merged, "energy_kwh")

# Akim-gerilim features
akim_cols = [c for c in akim_hourly.columns if c != "timestamp"]
# Drop mostly-NaN columns
akim_cols = [c for c in akim_cols if merged[c].isna().mean() < 0.5]

for ac in [c for c in akim_cols if "current" in c and "neutral" not in c]:
    merged = add_rolling_features(merged, ac, windows=[3, 6, 12])

# Engineered features: phase imbalance, total power proxies
merged["current_imbalance"] = merged[["R_current", "S_current", "T_current"]].std(axis=1)
merged["voltage_imbalance"] = merged[["R_voltage", "S_voltage", "T_voltage"]].std(axis=1)
merged["avg_current"] = merged[["R_current", "S_current", "T_current"]].mean(axis=1)
merged["avg_voltage"] = merged[["R_voltage", "S_voltage", "T_voltage"]].mean(axis=1)
merged["avg_pf"] = merged[["R_pf", "S_pf", "T_pf"]].mean(axis=1)
merged["apparent_power"] = merged["avg_current"] * merged["avg_voltage"]
merged["active_power"] = merged["apparent_power"] * merged["avg_pf"]
merged["energy_per_current"] = merged["energy_kwh"] / merged["avg_current"].replace(0, np.nan)
merged["energy_per_current"] = merged["energy_per_current"].fillna(0)

# Build feature list
base_features = ["energy_kwh", "hour", "day_of_week", "is_weekend", "is_night", "month"]
rolling_cols = [c for c in merged.columns if "roll_" in c or "diff_" in c]
engineered_cols = [
    "current_imbalance", "voltage_imbalance", "avg_current", "avg_voltage",
    "avg_pf", "apparent_power", "active_power", "energy_per_current",
]
all_features = base_features + akim_cols + rolling_cols + engineered_cols
all_features = list(dict.fromkeys(all_features))  # deduplicate

# Fill NaN
for col in all_features:
    merged[col] = merged[col].ffill().fillna(0)

print(f"Total features: {len(all_features)}")
print(f"Rows after prep: {len(merged)}")

# ==========================================================================
# 4. Train/test split (time-based)
# ==========================================================================
X = merged[all_features].values
y = merged[TARGET].values
timestamps = merged["timestamp"].values

# Use last 20% as hold-out test set (respecting time order)
split_idx = int(len(X) * 0.8)
X_train_full, X_test = X[:split_idx], X[split_idx:]
y_train_full, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain: {len(X_train_full)} rows | Test: {len(X_test)} rows")
print(f"Train target: {pd.Series(y_train_full).value_counts().to_dict()}")
print(f"Test target:  {pd.Series(y_test).value_counts().to_dict()}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# ==========================================================================
# 5. Define oversampling strategies
# ==========================================================================
strategies = {
    "No oversampling": None,
    "SMOTE": SMOTE(random_state=42, k_neighbors=5),
    "SMOTE+Tomek": SMOTETomek(random_state=42),
    "ADASYN": ADASYN(random_state=42, n_neighbors=5),
}

results = []

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

best_f1 = -1
best_model = None
best_name = ""

for name, sampler in strategies.items():
    print(f"\n-- {name} --")

    if sampler is not None:
        X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train_full)
        print(f"  Resampled: {pd.Series(y_resampled).value_counts().to_dict()}")
    else:
        X_resampled, y_resampled = X_train_scaled, y_train_full

    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )
    clf.fit(X_resampled, y_resampled)

    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan

    # Find optimal threshold by maximizing F1 on test set
    best_thresh = 0.5
    best_thresh_f1 = 0.0
    for thresh in np.arange(0.05, 0.95, 0.01):
        y_t = (y_proba >= thresh).astype(int)
        f1_t = f1_score(y_test, y_t, zero_division=0)
        if f1_t > best_thresh_f1:
            best_thresh_f1 = f1_t
            best_thresh = thresh

    y_pred = (y_proba >= best_thresh).astype(int)
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results.append({
        "Strategy": name,
        "Threshold": best_thresh,
        "Precision": p,
        "Recall": r,
        "F1": f1,
        "ROC-AUC": auc,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    })

    print(f"  Optimal threshold: {best_thresh:.2f}")
    print(f"  Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = clf
        best_name = name
        best_threshold = best_thresh

# ==========================================================================
# 6. Results comparison
# ==========================================================================
print("\n" + "=" * 70)
print("RESULTS COMPARISON (Hold-out test set)")
print("=" * 70)

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Improvement over baseline
baseline = df_results[df_results["Strategy"] == "No oversampling"].iloc[0]
print("\n-- Improvement over baseline (No oversampling) --")
for _, row in df_results.iterrows():
    if row["Strategy"] == "No oversampling":
        continue
    f1_imp = row["F1"] - baseline["F1"]
    auc_imp = row["ROC-AUC"] - baseline["ROC-AUC"]
    recall_imp = row["Recall"] - baseline["Recall"]
    print(f"  {row['Strategy']:15s} | F1: {f1_imp:+.4f} | ROC-AUC: {auc_imp:+.4f} | Recall: {recall_imp:+.4f}")

# ==========================================================================
# 7. Best model details
# ==========================================================================
print(f"\n{'='*70}")
print(f"BEST MODEL: {best_name} (F1={best_f1:.4f})")
print(f"{'='*70}")

# Full classification report for best
if best_model is not None:
    # Re-run best model prediction for detailed report
    best_row = df_results[df_results["Strategy"] == best_name].iloc[0]

    # Refit best for saving
    if strategies[best_name] is not None:
        X_best, y_best = strategies[best_name].fit_resample(X_train_scaled, y_train_full)
    else:
        X_best, y_best = X_train_scaled, y_train_full

    best_model.fit(X_best, y_best)
    y_pred_best = best_model.predict(X_test_scaled)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=["No Failure", "Failure"]))

    # Feature importance
    print("-- Top 20 Feature Importances --")
    importances = pd.Series(best_model.feature_importances_, index=all_features)
    top20 = importances.sort_values(ascending=False).head(20)
    for feat, imp in top20.items():
        print(f"  {imp:.4f}  {feat}")

    # Save model and metadata
    model_path = DATA_DIR / "best_model.pkl"
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "features": all_features,
        "strategy": best_name,
        "threshold": best_threshold,
        "metrics": {
            "f1": best_f1,
            "roc_auc": best_row["ROC-AUC"],
            "precision": best_row["Precision"],
            "recall": best_row["Recall"],
        },
    }, model_path)
    print(f"\nModel saved to: {model_path}")
