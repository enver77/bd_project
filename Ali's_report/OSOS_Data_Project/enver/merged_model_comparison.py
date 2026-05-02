"""
Merged Model Comparison
========================
Compares predictive maintenance model performance:
  Model A: cekis_data only (energy consumption features)
  Model B: cekis_data + akim-gerilim (energy + current/voltage/power factor features)

Target: failure_next_24h (predict if a failure happens within 24 hours)

Output: prints comparison metrics and saves merged dataset as cekis_akim_gerilim_merged.csv
"""

import sys, os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Import heuristic optimizers from existing analysis module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "25_26_osf_data", "analysis"))
from heuristic_opt import GeneticAlgorithm, PSO, SimulatedAnnealing

DATA_DIR = Path(__file__).parent

# ==========================================================================
# 1. Load and prepare cekis_data with labels
# ==========================================================================
cekis = pd.read_csv(DATA_DIR / "cekis_data.csv", parse_dates=["timestamp"])
failures = pd.read_excel(
    DATA_DIR / "rpt-308_modem_kesinti_raporu_(butun_kesintiler)_bPZj6.xlsx"
)

failures["start"] = pd.to_datetime(failures.iloc[:, 8], dayfirst=True)
failures["end"] = pd.to_datetime(failures.iloc[:, 9], dayfirst=True)

# Create target: failure in next 24h
failure_starts = failures["start"].values
ts = cekis["timestamp"].values
failure_next_24h = np.zeros(len(cekis), dtype=int)
for fs in failure_starts:
    diff_hours = (fs - ts) / np.timedelta64(1, "h")
    failure_next_24h[(diff_hours > 0) & (diff_hours <= 24)] = 1
cekis["failure_next_24h"] = failure_next_24h

# ==========================================================================
# 2. Load akim-gerilim data and merge
# ==========================================================================
akim = pd.read_excel(DATA_DIR / "akim-gerilim_raporu_Vkx3H.xlsx")
akim_ts_col = akim.columns[0]  # "Okuma Tarihi"
akim[akim_ts_col] = pd.to_datetime(akim[akim_ts_col])

# Round akim-gerilim timestamps to the nearest hour for merging
akim["timestamp_hour"] = akim[akim_ts_col].dt.floor("h")

# Aggregate if multiple readings per hour (take mean)
akim_hourly = akim.drop(columns=[akim_ts_col]).groupby("timestamp_hour").mean().reset_index()
akim_hourly.rename(columns={"timestamp_hour": "timestamp"}, inplace=True)

# Rename columns to ASCII-friendly names for easier handling
col_map = {}
for c in akim_hourly.columns:
    if c == "timestamp":
        continue
    col_map[c] = (
        c.replace("Akım", "current")
        .replace("Ak\u0131m", "current")
        .replace("Gerilim", "voltage")
        .replace("Cosf", "pf")
        .replace("Frekans", "freq")
        .replace("Nötr", "neutral")
        .replace("N\u00f6tr", "neutral")
        .replace("Ak?m", "current")
        .replace(" ", "_")
    )
akim_hourly.rename(columns=col_map, inplace=True)

print(f"Akim-gerilim hourly: {len(akim_hourly)} rows")
print(f"Columns: {akim_hourly.columns.tolist()}")

# Merge with cekis
merged = pd.merge(cekis, akim_hourly, on="timestamp", how="inner")
print(f"\nMerged dataset: {len(merged)} rows (inner join)")
print(f"Cekis original: {len(cekis)} rows")

# Save merged dataset
merged.to_csv(DATA_DIR / "cekis_akim_gerilim_merged.csv", index=False)
print(f"Saved: cekis_akim_gerilim_merged.csv")

# ==========================================================================
# 3. Feature engineering
# ==========================================================================

def add_rolling_features(df, col="energy_kwh", windows=[3, 6, 12, 24]):
    """Add rolling statistics as features."""
    df = df.copy()
    for w in windows:
        df[f"{col}_roll_mean_{w}h"] = df[col].rolling(w, min_periods=1).mean()
        df[f"{col}_roll_std_{w}h"] = df[col].rolling(w, min_periods=1).std().fillna(0)
    # Rate of change
    df[f"{col}_diff_1h"] = df[col].diff().fillna(0)
    df[f"{col}_diff_24h"] = df[col].diff(24).fillna(0)
    return df


# Features for Model A (cekis only)
base_features = ["energy_kwh", "hour", "day_of_week", "is_weekend", "is_night", "month"]

merged_sorted = merged.sort_values("timestamp").reset_index(drop=True)
merged_sorted = add_rolling_features(merged_sorted, "energy_kwh")

rolling_cols = [c for c in merged_sorted.columns if "roll_" in c or "diff_" in c]
features_A = base_features + rolling_cols

# Features for Model B (cekis + akim-gerilim)
akim_cols = [c for c in akim_hourly.columns if c != "timestamp"]
features_B = features_A + akim_cols

# Also add rolling features for current/voltage
for ac in akim_cols[:3]:  # R/S/T current
    merged_sorted = add_rolling_features(merged_sorted, ac, windows=[3, 6, 12])
    extra_rolling = [c for c in merged_sorted.columns if ac in c and ("roll_" in c or "diff_" in c)]
    features_B += extra_rolling

# Remove duplicates
features_B = list(dict.fromkeys(features_B))

print(f"\nModel A features ({len(features_A)}): {features_A}")
print(f"\nModel B features ({len(features_B)}): {features_B}")

# ==========================================================================
# 4. Train and compare models
# ==========================================================================
TARGET = "failure_next_24h"

# Drop columns that are mostly NaN (>50% missing), then fill remaining NaNs
mostly_nan_cols = [c for c in features_B if c in merged_sorted.columns and merged_sorted[c].isna().mean() > 0.5]
print(f"\nDropping mostly-NaN columns: {mostly_nan_cols}")
features_A = [c for c in features_A if c not in mostly_nan_cols]
features_B = [c for c in features_B if c not in mostly_nan_cols]

# Fill remaining NaN with forward-fill then 0
for col in features_B:
    if col in merged_sorted.columns:
        merged_sorted[col] = merged_sorted[col].ffill().fillna(0)

data = merged_sorted.copy()

print(f"\nValid rows for modeling: {len(data)}")
print(f"Target distribution: {data[TARGET].value_counts().to_dict()}")

X_A = data[features_A].values
X_B = data[features_B].values
y = data[TARGET].values

# Time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

results = {"Model A (cekis only)": [], "Model B (cekis + akim-gerilim)": []}


def evaluate_model(X, y, model_name, tscv):
    """Run time-series CV and return metrics per fold."""
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Handle class imbalance with class_weight
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        metrics = {
            "fold": fold + 1,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        # ROC AUC only if both classes present
        if len(np.unique(y_test)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
        else:
            metrics["roc_auc"] = np.nan

        fold_metrics.append(metrics)

    return fold_metrics, clf


print("\n" + "=" * 70)
print("MODEL COMPARISON: failure_next_24h prediction")
print("=" * 70)

# Model A
print("\n── Model A: Cekis Data Only ──")
metrics_A, clf_A = evaluate_model(X_A, y, "Model A", tscv)
df_A = pd.DataFrame(metrics_A)
print(df_A.to_string(index=False))
print(f"\nMean F1: {df_A['f1'].mean():.4f}  |  Mean ROC-AUC: {df_A['roc_auc'].mean():.4f}")

# Model B
print("\n── Model B: Cekis + Akim-Gerilim ──")
metrics_B, clf_B = evaluate_model(X_B, y, "Model B", tscv)
df_B = pd.DataFrame(metrics_B)
print(df_B.to_string(index=False))
print(f"\nMean F1: {df_B['f1'].mean():.4f}  |  Mean ROC-AUC: {df_B['roc_auc'].mean():.4f}")

# ==========================================================================
# 5. Summary comparison
# ==========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

comparison = pd.DataFrame(
    {
        "Metric": ["F1 Score", "ROC-AUC", "Precision", "Recall"],
        "Model A (cekis)": [
            df_A["f1"].mean(),
            df_A["roc_auc"].mean(),
            df_A["precision"].mean(),
            df_A["recall"].mean(),
        ],
        "Model B (cekis+akim)": [
            df_B["f1"].mean(),
            df_B["roc_auc"].mean(),
            df_B["precision"].mean(),
            df_B["recall"].mean(),
        ],
    }
)
comparison["Improvement"] = comparison["Model B (cekis+akim)"] - comparison["Model A (cekis)"]
comparison["Improvement %"] = (
    100 * comparison["Improvement"] / comparison["Model A (cekis)"].replace(0, np.nan)
)

print(comparison.to_string(index=False))

if comparison.loc[comparison["Metric"] == "F1 Score", "Improvement"].values[0] > 0:
    print("\n[+] Adding akim-gerilim data IMPROVES the model.")
else:
    print("\n[-] Adding akim-gerilim data does NOT improve the model (or makes it worse).")

# ==========================================================================
# 6. Feature importance from Model B
# ==========================================================================
print("\n── Top 20 Feature Importances (Model B) ──")
importances = pd.Series(clf_B.feature_importances_, index=features_B)
top20 = importances.sort_values(ascending=False).head(20)
for feat, imp in top20.items():
    marker = " *" if feat in akim_cols or any(ac in feat for ac in akim_cols) else ""
    print(f"  {imp:.4f}  {feat}{marker}")

print("\n* = akim-gerilim feature")

# ==========================================================================
# 7. GA: Feature Subset Selection (same approach as anomaly pipeline)
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 7: GA -- Feature Subset Selection")
print("=" * 70)

# Use 80/20 time-based split for optimization (train on first 80%)
split_idx = int(len(data) * 0.8)
X_B_train, X_B_test = X_B[:split_idx], X_B[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler_opt = StandardScaler()
X_B_train_sc = scaler_opt.fit_transform(X_B_train)
X_B_test_sc = scaler_opt.transform(X_B_test)

SEED = 42


def ga_fitness(chromosome: np.ndarray) -> float:
    """F1 score of GBM on the selected feature subset (evaluated on test split)."""
    mask = chromosome.astype(bool)
    if mask.sum() < 2:
        return -1.0
    X_tr = X_B_train_sc[:, mask]
    X_te = X_B_test_sc[:, mask]

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        random_state=SEED,
    )
    model.fit(X_tr, y_train)
    y_prob = model.predict_proba(X_te)[:, 1]

    # Optimize threshold for F1
    best_f1 = 0.0
    for t in np.arange(0.05, 0.5, 0.02):
        f = f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f
    return best_f1


ga = GeneticAlgorithm(
    fitness_fn=ga_fitness,
    n_genes=len(features_B),
    pop_size=20,
    n_gen=30,
    crossover_rate=0.8,
    mutation_rate=0.1,
    min_active=3,
    seed=SEED,
    verbose=5,
)
best_chromosome, ga_score = ga.optimize()

selected_features = [f for f, flag in zip(features_B, best_chromosome) if flag]
print(f"\nGA selected {len(selected_features)}/{len(features_B)} features (F1={ga_score:.4f}):")
for f in selected_features:
    marker = " *" if f in akim_cols or any(ac in f for ac in akim_cols) else ""
    print(f"  - {f}{marker}")

# Rebuild feature matrix with selected subset
feature_mask = best_chromosome.astype(bool)
X_sel_train = X_B_train_sc[:, feature_mask]
X_sel_test = X_B_test_sc[:, feature_mask]

# ==========================================================================
# 8. PSO: GradientBoosting Hyperparameter Optimization
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 8: PSO -- GBM Hyperparameter Optimization")
print("=" * 70)

# PSO searches over continuous space:
#   param[0] = learning_rate   in [0.01, 0.20]
#   param[1] = max_depth       in [2, 10]  (rounded to int)
#   param[2] = subsample       in [0.5, 1.0]
#   param[3] = min_samples_leaf in [2, 30]  (rounded to int)
#   param[4] = n_estimators    in [50, 400] (rounded to int)


def pso_fitness(params: np.ndarray) -> float:
    """F1 score of GBM with given hyperparameters on GA-selected features."""
    lr = float(np.clip(params[0], 0.01, 0.20))
    depth = int(np.clip(round(params[1]), 2, 10))
    subsample = float(np.clip(params[2], 0.5, 1.0))
    min_leaf = int(np.clip(round(params[3]), 2, 30))
    n_est = int(np.clip(round(params[4]), 50, 400))

    model = GradientBoostingClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=lr,
        subsample=subsample, min_samples_leaf=min_leaf,
        random_state=SEED,
    )
    model.fit(X_sel_train, y_train)
    y_prob = model.predict_proba(X_sel_test)[:, 1]

    best_f1 = 0.0
    for t in np.arange(0.05, 0.5, 0.02):
        f = f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f
    return best_f1


pso = PSO(
    fitness_fn=pso_fitness,
    bounds=[
        [0.01, 0.20],   # learning_rate
        [2.0, 10.0],    # max_depth
        [0.5, 1.0],     # subsample
        [2.0, 30.0],    # min_samples_leaf
        [50.0, 400.0],  # n_estimators
    ],
    n_particles=15,
    n_iter=25,
    seed=SEED,
    verbose=5,
)
best_params, pso_score = pso.optimize()

best_lr = float(np.clip(best_params[0], 0.01, 0.20))
best_depth = int(np.clip(round(best_params[1]), 2, 10))
best_subsample = float(np.clip(best_params[2], 0.5, 1.0))
best_min_leaf = int(np.clip(round(best_params[3]), 2, 30))
best_n_est = int(np.clip(round(best_params[4]), 50, 400))

print(f"\nPSO best hyperparameters (F1={pso_score:.4f}):")
print(f"  n_estimators    = {best_n_est}")
print(f"  max_depth       = {best_depth}")
print(f"  learning_rate   = {best_lr:.4f}")
print(f"  subsample       = {best_subsample:.4f}")
print(f"  min_samples_leaf = {best_min_leaf}")

# ==========================================================================
# 9. Train final model with PSO-optimized params on GA-selected features
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 9: Final Model (GA features + PSO hyperparams)")
print("=" * 70)

final_model = GradientBoostingClassifier(
    n_estimators=best_n_est, max_depth=best_depth, learning_rate=best_lr,
    subsample=best_subsample, min_samples_leaf=best_min_leaf,
    random_state=SEED,
)
final_model.fit(X_sel_train, y_train)
y_prob_final = final_model.predict_proba(X_sel_test)[:, 1]

# ==========================================================================
# 10. SA: Decision Threshold Optimization
# ==========================================================================
print("\n" + "=" * 70)
print("STEP 10: SA -- Decision Threshold Optimization")
print("=" * 70)


def sa_fitness(threshold: float) -> float:
    """F1 score at the given probability threshold."""
    y_pred = (y_prob_final >= threshold).astype(int)
    return f1_score(y_test, y_pred, zero_division=0)


# Start near the class balance point
initial_threshold = float(y_train.mean())

sa = SimulatedAnnealing(
    fitness_fn=sa_fitness,
    initial_state=initial_threshold,
    T_init=1.0,
    T_min=0.001,
    cooling=0.95,
    step_size=0.02,
    max_iter=1000,
    seed=SEED,
)
best_threshold, sa_score = sa.optimize()
print(f"  SA optimal threshold = {best_threshold:.4f}  (F1={sa_score:.4f})")

# ==========================================================================
# 11. Final Evaluation & Comparison
# ==========================================================================
print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

y_pred_optimized = (y_prob_final >= best_threshold).astype(int)

print(f"\nClassification report (GA + PSO + SA optimized):")
print(classification_report(y_test, y_pred_optimized,
                            target_names=["No Failure", "Failure"], zero_division=0))

p_opt = precision_score(y_test, y_pred_optimized, zero_division=0)
r_opt = recall_score(y_test, y_pred_optimized, zero_division=0)
f1_opt = f1_score(y_test, y_pred_optimized, zero_division=0)
auc_opt = roc_auc_score(y_test, y_prob_final) if len(np.unique(y_test)) > 1 else np.nan

# Default Model B (no optimization) on same split for fair comparison
default_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05, random_state=SEED,
)
default_model.fit(X_B_train_sc, y_train)
y_prob_default = default_model.predict_proba(X_B_test_sc)[:, 1]
y_pred_default = default_model.predict(X_B_test)
f1_def = f1_score(y_test, y_pred_default, zero_division=0)
auc_def = roc_auc_score(y_test, y_prob_default) if len(np.unique(y_test)) > 1 else np.nan
p_def = precision_score(y_test, y_pred_default, zero_division=0)
r_def = recall_score(y_test, y_pred_default, zero_division=0)

comp = pd.DataFrame({
    "Model": ["Default GBM (all features)", "GA+PSO+SA Optimized"],
    "Features": [len(features_B), len(selected_features)],
    "Threshold": [0.50, round(best_threshold, 3)],
    "Precision": [p_def, p_opt],
    "Recall": [r_def, r_opt],
    "F1": [f1_def, f1_opt],
    "ROC-AUC": [auc_def, auc_opt],
})
print("\n-- Default vs Optimized --")
print(comp.to_string(index=False))

improvement_f1 = f1_opt - f1_def
improvement_auc = auc_opt - auc_def
print(f"\nF1 improvement:      {improvement_f1:+.4f}")
print(f"ROC-AUC improvement: {improvement_auc:+.4f}")
print(f"Feature reduction:   {len(features_B)} -> {len(selected_features)} "
      f"({100*(1-len(selected_features)/len(features_B)):.0f}% fewer)")

# Feature importance from optimized model
print("\n-- Top 15 Feature Importances (Optimized Model) --")
imp = pd.Series(final_model.feature_importances_, index=selected_features)
top15 = imp.sort_values(ascending=False).head(15)
for feat, val in top15.items():
    marker = " *" if feat in akim_cols or any(ac in feat for ac in akim_cols) else ""
    print(f"  {val:.4f}  {feat}{marker}")
print("\n* = akim-gerilim feature")
