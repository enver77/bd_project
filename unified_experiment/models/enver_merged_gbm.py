#!/usr/bin/env python3
"""
Enver's Approach 1 - Merged Model Comparison (GBM + GA/PSO/SA)
================================================================
Adapted to use unified preprocessing output.

Compares:
  Model A: energy consumption features only
  Model B: energy + current/voltage (akim-gerilim) features

Then applies GA (feature selection) + PSO (hyperparameter tuning) +
SA (threshold optimization) to the best model.

Reads from: RESULTS_DIR/hourly_data_labeled.csv + DATA_DIR/akim-gerilim_raporu_Vkx3H.xlsx
Writes to:  RESULTS_DIR/enver_merged_*.csv (prints comparison tables)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Add models dir to path for heuristic_opt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heuristic_opt import GeneticAlgorithm, PSO, SimulatedAnnealing

RESULTS_DIR = os.environ.get("BEDAS_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
DATA_DIR = os.environ.get("BEDAS_DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


def add_rolling_features(df, col="energy_kwh", windows=[3, 6, 12, 24]):
    df = df.copy()
    for w in windows:
        df[f"{col}_roll_mean_{w}h"] = df[col].rolling(w, min_periods=1).mean()
        df[f"{col}_roll_std_{w}h"] = df[col].rolling(w, min_periods=1).std().fillna(0)
    df[f"{col}_diff_1h"] = df[col].diff().fillna(0)
    df[f"{col}_diff_24h"] = df[col].diff(24).fillna(0)
    return df


def main():
    print("=" * 70)
    print(" Enver's Approach 1 - Merged Model Comparison (Unified Data)")
    print("=" * 70)

    # ── Load unified data ──
    cekis = pd.read_csv(os.path.join(RESULTS_DIR, "hourly_data_labeled.csv"),
                        parse_dates=["timestamp"])
    cekis = cekis.sort_values("timestamp").reset_index(drop=True)
    print(f"\nLoaded: {len(cekis)} rows")

    TARGET = "failure_next_24h"
    if TARGET not in cekis.columns:
        print(f"ERROR: {TARGET} column not found. Run unified_preprocessing first.")
        sys.exit(1)

    # ── Load akim-gerilim ──
    akim_file = os.path.join(DATA_DIR, "akim-gerilim_raporu_Vkx3H.xlsx")
    if not os.path.exists(akim_file):
        print(f"WARNING: {akim_file} not found. Running Model A only.")
        akim_available = False
    else:
        akim_available = True
        akim = pd.read_excel(akim_file)
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
                c.replace("Ak\u0131m", "current").replace("Gerilim", "voltage")
                .replace("Cosf", "pf").replace("Frekans", "freq")
                .replace("N\u00f6tr", "neutral").replace(" ", "_")
            )
        akim_hourly.rename(columns=col_map, inplace=True)

        print(f"Akim-gerilim hourly: {len(akim_hourly)} rows")

        merged = pd.merge(cekis, akim_hourly, on="timestamp", how="inner")
        print(f"Merged dataset: {len(merged)} rows (inner join)")

        merged.to_csv(os.path.join(RESULTS_DIR, "enver_cekis_akim_gerilim_merged.csv"), index=False)

    # ── Feature engineering ──
    data = merged if akim_available else cekis.copy()
    data = data.sort_values("timestamp").reset_index(drop=True)
    data = add_rolling_features(data, "energy_kwh")

    base_features = ["energy_kwh", "hour", "day_of_week", "is_weekend", "is_night", "month"]
    rolling_cols = [c for c in data.columns if "roll_" in c or "diff_" in c]
    features_A = base_features + rolling_cols

    if akim_available:
        akim_cols = [c for c in akim_hourly.columns if c != "timestamp"]
        features_B = features_A + akim_cols

        for ac in akim_cols[:3]:
            data = add_rolling_features(data, ac, windows=[3, 6, 12])
            extra = [c for c in data.columns if ac in c and ("roll_" in c or "diff_" in c)]
            features_B += extra
        features_B = list(dict.fromkeys(features_B))
    else:
        features_B = features_A
        akim_cols = []

    # Drop mostly-NaN columns
    mostly_nan = [c for c in features_B if c in data.columns and data[c].isna().mean() > 0.5]
    features_A = [c for c in features_A if c not in mostly_nan]
    features_B = [c for c in features_B if c not in mostly_nan]

    for col in features_B:
        if col in data.columns:
            data[col] = data[col].ffill().fillna(0)

    print(f"\nModel A features: {len(features_A)}")
    print(f"Model B features: {len(features_B)}")
    print(f"Target distribution: {data[TARGET].value_counts().to_dict()}")

    X_A = data[features_A].values
    X_B = data[features_B].values
    y = data[TARGET].values

    # ── Time-series CV ──
    tscv = TimeSeriesSplit(n_splits=5)

    def evaluate_model(X, y, tscv):
        fold_metrics = []
        clf = None
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)
            metrics = {
                "fold": fold + 1,
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }
            if len(np.unique(y_test)) > 1:
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                metrics["roc_auc"] = np.nan
            fold_metrics.append(metrics)
        return fold_metrics, clf

    print("\n" + "=" * 70)
    print("MODEL COMPARISON: failure_next_24h prediction")
    print("=" * 70)

    print("\n-- Model A: Energy Data Only --")
    metrics_A, clf_A = evaluate_model(X_A, y, tscv)
    df_A = pd.DataFrame(metrics_A)
    print(df_A.to_string(index=False))
    print(f"\nMean F1: {df_A['f1'].mean():.4f}  |  Mean ROC-AUC: {df_A['roc_auc'].mean():.4f}")

    if akim_available:
        print("\n-- Model B: Energy + Akim-Gerilim --")
        metrics_B, clf_B = evaluate_model(X_B, y, tscv)
        df_B = pd.DataFrame(metrics_B)
        print(df_B.to_string(index=False))
        print(f"\nMean F1: {df_B['f1'].mean():.4f}  |  Mean ROC-AUC: {df_B['roc_auc'].mean():.4f}")

        comparison = pd.DataFrame({
            "Metric": ["F1 Score", "ROC-AUC", "Precision", "Recall"],
            "Model A (energy)": [df_A["f1"].mean(), df_A["roc_auc"].mean(),
                                  df_A["precision"].mean(), df_A["recall"].mean()],
            "Model B (energy+akim)": [df_B["f1"].mean(), df_B["roc_auc"].mean(),
                                       df_B["precision"].mean(), df_B["recall"].mean()],
        })
        comparison["Improvement"] = comparison["Model B (energy+akim)"] - comparison["Model A (energy)"]
        print("\n" + comparison.to_string(index=False))

    # ── GA + PSO + SA optimization ──
    print("\n" + "=" * 70)
    print("GA + PSO + SA OPTIMIZATION")
    print("=" * 70)

    split_idx = int(len(data) * 0.8)
    X_opt = X_B if akim_available else X_A
    features_opt = features_B if akim_available else features_A

    X_train_opt, X_test_opt = X_opt[:split_idx], X_opt[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler_opt = StandardScaler()
    X_train_sc = scaler_opt.fit_transform(X_train_opt)
    X_test_sc = scaler_opt.transform(X_test_opt)

    SEED = 42

    def ga_fitness(chromosome):
        mask = chromosome.astype(bool)
        if mask.sum() < 2: return -1.0
        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05, random_state=SEED)
        model.fit(X_train_sc[:, mask], y_train)
        y_prob = model.predict_proba(X_test_sc[:, mask])[:, 1]
        best_f1 = 0.0
        for t in np.arange(0.05, 0.5, 0.02):
            f = f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
            if f > best_f1: best_f1 = f
        return best_f1

    ga = GeneticAlgorithm(fitness_fn=ga_fitness, n_genes=len(features_opt),
                          pop_size=20, n_gen=30, crossover_rate=0.8,
                          mutation_rate=0.1, min_active=3, seed=SEED, verbose=5)
    best_chromosome, ga_score = ga.optimize()

    selected = [f for f, flag in zip(features_opt, best_chromosome) if flag]
    print(f"\nGA selected {len(selected)}/{len(features_opt)} features (F1={ga_score:.4f})")

    feature_mask = best_chromosome.astype(bool)
    X_sel_train = X_train_sc[:, feature_mask]
    X_sel_test = X_test_sc[:, feature_mask]

    def pso_fitness(params):
        lr = float(np.clip(params[0], 0.01, 0.20))
        depth = int(np.clip(round(params[1]), 2, 10))
        subsample = float(np.clip(params[2], 0.5, 1.0))
        min_leaf = int(np.clip(round(params[3]), 2, 30))
        n_est = int(np.clip(round(params[4]), 50, 400))
        model = GradientBoostingClassifier(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            subsample=subsample, min_samples_leaf=min_leaf, random_state=SEED)
        model.fit(X_sel_train, y_train)
        y_prob = model.predict_proba(X_sel_test)[:, 1]
        best_f1 = 0.0
        for t in np.arange(0.05, 0.5, 0.02):
            f = f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
            if f > best_f1: best_f1 = f
        return best_f1

    pso = PSO(fitness_fn=pso_fitness,
              bounds=[[0.01, 0.20], [2.0, 10.0], [0.5, 1.0], [2.0, 30.0], [50.0, 400.0]],
              n_particles=15, n_iter=25, seed=SEED, verbose=5)
    best_params, pso_score = pso.optimize()

    best_lr = float(np.clip(best_params[0], 0.01, 0.20))
    best_depth = int(np.clip(round(best_params[1]), 2, 10))
    best_subsample = float(np.clip(best_params[2], 0.5, 1.0))
    best_min_leaf = int(np.clip(round(best_params[3]), 2, 30))
    best_n_est = int(np.clip(round(best_params[4]), 50, 400))

    print(f"\nPSO best hyperparameters (F1={pso_score:.4f}):")
    print(f"  n_estimators={best_n_est}, max_depth={best_depth}, "
          f"lr={best_lr:.4f}, subsample={best_subsample:.4f}, min_leaf={best_min_leaf}")

    final_model = GradientBoostingClassifier(
        n_estimators=best_n_est, max_depth=best_depth, learning_rate=best_lr,
        subsample=best_subsample, min_samples_leaf=best_min_leaf, random_state=SEED)
    final_model.fit(X_sel_train, y_train)
    y_prob_final = final_model.predict_proba(X_sel_test)[:, 1]

    def sa_fitness(threshold):
        return f1_score(y_test, (y_prob_final >= threshold).astype(int), zero_division=0)

    sa = SimulatedAnnealing(fitness_fn=sa_fitness, initial_state=float(y_train.mean()),
                            T_init=1.0, T_min=0.001, cooling=0.95, step_size=0.02,
                            max_iter=1000, seed=SEED)
    best_threshold, sa_score = sa.optimize()

    # ── Final evaluation ──
    y_pred_opt = (y_prob_final >= best_threshold).astype(int)
    print(f"\nClassification report (GA + PSO + SA optimized):")
    print(classification_report(y_test, y_pred_opt,
                                target_names=["No Failure", "Failure"], zero_division=0))

    # Feature importance
    print("-- Top 15 Feature Importances (Optimized Model) --")
    imp = pd.Series(final_model.feature_importances_, index=selected)
    top15 = imp.sort_values(ascending=False).head(15)
    for feat, val in top15.items():
        marker = " *" if akim_available and (feat in akim_cols or any(ac in feat for ac in akim_cols)) else ""
        print(f"  {val:.4f}  {feat}{marker}")
    if akim_available:
        print("\n* = akim-gerilim feature")

    print("\n" + "=" * 70)
    print(" Enver Merged Model Comparison COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
