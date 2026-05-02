"""
Tier 2 — Machine Learning Anomaly Detection
  2a. Isolation Forest (Liu et al., 2008)
  2b. LSTM Autoencoder (Malhotra et al., 2016)  — ablation comparison

Output: outputs/tier2_results.json, outputs/models/iforest_model.pkl
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAILY_CSV    = PROJECT_ROOT / "data" / "processed" / "preprocessed_daily.csv"
OUT_JSON     = PROJECT_ROOT / "outputs" / "tier2_results.json"
MODEL_PATH   = PROJECT_ROOT / "outputs" / "models" / "iforest_model.pkl"

FEATURE_COLS = [
    "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
    "daily_active_kwh", "daily_kwh_norm", "mean_active_intensity",
    "p10_active", "active_hours_count", "night_missing_frac",
    "delta_1d", "delta_7d", "rolling_7d_std", "rolling_28d_zscore",
]

SEQ_LEN = 7  # LSTM lookback window


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    feat = daily[FEATURE_COLS].copy()
    feat = feat.fillna(0.0)
    scaler = RobustScaler()
    X = scaler.fit_transform(feat)
    return feat, X, scaler


# ---------------------------------------------------------------------------
# 2a. Isolation Forest
# ---------------------------------------------------------------------------

def run_isolation_forest(
    X: np.ndarray,
    index: pd.DatetimeIndex,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series]:
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X)
    raw_scores = clf.score_samples(X)          # lower = more anomalous
    # Normalize to [0,1] where 1 = most anomalous
    norm = (raw_scores - raw_scores.max()) / (raw_scores.min() - raw_scores.max() + 1e-9)
    norm = np.clip(norm, 0.0, 1.0)
    predictions = clf.predict(X)               # -1 = anomaly, +1 = normal

    scores = pd.Series(norm, index=index, name="t2a_iforest_severity")
    flags  = pd.Series(predictions == -1, index=index, name="t2a_flag")
    return scores, flags, clf


def iforest_sensitivity(
    X: np.ndarray,
    index: pd.DatetimeIndex,
    contaminations: list = [0.02, 0.05, 0.10],
) -> dict:
    """Return flagged date sets per contamination level."""
    results = {}
    for c in contaminations:
        clf = IsolationForest(n_estimators=200, contamination=c, random_state=42, n_jobs=-1)
        clf.fit(X)
        pred = clf.predict(X)
        flagged = index[pred == -1].strftime("%Y-%m-%d").tolist()
        results[str(c)] = flagged
    return results


def iforest_bootstrap_stability(
    X: np.ndarray,
    index: pd.DatetimeIndex,
    seeds: list = [42, 7, 13, 99, 2025],
) -> dict:
    """Run iForest with 5 seeds, return Jaccard similarity matrix."""
    flag_sets = {}
    for s in seeds:
        clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=s, n_jobs=-1)
        clf.fit(X)
        pred = clf.predict(X)
        flag_sets[s] = set(index[pred == -1].strftime("%Y-%m-%d").tolist())

    seed_list = list(flag_sets.keys())
    jaccard = {}
    for i, si in enumerate(seed_list):
        for j, sj in enumerate(seed_list):
            if i < j:
                a, b = flag_sets[si], flag_sets[sj]
                jac = len(a & b) / len(a | b) if (a | b) else 1.0
                jaccard[f"{si}_vs_{sj}"] = float(jac)
    avg_jaccard = float(np.mean(list(jaccard.values()))) if jaccard else 1.0
    return {"jaccard_pairs": jaccard, "avg_jaccard": avg_jaccard}


# ---------------------------------------------------------------------------
# 2b. LSTM Autoencoder
# ---------------------------------------------------------------------------

def build_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    seqs = []
    for i in range(len(X) - seq_len + 1):
        seqs.append(X[i : i + seq_len])
    return np.array(seqs)


def run_lstm_autoencoder(
    X: np.ndarray,
    index: pd.DatetimeIndex,
    seq_len: int = SEQ_LEN,
    n_train_months: int = 10,
) -> tuple[pd.Series, pd.Series]:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, LSTM, Dense, RepeatVector, TimeDistributed
        )
        from tensorflow.keras.callbacks import EarlyStopping
        tf.random.set_seed(42)
    except ImportError:
        print("  TensorFlow not available — skipping LSTM autoencoder")
        dummy = pd.Series(0.0, index=index, name="t2b_lstm_severity")
        return dummy, pd.Series(False, index=index, name="t2b_flag")

    # Train/test split: first n_train_months months = "normal"
    train_cutoff = index[0] + pd.DateOffset(months=n_train_months)
    train_mask   = index < train_cutoff

    n_features = X.shape[1]
    seqs_all = build_sequences(X, seq_len)
    # Align sequence index: each sequence's label = last day of the window
    seq_idx = index[seq_len - 1 :]

    train_seq_mask = seq_idx < train_cutoff
    X_train = seqs_all[train_seq_mask]
    X_test  = seqs_all

    if len(X_train) < 20:
        print("  Not enough training data for LSTM — skipping")
        dummy = pd.Series(0.0, index=index, name="t2b_lstm_severity")
        return dummy, pd.Series(False, index=index, name="t2b_flag")

    # Build autoencoder
    inp = Input(shape=(seq_len, n_features))
    enc = LSTM(32, activation="relu", return_sequences=False)(inp)
    rep = RepeatVector(seq_len)(enc)
    dec = LSTM(32, activation="relu", return_sequences=True)(rep)
    out = TimeDistributed(Dense(n_features))(dec)
    ae  = Model(inp, out)
    ae.compile(optimizer="adam", loss="mae")

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ae.fit(
        X_train, X_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[es],
        verbose=0,
    )

    # Reconstruction error
    recon       = ae.predict(X_test, verbose=0)
    recon_error = np.mean(np.abs(X_test - recon), axis=(1, 2))

    # Threshold: 95th percentile of training reconstruction error
    train_recon  = ae.predict(X_train, verbose=0)
    train_error  = np.mean(np.abs(X_train - train_recon), axis=(1, 2))
    threshold    = float(np.percentile(train_error, 95))

    # Severity
    sev = np.minimum(recon_error / (threshold + 1e-9), 1.0)

    # Map back to daily index (pad first seq_len-1 days with 0)
    severity_full = np.zeros(len(index))
    severity_full[seq_len - 1 :] = sev

    severity_series = pd.Series(severity_full, index=index, name="t2b_lstm_severity")
    flag_series     = pd.Series(recon_error > threshold, index=seq_idx, name="t2b_flag").reindex(index, fill_value=False)

    print(f"  LSTM threshold: {threshold:.4f}, flagged: {flag_series.sum()} days")
    return severity_series, flag_series


# ---------------------------------------------------------------------------
# PCA projection for visualization
# ---------------------------------------------------------------------------

def pca_projection(X: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X)
    return pd.DataFrame({"pc1": proj[:, 0], "pc2": proj[:, 1]}, index=index)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(daily: pd.DataFrame = None) -> pd.DataFrame:
    if daily is None:
        daily = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)

    print("Preparing features …")
    feat_df, X, scaler = prepare_features(daily)

    # 2a Isolation Forest
    print("Running Isolation Forest (contamination=0.05) …")
    t2a_sev, t2a_flag, clf = run_isolation_forest(X, feat_df.index, contamination=0.05)
    print(f"  Flagged {t2a_flag.sum()} days")

    # Sensitivity analysis
    print("Sensitivity analysis (contamination ∈ {0.02, 0.05, 0.10}) …")
    sensitivity = iforest_sensitivity(X, feat_df.index)

    # Bootstrap stability
    print("Bootstrap stability (5 seeds) …")
    bootstrap = iforest_bootstrap_stability(X, feat_df.index)
    print(f"  Average Jaccard: {bootstrap['avg_jaccard']:.3f}")

    # PCA
    pca_df = pca_projection(X, feat_df.index)

    # 2b LSTM Autoencoder
    print("Running LSTM Autoencoder …")
    t2b_sev, t2b_flag = run_lstm_autoencoder(X, feat_df.index)

    # Assemble tier2 DataFrame
    tier2 = pd.DataFrame({
        "t2a_iforest_severity": t2a_sev,
        "t2a_flag": t2a_flag,
        "t2b_lstm_severity": t2b_sev,
        "t2b_flag": t2b_flag,
        "pc1": pca_df["pc1"],
        "pc2": pca_df["pc2"],
    }, index=feat_df.index)

    tier2["tier2_severity"] = np.maximum(tier2["t2a_iforest_severity"], tier2["t2b_lstm_severity"])
    tier2["tier2_flag"]     = tier2["t2a_flag"] | tier2["t2b_flag"]

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler, "features": FEATURE_COLS}, f)
    print(f"iForest model saved → {MODEL_PATH}")

    # Save JSON results
    flagged_t2a = tier2[tier2["t2a_flag"]].index.strftime("%Y-%m-%d").tolist()
    flagged_t2b = tier2[tier2["t2b_flag"]].index.strftime("%Y-%m-%d").tolist()

    results = {
        "method": "IsolationForest + LSTM Autoencoder",
        "iforest": {
            "n_estimators": 200,
            "contamination": 0.05,
            "n_flagged": int(t2a_flag.sum()),
            "flagged_dates": flagged_t2a,
        },
        "lstm": {
            "seq_len": SEQ_LEN,
            "n_flagged": int(t2b_flag.sum()),
            "flagged_dates": flagged_t2b,
        },
        "sensitivity_analysis": {
            c: len(v) for c, v in sensitivity.items()
        },
        "bootstrap_stability": bootstrap,
        "features_used": FEATURE_COLS,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved → {OUT_JSON}")

    return tier2


if __name__ == "__main__":
    main()
