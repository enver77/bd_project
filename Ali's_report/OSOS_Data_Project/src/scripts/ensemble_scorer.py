"""
Ensemble Scorer
Fuses Tier1 + Tier2 + Tier3 scores into a final anomaly report.

S1 = max(T1a_zscore_severity, T1b_cusum_severity)
S2 = max(T2a_iforest,         T2b_lstm)
S3 = 1.0 if ±2 days of CP, else 0.0
S_final = 0.35*S1 + 0.35*S2 + 0.30*S3

Output: outputs/anomaly_report.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAILY_CSV    = PROJECT_ROOT / "data" / "processed" / "preprocessed_daily.csv"
GT_CSV       = PROJECT_ROOT / "data" / "processed" / "ground_truth_labels.csv"
OUT_CSV      = PROJECT_ROOT / "outputs" / "anomaly_report.csv"

W1, W2, W3 = 0.35, 0.35, 0.30


def compute_ensemble(
    tier1: pd.DataFrame,
    tier2: pd.DataFrame,
    tier3: pd.DataFrame,
    daily: pd.DataFrame,
    gt: pd.DataFrame = None,
) -> pd.DataFrame:

    idx = daily.index

    # Align all tiers to same index
    t1 = tier1.reindex(idx)
    t2 = tier2.reindex(idx)
    t3 = tier3.reindex(idx)

    # Component scores
    s1 = np.maximum.reduce([
        t1["t1a_severity"].fillna(0.0).values,
        t1["t1b_severity"].fillna(0.0).values,
        t1.get("t1c_severity", pd.Series(0.0, index=idx)).reindex(idx, fill_value=0.0).values,
    ])
    s2 = np.maximum(
        t2["t2a_iforest_severity"].fillna(0.0).values,
        t2["t2b_lstm_severity"].fillna(0.0).values,
    )
    s3 = t3["t3_severity"].fillna(0.0).values

    s_final = W1 * s1 + W2 * s2 + W3 * s3

    # Vote count (how many tiers flagged the day)
    vote1 = t1.get("tier1_flag", pd.Series(False, index=idx)).reindex(idx, fill_value=False).astype(int).values
    vote2 = t2["tier2_flag"].fillna(False).astype(int).values
    vote3 = t3["t3_flag"].fillna(False).astype(int).values
    vote_count = vote1 + vote2 + vote3

    # Anomaly type — uses already-computed baseline column
    def classify_type(row):
        kwh  = row["daily_active_kwh"]
        base = row["baseline_kwh"]
        if base == 0 or np.isnan(base):
            base = kwh if kwh > 0 else 1.0
        dev = (kwh - base) / (base + 1e-9)
        if dev < -0.30:
            return "outage"
        elif dev < -0.10:
            return "sudden_drop"
        elif dev < -0.05:
            return "gradual_decline"
        elif dev > 0.10:
            return "increase"
        return "normal"

    # Deviation %
    if "rolling_28d_median" in daily.columns:
        baseline = daily["rolling_28d_median"].fillna(daily["daily_active_kwh"].median())
    else:
        baseline = daily["daily_active_kwh"].rolling(28, min_periods=7).median().shift(1).bfill()
        baseline = baseline.fillna(daily["daily_active_kwh"].median())
    deviation_pct = (daily["daily_active_kwh"] - baseline) / (baseline + 1e-9) * 100.0

    # Confidence
    def confidence(score, votes):
        if score >= 0.60 or votes >= 2:
            return "high"
        elif score >= 0.35 or votes >= 1:
            return "moderate"
        return "low"

    report = pd.DataFrame({
        "date": idx,
        "daily_active_kwh": daily["daily_active_kwh"].values,
        "baseline_kwh": baseline.values,
        "deviation_pct": deviation_pct.values,
        "active_hours_count": daily["active_hours_count"].values,
        "mean_active_intensity": daily["mean_active_intensity"].values,
        "t1a_zscore_severity": t1["t1a_severity"].fillna(0.0).values,
        "t1b_cusum_severity":  t1["t1b_severity"].fillna(0.0).values,
        "t2a_iforest_severity": t2["t2a_iforest_severity"].fillna(0.0).values,
        "t2b_lstm_severity":    t2["t2b_lstm_severity"].fillna(0.0).values,
        "t3_changepoint_severity": s3,
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "ensemble_score": s_final,
        "vote_count": vote_count,
    }, index=idx)

    report["anomaly_type"] = report.apply(classify_type, axis=1)
    report["confidence"]   = [
        confidence(score, votes)
        for score, votes in zip(s_final, vote_count)
    ]

    # Physical plausibility: deviation < -20%
    report["physically_supported"] = deviation_pct < -20.0

    # Ground truth column
    if gt is not None:
        gt_aligned = gt["outage"].reindex(idx, fill_value=0)
        report["is_ground_truth"] = gt_aligned.astype(int).values
    else:
        report["is_ground_truth"] = 0

    report = report.set_index("date")
    return report


def main(tier1=None, tier2=None, tier3=None, daily=None, gt=None) -> pd.DataFrame:
    if daily is None:
        daily = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)

    # Lazy imports to allow standalone use
    if tier1 is None:
        from tier1_statistical import main as t1_main
        tier1 = t1_main(daily.copy())

    if tier2 is None:
        from tier2_ml import main as t2_main
        tier2 = t2_main(daily.copy())

    if tier3 is None:
        from tier3_changepoint import main as t3_main
        t3_result = t3_main(daily.copy())
        tier3 = t3_result[0] if isinstance(t3_result, tuple) else t3_result

    if gt is None and GT_CSV.exists():
        gt = pd.read_csv(GT_CSV, index_col=0, parse_dates=True)

    print("Computing ensemble scores …")
    report = compute_ensemble(tier1, tier2, tier3, daily, gt)

    n_high = (report["confidence"] == "high").sum()
    n_anomaly = (report["ensemble_score"] >= 0.35).sum()
    print(f"  High-confidence anomalies: {n_high}")
    print(f"  Total flagged (score≥0.35): {n_anomaly}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT_CSV)
    print(f"Saved → {OUT_CSV}")

    return report


if __name__ == "__main__":
    main()
