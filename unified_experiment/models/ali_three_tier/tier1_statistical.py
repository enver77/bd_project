"""
Ali's Tier 1 -- Statistical Anomaly Detection
  1a. STL decomposition + Modified Z-Score on residuals
  1b. CUSUM control chart on STL residuals
  1c. Rolling-baseline deviation (direct signal)

Reads from: RESULTS_DIR/ali_preprocessed_daily.csv
Writes to:  RESULTS_DIR/ali_tier1_results.json
"""

import os
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

RESULTS_DIR = os.environ.get("BEDAS_RESULTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"))

DAILY_CSV = os.path.join(RESULTS_DIR, "ali_preprocessed_daily.csv")
OUT_JSON = os.path.join(RESULTS_DIR, "ali_tier1_results.json")


def run_stl(series, period=365, robust=True):
    stl = STL(series.dropna(), period=period, robust=robust)
    result = stl.fit()
    residual = pd.Series(result.resid, index=series.dropna().index)
    return result, residual


def modified_zscore(series):
    values = series.values
    median = np.median(values[np.isfinite(values)])
    mad = np.median(np.abs(values[np.isfinite(values)] - median))
    if mad < 1e-10:
        q75, q25 = np.percentile(values[np.isfinite(values)], [75, 25])
        mad = (q75 - q25) / 1.349
    if mad < 1e-10:
        mad = np.std(values[np.isfinite(values)]) + 1e-10
    return 0.6745 * (series - median) / mad


def zscore_severity(mzs, threshold=3.5, scale=None):
    if scale is None:
        scale = 2.0 * threshold
    sev = pd.Series(0.0, index=mzs.index)
    mask = mzs < -threshold
    sev[mask] = np.minimum(np.abs(mzs[mask]) / scale, 1.0)
    return sev


def cusum_lower(series, k_factor=0.5, h_factor=5.0):
    vals = series.fillna(0.0).values
    sigma = float(np.std(vals, ddof=1)) or 1.0
    k = k_factor * sigma
    h = h_factor * sigma
    s_neg = np.zeros(len(vals))
    for i in range(1, len(vals)):
        s_neg[i] = max(0.0, s_neg[i-1] - vals[i] - k)
    s_neg_s = pd.Series(s_neg, index=series.index, name="cusum_neg")
    alarms = s_neg_s > h
    severity = pd.Series(
        np.minimum(s_neg_s.values / (2 * h + 1e-9), 1.0),
        index=series.index, name="cusum_severity"
    )
    return s_neg_s, alarms, severity, {"k": round(k, 4), "h": round(h, 4), "sigma": round(sigma, 4)}


def rolling_baseline_anomaly(series, baseline, mzs_threshold=1.5):
    deviation = (series - baseline) / (baseline.replace(0, np.nan) + 1e-9)
    deviation = deviation.fillna(0.0)
    mzs_dev = modified_zscore(deviation)
    sev_dev = zscore_severity(mzs_dev, threshold=mzs_threshold, scale=3.0 * mzs_threshold)
    flag_dev = mzs_dev < -mzs_threshold
    _, cusum_alarms, cusum_sev, _ = cusum_lower(deviation, k_factor=0.3, h_factor=4.0)
    flag_combined = flag_dev | cusum_alarms
    sev_combined = np.maximum(sev_dev, cusum_sev)
    return mzs_dev, flag_combined, sev_combined


def stationarity_checks(residual):
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller

    clean = residual.dropna()
    if clean.std() < 1e-10:
        return {"adf_pvalue": None, "note": "near-constant residual"}

    adf_stat, adf_p, *_ = adfuller(clean, autolag="AIC")
    lb = acorr_ljungbox(clean, lags=[10], return_df=True)
    dw = durbin_watson(clean)
    return {
        "adf_stat": float(adf_stat), "adf_pvalue": float(adf_p),
        "adf_stationary": bool(adf_p < 0.05),
        "ljungbox_lb_stat": float(lb["lb_stat"].iloc[0]),
        "ljungbox_lb_pvalue": float(lb["lb_pvalue"].iloc[0]),
        "ljungbox_no_autocorr": bool(lb["lb_pvalue"].iloc[0] > 0.05),
        "durbin_watson": float(dw),
    }


def main(daily=None):
    if daily is None:
        daily = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)

    series = daily["daily_active_kwh"].copy().fillna(0.0)

    print("Running STL decomposition (period=365, robust=True) ...")
    stl_result, residual = run_stl(series, period=365, robust=True)

    tier1 = daily[["daily_active_kwh", "rolling_28d_median"]].copy()
    tier1["stl_trend"] = pd.Series(stl_result.trend, index=series.index)
    tier1["stl_seasonal"] = pd.Series(stl_result.seasonal, index=series.index)
    tier1["stl_residual"] = residual.reindex(series.index, fill_value=0.0)

    resid_filled = tier1["stl_residual"].fillna(0.0)

    print("Computing Modified Z-Score on STL residual ...")
    mzs_stl = modified_zscore(resid_filled)
    t1a_sev = zscore_severity(mzs_stl, threshold=3.5)
    tier1["mzs_stl"] = mzs_stl
    tier1["t1a_flag"] = (mzs_stl < -3.5)
    tier1["t1a_severity"] = t1a_sev

    print("Computing lower-side CUSUM on STL residual ...")
    s_neg, alarms, cusum_sev, cusum_params = cusum_lower(resid_filled)
    tier1["cusum_neg"] = s_neg
    tier1["t1b_alarm"] = alarms
    tier1["t1b_severity"] = cusum_sev

    print("Computing rolling-baseline deviation anomaly score ...")
    baseline = daily["rolling_28d_median"].fillna(series.median())
    mzs_dev, flag_dev, sev_dev = rolling_baseline_anomaly(series, baseline, mzs_threshold=1.5)
    tier1["mzs_deviation"] = mzs_dev
    tier1["t1c_flag"] = flag_dev
    tier1["t1c_severity"] = sev_dev

    tier1["tier1_severity"] = np.maximum.reduce([
        tier1["t1a_severity"].fillna(0.0),
        tier1["t1b_severity"].fillna(0.0),
        tier1["t1c_severity"].fillna(0.0),
    ])
    tier1["tier1_flag"] = (
        tier1["t1a_flag"].fillna(False) |
        tier1["t1b_alarm"].fillna(False) |
        tier1["t1c_flag"].fillna(False)
    )

    print("Running stationarity checks on STL residual ...")
    stat_checks = stationarity_checks(tier1["stl_residual"])
    if stat_checks.get("adf_pvalue") is not None:
        print(f"  ADF p={stat_checks['adf_pvalue']:.4f} stationary={stat_checks['adf_stationary']}")

    flagged_days = tier1[tier1["tier1_flag"]].index.strftime("%Y-%m-%d").tolist()
    print(f"Tier1 flagged {len(flagged_days)} anomalous days "
          f"(T1a:{int(tier1['t1a_flag'].sum())} T1b:{int(tier1['t1b_alarm'].sum())} "
          f"T1c:{int(tier1['t1c_flag'].sum())})")

    results = {
        "method": "STL + Modified-ZScore + CUSUM + Rolling-Baseline-Deviation",
        "stl_period": 365, "stl_robust": True,
        "zscore_threshold_stl": 3.5, "zscore_threshold_deviation": 1.5,
        "cusum_params": cusum_params,
        "stationarity_checks": stat_checks,
        "n_flagged_t1a": int(tier1["t1a_flag"].sum()),
        "n_flagged_t1b": int(tier1["t1b_alarm"].sum()),
        "n_flagged_t1c": int(tier1["t1c_flag"].sum()),
        "n_flagged_total": int(tier1["tier1_flag"].sum()),
        "flagged_dates": flagged_days,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {OUT_JSON}")

    return tier1


if __name__ == "__main__":
    main()
