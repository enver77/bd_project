"""
Tier 3 — Change-Point Detection
  PELT algorithm (Killick et al., 2012) via ruptures library (Truong et al., 2020)

Output: outputs/tier3_results.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAILY_CSV    = PROJECT_ROOT / "data" / "processed" / "preprocessed_daily.csv"
OUT_JSON     = PROJECT_ROOT / "outputs" / "tier3_results.json"

WINDOW = 14  # days before/after change point for mean comparison


# ---------------------------------------------------------------------------
# PELT
# ---------------------------------------------------------------------------

def run_pelt(
    signal: np.ndarray,
    index: pd.DatetimeIndex,
    model: str = "rbf",
    min_size: int = 3,
    penalty: float = 10.0,
) -> list[pd.Timestamp]:
    """Return list of change-point timestamps (start of each new segment)."""
    try:
        import ruptures as rpt
    except ImportError:
        print("  ruptures not installed — skipping PELT")
        return [], []

    algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
    # fit() expects 1D array; pass as column
    breakpoints = algo.predict(pen=penalty)
    # breakpoints are indices (1-based end of segment, last = len(signal))
    cp_indices = [bp for bp in breakpoints if bp < len(signal)]
    cp_dates   = [index[i] for i in cp_indices]
    return cp_dates, cp_indices


def analyze_changepoints(
    series: pd.Series,
    cp_dates: list,
    window: int = WINDOW,
) -> pd.DataFrame:
    """Compute delta and relative change around each change point."""
    records = []
    for cp in cp_dates:
        loc = series.index.get_loc(cp)
        before_start = max(0, loc - window)
        after_end    = min(len(series), loc + window)

        before = series.iloc[before_start:loc]
        after  = series.iloc[loc:after_end]

        mean_before = before.mean() if len(before) > 0 else np.nan
        mean_after  = after.mean()  if len(after)  > 0 else np.nan
        delta       = mean_after - mean_before if not (np.isnan(mean_before) or np.isnan(mean_after)) else np.nan

        if not np.isnan(mean_before) and mean_before != 0:
            rel_change = delta / abs(mean_before)
        else:
            rel_change = np.nan

        # Classify
        if np.isnan(rel_change):
            cp_type = "unknown"
        elif rel_change < -0.20:
            cp_type = "sudden_drop"
        elif rel_change < -0.05:
            cp_type = "partial_fault"
        elif rel_change > 0.05:
            cp_type = "grid_increase"
        else:
            cp_type = "minor"

        records.append({
            "changepoint_date": cp,
            "mean_before": round(mean_before, 4) if not np.isnan(mean_before) else None,
            "mean_after":  round(mean_after,  4) if not np.isnan(mean_after)  else None,
            "delta":       round(delta,        4) if not np.isnan(delta)       else None,
            "relative_change": round(rel_change, 4) if not np.isnan(rel_change) else None,
            "type": cp_type,
        })

    return pd.DataFrame(records)


def pelt_severity_series(
    cp_analysis: pd.DataFrame,
    index: pd.DatetimeIndex,
    window: int = 2,
) -> pd.Series:
    """
    For each day, severity = 1.0 if within ±window days of a significant drop CP,
    else 0.0.
    """
    sev = pd.Series(0.0, index=index, name="t3_severity")
    if cp_analysis.empty:
        return sev

    drop_cps = cp_analysis[cp_analysis["type"].isin(["sudden_drop", "partial_fault"])]
    for _, row in drop_cps.iterrows():
        cp_date = row["changepoint_date"]
        rel = abs(row["relative_change"] or 0)
        strength = min(rel / 0.20, 1.0)
        for d in index:
            if abs((d - cp_date).days) <= window:
                sev[d] = max(sev[d], strength)
    return sev


# ---------------------------------------------------------------------------
# Sensitivity analysis over penalty values
# ---------------------------------------------------------------------------

def pelt_sensitivity(
    signal: np.ndarray,
    index: pd.DatetimeIndex,
    penalties: list = [5, 10, 20],
) -> dict:
    try:
        import ruptures as rpt
    except ImportError:
        return {}

    results = {}
    for pen in penalties:
        algo = rpt.Pelt(model="rbf", min_size=3).fit(signal)
        bps = algo.predict(pen=pen)
        cp_idx = [bp for bp in bps if bp < len(signal)]
        cp_dates = [index[i].strftime("%Y-%m-%d") for i in cp_idx]
        results[str(pen)] = {"n_changepoints": len(cp_dates), "dates": cp_dates}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(daily: pd.DataFrame = None) -> pd.DataFrame:
    if daily is None:
        daily = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)

    series_kwh  = daily["daily_active_kwh"].fillna(0.0)
    series_int  = daily["mean_active_intensity"].fillna(0.0)

    print("Running PELT on daily_active_kwh (pen=10) …")
    cp_dates_kwh, cp_idx_kwh = run_pelt(
        series_kwh.values.reshape(-1, 1),
        series_kwh.index,
        model="rbf",
        min_size=3,
        penalty=10.0,
    )
    print(f"  Found {len(cp_dates_kwh)} change points in kWh series")

    print("Running PELT on mean_active_intensity (pen=10) …")
    cp_dates_int, cp_idx_int = run_pelt(
        series_int.values.reshape(-1, 1),
        series_int.index,
        model="rbf",
        min_size=3,
        penalty=10.0,
    )
    print(f"  Found {len(cp_dates_int)} change points in intensity series")

    # Analyze change points
    cp_kwh_df = analyze_changepoints(series_kwh, cp_dates_kwh)
    cp_int_df = analyze_changepoints(series_int, cp_dates_int)

    # Severity series (±2 days)
    t3_sev = pelt_severity_series(cp_kwh_df, daily.index, window=2)

    tier3 = daily[["daily_active_kwh"]].copy()
    tier3["t3_severity"] = t3_sev
    tier3["t3_flag"]     = t3_sev > 0.0

    # Sensitivity analysis
    print("Sensitivity analysis (pen ∈ {5, 10, 20}) …")
    sensitivity = pelt_sensitivity(
        series_kwh.values.reshape(-1, 1), series_kwh.index
    )
    for pen, info in sensitivity.items():
        print(f"  pen={pen}: {info['n_changepoints']} change points")

    # Save JSON
    def cp_row_to_dict(row):
        d = row.to_dict()
        if "changepoint_date" in d and hasattr(d["changepoint_date"], "strftime"):
            d["changepoint_date"] = d["changepoint_date"].strftime("%Y-%m-%d")
        return d

    results = {
        "method": "PELT (ruptures, model=rbf, MBIC-like penalty)",
        "kwh_series": {
            "n_changepoints": len(cp_dates_kwh),
            "changepoints": [cp_row_to_dict(r) for _, r in cp_kwh_df.iterrows()],
        },
        "intensity_series": {
            "n_changepoints": len(cp_dates_int),
            "changepoints": [cp_row_to_dict(r) for _, r in cp_int_df.iterrows()],
        },
        "sensitivity": sensitivity,
        "n_flagged_days": int(tier3["t3_flag"].sum()),
        "flagged_dates": tier3[tier3["t3_flag"]].index.strftime("%Y-%m-%d").tolist(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved → {OUT_JSON}")

    return tier3, cp_kwh_df


if __name__ == "__main__":
    main()
