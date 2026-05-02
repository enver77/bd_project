"""
Publication-Quality Visualizations
  1. STL Decomposition (4-panel)
  2. CUSUM Chart
  3. iForest PCA Scatter
  4. Change-Point Timeline
  5. Calendar Heat Map

Output: outputs/visualizations/anomaly_detection/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAILY_CSV    = PROJECT_ROOT / "data" / "processed" / "preprocessed_daily.csv"
GT_CSV       = PROJECT_ROOT / "data" / "processed" / "ground_truth_labels.csv"
REPORT_CSV   = PROJECT_ROOT / "outputs" / "anomaly_report.csv"
OUT_DIR      = PROJECT_ROOT / "outputs" / "visualizations" / "anomaly_detection"

COLORS = {
    "primary":    "#1a73e8",
    "anomaly":    "#d32f2f",
    "gt":         "#f57c00",
    "cusum":      "#6a1b9a",
    "trend":      "#2e7d32",
    "seasonal":   "#00838f",
    "residual":   "#455a64",
    "background": "#fafafa",
}


def savefig(fig, name: str, dpi: int = 300):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# 1. STL Decomposition (4-panel)
# ---------------------------------------------------------------------------

def plot_stl(tier1: pd.DataFrame, report: pd.DataFrame, gt: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "STL Decomposition — BEDAŞ Sokak Aydınlatma Hattı 2193681000\n"
        "Günlük Aktif Enerji (kWh)",
        fontsize=13, fontweight="bold", y=0.98
    )

    # Anomaly dates (high confidence)
    anom_dates = report[report["confidence"] == "high"].index
    gt_dates   = gt[gt["outage"] == 1].index if gt is not None else pd.DatetimeIndex([])

    series_labels = [
        ("daily_active_kwh",   "Ham Seri (kWh)",     COLORS["primary"]),
        ("stl_trend",          "Trend",              COLORS["trend"]),
        ("stl_seasonal",       "Mevsimsel",          COLORS["seasonal"]),
        ("stl_residual",       "Kalıntı",            COLORS["residual"]),
    ]

    for ax, (col, label, color) in zip(axes, series_labels):
        vals = tier1[col] if col in tier1.columns else None
        if vals is None and col == "daily_active_kwh":
            vals = report["daily_active_kwh"]
        if vals is None:
            ax.set_visible(False)
            continue

        ax.plot(vals.index, vals.values, color=color, linewidth=0.8, alpha=0.85, label=label)

        # Mark high-confidence anomalies with red triangles
        if col == "daily_active_kwh":
            for d in anom_dates:
                if d in vals.index:
                    ax.scatter(d, vals[d], marker="v", color=COLORS["anomaly"],
                               s=60, zorder=5, label="_nolegend_")

        # Mark ground truth with vertical orange band
        for d in gt_dates:
            ax.axvspan(d, d + pd.Timedelta(days=1),
                       alpha=0.15, color=COLORS["gt"], zorder=2)

        # MAD bands for residual
        if col == "stl_residual":
            mad = np.median(np.abs(vals.dropna() - vals.dropna().median()))
            ax.axhline(0,        color="gray",         lw=0.8, linestyle="--")
            ax.axhline( mad,     color=COLORS["anomaly"], lw=0.7, linestyle=":", alpha=0.7, label="+MAD")
            ax.axhline(-mad,     color=COLORS["anomaly"], lw=0.7, linestyle=":", alpha=0.7, label="-MAD")
            ax.axhline( 3.5*mad, color=COLORS["anomaly"], lw=0.9, linestyle="--", alpha=0.8, label="±3.5·MAD")
            ax.axhline(-3.5*mad, color=COLORS["anomaly"], lw=0.9, linestyle="--", alpha=0.8)

        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_facecolor(COLORS["background"])

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=COLORS["anomaly"], alpha=0.8, label="Yüksek Güven Anomali"),
        mpatches.Patch(color=COLORS["gt"],      alpha=0.4, label="Ground Truth (rpt-300)"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    savefig(fig, "01_stl_decomposition.png")


# ---------------------------------------------------------------------------
# 2. CUSUM Chart
# ---------------------------------------------------------------------------

def plot_cusum(tier1: pd.DataFrame, gt: pd.DataFrame):
    if "cusum_neg" not in tier1.columns:
        print("  CUSUM data not available — skipping")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(
        "CUSUM Kontrol Grafiği — STL Kalıntısı Üzerine\n"
        "(Alt-taraf CUSUM, Sayaç Kesinti Alarmları)",
        fontsize=12, fontweight="bold"
    )

    # Top: STL residual
    resid = tier1["stl_residual"].fillna(0.0)
    ax1.plot(resid.index, resid.values, color=COLORS["residual"], linewidth=0.8, label="STL Kalıntısı")
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.set_ylabel("STL Kalıntısı (kWh)", fontsize=9)

    # Bottom: S_neg + alarm bands
    s_neg = tier1["cusum_neg"].fillna(0.0)
    ax2.plot(s_neg.index, s_neg.values, color=COLORS["cusum"], linewidth=1.0, label="S⁻(i)")

    # Estimate h threshold from data (5*sigma of residual)
    sigma = resid.std()
    h = 5 * sigma
    ax2.axhline(h, color=COLORS["anomaly"], lw=1.2, linestyle="--", label=f"h = 5σ = {h:.2f}")

    # Shade alarm periods
    in_alarm = s_neg > h
    alarm_starts = s_neg.index[in_alarm & ~in_alarm.shift(1, fill_value=False)]
    alarm_ends   = s_neg.index[in_alarm & ~in_alarm.shift(-1, fill_value=False)]
    for start, end in zip(alarm_starts, alarm_ends):
        ax2.axvspan(start, end + pd.Timedelta(days=1),
                    alpha=0.25, color=COLORS["anomaly"])

    # GT vertical bands on both axes
    gt_dates = gt[gt["outage"] == 1].index if gt is not None else pd.DatetimeIndex([])
    for ax in (ax1, ax2):
        for d in gt_dates:
            ax.axvspan(d, d + pd.Timedelta(days=1),
                       alpha=0.25, color=COLORS["gt"])
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_facecolor(COLORS["background"])
        ax.tick_params(labelsize=8)

    ax2.set_ylabel("S⁻(i)", fontsize=9)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())

    legend_handles = [
        mpatches.Patch(color=COLORS["anomaly"], alpha=0.3, label="CUSUM Alarmı"),
        mpatches.Patch(color=COLORS["gt"],      alpha=0.3, label="Ground Truth"),
    ]
    ax1.legend(handles=legend_handles + [mpatches.Patch(color=COLORS["residual"], label="STL Kalıntısı")],
               fontsize=8, loc="upper right")

    fig.tight_layout()
    savefig(fig, "02_cusum_chart.png")


# ---------------------------------------------------------------------------
# 3. iForest PCA Scatter
# ---------------------------------------------------------------------------

def plot_pca_scatter(tier2: pd.DataFrame, gt: pd.DataFrame, report: pd.DataFrame):
    if "pc1" not in tier2.columns or "pc2" not in tier2.columns:
        print("  PCA data not available — skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(
        "Isolation Forest — PCA Projeksiyonu (2B)\n"
        "Renk: Anomali Skoru, Yıldız: Ground Truth Günleri",
        fontsize=11, fontweight="bold"
    )

    sev = tier2["t2a_iforest_severity"].fillna(0.0)
    sc = ax.scatter(
        tier2["pc1"], tier2["pc2"],
        c=sev, cmap="RdYlGn_r", alpha=0.65, s=20, vmin=0, vmax=1,
        linewidths=0.3, edgecolors="gray"
    )
    plt.colorbar(sc, ax=ax, label="Anomali Skoru")

    # Ground truth days with star markers
    gt_dates = gt[gt["outage"] == 1].index if gt is not None else pd.DatetimeIndex([])
    gt_in_tier2 = tier2.reindex(gt_dates).dropna(subset=["pc1", "pc2"])
    if len(gt_in_tier2) > 0:
        ax.scatter(
            gt_in_tier2["pc1"], gt_in_tier2["pc2"],
            marker="*", color=COLORS["gt"], s=200, zorder=5,
            linewidths=0.5, edgecolors="black", label="Ground Truth (rpt-300)"
        )
        ax.legend(fontsize=9)

    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_facecolor(COLORS["background"])
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    savefig(fig, "03_iforest_pca_scatter.png")


# ---------------------------------------------------------------------------
# 4. Change-Point Timeline
# ---------------------------------------------------------------------------

def plot_changepoint_timeline(tier3: pd.DataFrame, cp_df: pd.DataFrame,
                               report: pd.DataFrame, gt: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title(
        "Değişim Noktası Tespiti (PELT, rbf modeli)\n"
        "Günlük Aktif Enerji + Kırılma Noktaları",
        fontsize=11, fontweight="bold"
    )

    series = report["daily_active_kwh"]
    ax.plot(series.index, series.values, color=COLORS["primary"],
            linewidth=0.9, alpha=0.8, label="Günlük Aktif kWh")

    # Rolling baseline
    baseline = report["baseline_kwh"]
    ax.plot(baseline.index, baseline.values, color=COLORS["trend"],
            linewidth=1.0, linestyle="--", alpha=0.7, label="28-gün kayan medyan")

    # Change points
    if cp_df is not None and not cp_df.empty:
        for _, row in cp_df.iterrows():
            cp = row["changepoint_date"]
            rel = row.get("relative_change") or 0.0
            cp_type = row.get("type", "minor")

            color = COLORS["anomaly"] if cp_type in ("sudden_drop", "partial_fault") else COLORS["trend"]
            ax.axvline(cp, color=color, lw=1.5, linestyle="--", alpha=0.8)

            if cp in series.index:
                y_val = series[cp]
            else:
                y_val = series.mean()

            label_text = f"{rel*100:+.0f}%"
            ax.text(cp, y_val * 1.02, label_text,
                    rotation=90, fontsize=7, color=color, ha="center", va="bottom")

    # Ground truth upper band
    gt_dates = gt[gt["outage"] == 1].index if gt is not None else pd.DatetimeIndex([])
    y_max = series.max() * 1.12
    for d in gt_dates:
        ax.axvspan(d, d + pd.Timedelta(days=1),
                   ymin=0.88, ymax=1.0, alpha=0.6, color=COLORS["gt"])

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS["primary"],  lw=1.0, label="Günlük kWh"),
        Line2D([0], [0], color=COLORS["trend"],    lw=1.0, linestyle="--", label="28-gün Medyan"),
        Line2D([0], [0], color=COLORS["anomaly"],  lw=1.5, linestyle="--", label="Kırılma Noktası (düşüş)"),
        mpatches.Patch(color=COLORS["gt"], alpha=0.6, label="Ground Truth"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower left")
    ax.set_ylabel("kWh", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_facecolor(COLORS["background"])
    fig.tight_layout()
    savefig(fig, "04_changepoint_timeline.png")


# ---------------------------------------------------------------------------
# 5. Calendar Heat Map (month × day)
# ---------------------------------------------------------------------------

def plot_calendar_heatmap(report: pd.DataFrame, gt: pd.DataFrame):
    try:
        import calmap
        _has_calmap = True
    except ImportError:
        _has_calmap = False

    if _has_calmap:
        _plot_calendar_calmap(report, gt)
    else:
        _plot_calendar_manual(report, gt)


def _plot_calendar_calmap(report: pd.DataFrame, gt: pd.DataFrame):
    import calmap
    score_series = report["ensemble_score"].copy()
    score_series.index = pd.DatetimeIndex(score_series.index)

    # Only plot years present in data
    years = score_series.index.year.unique()
    n_years = len(years)

    fig, axes = calmap.calendarplot(
        score_series,
        cmap="YlOrRd",
        fillcolor="lightgrey",
        linewidth=0.5,
        fig_kws={"figsize": (16, 3 * n_years)},
        vmin=0, vmax=1,
    )
    fig.suptitle(
        "Ensemble Anomali Skoru — Takvim Isı Haritası\n"
        "(Siyah çerçeve = Ground Truth kesinti günü)",
        fontsize=11, fontweight="bold", y=1.01
    )

    # Overlay ground truth black borders — calmap axes are complex; skip for now
    # Add colorbar
    sm = ScalarMappable(cmap="YlOrRd", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=axes, orientation="horizontal", pad=0.02, label="Ensemble Anomali Skoru")
    savefig(fig, "05_calendar_heatmap.png")


def _plot_calendar_manual(report: pd.DataFrame, gt: pd.DataFrame):
    """Fallback: month × day-of-month grid."""
    score = report["ensemble_score"].copy()
    months = score.resample("ME").max()  # not used, just for reference

    # Build pivot: rows=months, cols=day_of_month
    df = pd.DataFrame({
        "month":      score.index.to_period("M"),
        "day":        score.index.day,
        "score":      score.values,
    })
    pivot = df.pivot_table(index="month", columns="day", values="score", aggfunc="max")

    fig, ax = plt.subplots(figsize=(18, max(4, len(pivot) * 0.5 + 1)))
    ax.set_title(
        "Ensemble Anomali Skoru — Takvim Isı Haritası\n"
        "(Siyah çerçeve = Ground Truth kesinti günü)",
        fontsize=11, fontweight="bold"
    )

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels([str(p) for p in pivot.index], fontsize=8)
    ax.set_xticks(range(31))
    ax.set_xticklabels(range(1, 32), fontsize=7)
    ax.set_xlabel("Gün", fontsize=10)
    ax.set_ylabel("Ay", fontsize=10)

    # Draw black rectangles for GT days
    gt_dates = gt[gt["outage"] == 1].index if gt is not None else pd.DatetimeIndex([])
    month_list = list(pivot.index)
    for d in gt_dates:
        period = d.to_period("M")
        if period in month_list:
            row_idx = month_list.index(period)
            col_idx = d.day - 1
            rect = plt.Rectangle(
                (col_idx - 0.5, row_idx - 0.5), 1, 1,
                linewidth=2, edgecolor="black", facecolor="none", zorder=5
            )
            ax.add_patch(rect)

    plt.colorbar(im, ax=ax, orientation="vertical", label="Ensemble Anomali Skoru")
    fig.tight_layout()
    savefig(fig, "05_calendar_heatmap.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(tier1=None, tier2=None, tier3=None, cp_df=None, report=None, gt=None):
    print("Loading data for visualization …")

    if report is None:
        report = pd.read_csv(REPORT_CSV, index_col=0, parse_dates=True)

    if gt is None and GT_CSV.exists():
        gt = pd.read_csv(GT_CSV, index_col=0, parse_dates=True)

    if tier1 is None:
        daily = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)
        from tier1_statistical import main as t1_main
        tier1 = t1_main(daily)

    if tier2 is None:
        daily = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)
        from tier2_ml import main as t2_main
        tier2 = t2_main(daily)

    if tier3 is None:
        daily = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)
        from tier3_changepoint import main as t3_main
        t3_result = t3_main(daily)
        tier3 = t3_result[0] if isinstance(t3_result, tuple) else t3_result
        cp_df  = t3_result[1] if isinstance(t3_result, tuple) else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating figures …")

    print("  [1/5] STL Decomposition …")
    plot_stl(tier1, report, gt)

    print("  [2/5] CUSUM Chart …")
    plot_cusum(tier1, gt)

    print("  [3/5] iForest PCA Scatter …")
    plot_pca_scatter(tier2, gt, report)

    print("  [4/5] Change-Point Timeline …")
    plot_changepoint_timeline(tier3, cp_df, report, gt)

    print("  [5/5] Calendar Heat Map …")
    plot_calendar_heatmap(report, gt)

    print(f"All figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
