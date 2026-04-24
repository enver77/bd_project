import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(DATA_DIR, "report_figures")
CUTOFF = pd.Timestamp("2026-02-13")

def main():
    gaf = pd.read_csv(os.path.join(DATA_DIR, "gaf_anomaly_scores.csv"), parse_dates=["Date"])
    gaf = gaf[gaf["Date"] <= CUTOFF].copy()
    gaf["YearMonth"] = gaf["Date"].dt.to_period("M").astype(str)

    months = sorted(gaf["YearMonth"].unique())
    data = [gaf[gaf["YearMonth"] == m]["GAF_Anomaly_Score"].values for m in months]

    fig, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(data, tick_labels=months, patch_artist=True, widths=0.6,
                    medianprops=dict(color="#e65100", linewidth=2),
                    flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="black", markersize=6))

    for patch in bp["boxes"]:
        patch.set_facecolor("#90CAF9")
        patch.set_edgecolor("#1565C0")

    # Y eksenini 0.30'da kesip, outlier'ları ayrıca yazıyla gösterelim
    y_cutoff = 0.30
    ax.set_ylim(-0.01, y_cutoff)
    ax.set_title("Monthly Distribution of GAF Anomaly Scores", fontweight="bold", fontsize=14)
    ax.set_xlabel("Month", fontweight="bold", fontsize=12)
    ax.set_ylabel("GAF Anomaly Score", fontweight="bold", fontsize=12)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

    # Kesilen outlier'ları etiket ile göster
    outliers = {
        "2025-01": [1.0000, 0.7202],
        "2025-02": [0.4879],
        "2025-03": [0.4936, 0.3725],
    }
    for month_str, values in outliers.items():
        idx = months.index(month_str) + 1  # boxplot 1-indexed
        above = sorted([v for v in values if v > y_cutoff], reverse=True)
        for j, val in enumerate(above):
            y_pos = y_cutoff - 0.008 - j * 0.022
            ax.annotate(f"{val:.2f}", xy=(idx, y_pos),
                        fontsize=8, fontweight="bold", ha="center", color="#c62828",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3e0", edgecolor="#e65100", alpha=0.9))

    ax.text(0.98, 0.97, f"▲ Outliers above {y_cutoff} shown as labels",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            style="italic", color="#666")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig4_monthly_dist.png")
    fig.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
