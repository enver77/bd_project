import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(DATA_DIR, "report_figures")
CUTOFF   = pd.Timestamp("2026-02-13")

def main():
    gaf   = pd.read_csv(os.path.join(DATA_DIR, "gaf_anomaly_scores.csv"), parse_dates=["Date"])
    gaf   = gaf[gaf["Date"] <= CUTOFF]

    df300 = pd.read_excel(os.path.join(DATA_DIR,
            "rpt-300_sayac_kesinti_raporu_(kesinti_bazli)_0XldY.xlsx"), sheet_name=0)
    df300["Date"] = pd.to_datetime(df300["Başlangıç Tarihi"], dayfirst=True).dt.normalize()
    outage_days = set(df300["Date"].dt.date)

    gaf["Group"] = gaf["Date"].dt.date.apply(
        lambda d: "Outage Day\n(n=7)" if d in outage_days else "Non-Outage Day\n(n=400)")

    outage_scores     = gaf[gaf["Group"].str.startswith("Outage")]["GAF_Anomaly_Score"]
    non_outage_scores = gaf[gaf["Group"].str.startswith("Non")]["GAF_Anomaly_Score"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [1, 1.8]})

    # --- Sol panel: Box plot karşılaştırması ---
    ax = axes[0]
    bp = ax.boxplot(
        [outage_scores.values, non_outage_scores.values],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="#c62828", linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.5)
    )
    colors = ["#EF9A9A", "#90CAF9"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Ortalama noktaları
    means = [outage_scores.mean(), non_outage_scores.mean()]
    ax.scatter([1, 2], means, marker="D", color=["#b71c1c","#1565C0"],
               zorder=5, s=60, label="Mean")

    # Değer annotasyonları
    for i, (m, med) in enumerate(zip(means, [outage_scores.median(), non_outage_scores.median()])):
        ax.text(i+1, m+0.003, f"μ={m:.3f}", ha="center", fontsize=8.5,
                color=["#b71c1c","#1565C0"][i], fontweight="bold")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Outage Days\n(n=7)", "Non-Outage Days\n(n=400)"], fontsize=10)
    ax.set_ylabel("GAF Anomaly Score", fontweight="bold", fontsize=11)
    ax.set_title("GAF Score Distribution:\nOutage vs Non-Outage Days", fontweight="bold", fontsize=11)
    ax.set_ylim(-0.01, 0.18)
    ax.grid(True, ls="--", alpha=0.35, axis="y")

    red_p   = mpatches.Patch(color="#EF9A9A", label="Outage Days")
    blue_p  = mpatches.Patch(color="#90CAF9", label="Non-Outage Days")
    dia     = mpatches.Patch(color="gray", label="◆ Mean")
    ax.legend(handles=[red_p, blue_p], fontsize=8.5, loc="upper right")

    # --- Sağ panel: Scatter - her kesinti günü olayı ---
    ax2 = axes[1]
    # Tüm günler
    non_out = gaf[gaf["Group"].str.startswith("Non")]
    ax2.scatter(non_out["Date"], non_out["GAF_Anomaly_Score"],
                color="#90CAF9", s=18, alpha=0.4, label="Non-Outage Days", zorder=2)

    # Kesinti günleri
    out_df = pd.merge(
        gaf[gaf["Group"].str.startswith("Outage")],
        df300[["Date","Kesinti Süre (dk)"]].groupby("Date").sum().reset_index(),
        on="Date", how="left"
    )
    sc = ax2.scatter(out_df["Date"], out_df["GAF_Anomaly_Score"],
                     color="#c62828", s=out_df["Kesinti Süre (dk)"] * 0.8 + 60,
                     zorder=5, edgecolors="white", linewidths=0.8,
                     label="Outage Days (size ∝ duration)")
    for _, r in out_df.iterrows():
        ax2.annotate(f'{r["Kesinti Süre (dk)"]:.0f} min',
                     xy=(r["Date"], r["GAF_Anomaly_Score"]),
                     xytext=(0, 12), textcoords="offset points",
                     fontsize=7.5, ha="center", color="#b71c1c",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                               edgecolor="#c62828", alpha=0.85))

    ax2.set_title("GAF Anomaly Score Timeline:\nOutage Events Highlighted",
                  fontweight="bold", fontsize=11)
    ax2.set_xlabel("Date", fontweight="bold", fontsize=11)
    ax2.set_ylabel("GAF Anomaly Score", fontweight="bold", fontsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8.5)
    ax2.set_ylim(-0.01, 0.22)
    ax2.grid(True, ls="--", alpha=0.3)
    ax2.legend(fontsize=8.5, loc="upper right")

    plt.suptitle(
        "Officially Recorded Outage Days Yield Low GAF Anomaly Scores",
        fontweight="bold", fontsize=13, y=1.01
    )
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_bulgu2_outage_vs_gaf.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
