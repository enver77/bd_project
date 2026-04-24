import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(DATA_DIR, "report_figures")
CUTOFF   = pd.Timestamp("2026-02-13")
CP_DATE  = pd.Timestamp("2025-12-13")  # kalıcı kırılma noktası

def main():
    daily = pd.read_csv(os.path.join(DATA_DIR, "daily_features.csv"), parse_dates=["Date"])
    gaf   = pd.read_csv(os.path.join(DATA_DIR, "gaf_anomaly_scores.csv"), parse_dates=["Date"])

    daily = daily[(daily["Date"] <= CUTOFF) &
                  (daily["data_quality_flag"] != "data_quality_low") &
                  (~daily["AllDayZero"])].sort_values("Date")
    gaf   = gaf[gaf["Date"] <= CUTOFF].sort_values("Date")
    df    = pd.merge(daily[["Date","NightMean"]], gaf[["Date","GAF_Anomaly_Score"]], on="Date")

    # 14-day rolling
    df["NM_roll"]  = df["NightMean"].rolling(14, min_periods=7).mean()
    df["GAF_roll"] = df["GAF_Anomaly_Score"].rolling(14, min_periods=7).mean()

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    # --- NightMean (sol eksen) ---
    ax1.plot(df["Date"], df["NightMean"],   color="#90CAF9", lw=0.7, alpha=0.5)
    ax1.plot(df["Date"], df["NM_roll"],     color="#1565C0", lw=2.2, label="NightMean (14-day avg)")

    # Dönem ortalamaları
    pre  = df[df["Date"] <  CP_DATE]["NightMean"].mean()
    post = df[df["Date"] >= CP_DATE]["NightMean"].mean()
    ax1.axhline(pre,  color="#1565C0", lw=1.2, ls=":", alpha=0.6)
    ax1.axhline(post, color="#1565C0", lw=1.2, ls=":", alpha=0.6)
    ax1.annotate(f"Pre avg: {pre:.2f} kWh",
                 xy=(pd.Timestamp("2025-01-10"), pre+0.07),
                 fontsize=8, color="#1565C0")
    ax1.annotate(f"Post avg: {post:.2f} kWh",
                 xy=(pd.Timestamp("2025-12-15"), post+0.07),
                 fontsize=8, color="#1565C0")

    # --- GAF skoru (sağ eksen) ---
    ax2.plot(df["Date"], df["GAF_Anomaly_Score"], color="#FFCC80", lw=0.6, alpha=0.5)
    ax2.plot(df["Date"], df["GAF_roll"],          color="#E65100", lw=2.0,
             label="GAF Anomaly Score (14-day avg)", ls="--")

    pre_gaf  = df[df["Date"] <  CP_DATE]["GAF_Anomaly_Score"].mean()
    post_gaf = df[df["Date"] >= CP_DATE]["GAF_Anomaly_Score"].mean()

    # --- Kırılma çizgisi ---
    ax1.axvline(CP_DATE, color="#c62828", lw=2.0, ls="--", alpha=0.9)
    ax1.text(CP_DATE, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 5 else 8,
             f" Change Point\n {CP_DATE.strftime('%Y-%m-%d')}",
             fontsize=8.5, color="#c62828", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#c62828", alpha=0.9))

    # Dönem renklendirmesi
    ax1.axvspan(df["Date"].min(), CP_DATE, alpha=0.04, color="#1565C0")
    ax1.axvspan(CP_DATE, CUTOFF,           alpha=0.07, color="#c62828")

    # Annotasyon: yüzde değişim
    ax1.annotate(
        f"NightMean: {pre:.2f} → {post:.2f} kWh\n({(post-pre)/pre*100:+.1f}%)",
        xy=(CP_DATE, (pre+post)/2),
        xytext=(pd.Timestamp("2025-10-01"), 5.2),
        fontsize=8.5, color="#c62828",
        arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0",
                  edgecolor="#e65100", alpha=0.95)
    )
    ax2.annotate(
        f"GAF avg: {pre_gaf:.3f} → {post_gaf:.3f}\n({(post_gaf-pre_gaf)/pre_gaf*100:+.1f}%)",
        xy=(CP_DATE, post_gaf),
        xytext=(pd.Timestamp("2025-10-01"), 0.08),
        fontsize=8.5, color="#E65100",
        arrowprops=dict(arrowstyle="->", color="#E65100", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3e0",
                  edgecolor="#e65100", alpha=0.95)
    )

    # Eksen ayarları
    ax1.set_ylabel("NightMean (kWh)", color="#1565C0", fontweight="bold", fontsize=11)
    ax2.set_ylabel("GAF Anomaly Score", color="#E65100", fontweight="bold", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#1565C0")
    ax2.tick_params(axis="y", labelcolor="#E65100")
    ax1.set_ylim(0, 10)
    ax2.set_ylim(0, 0.25)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax1.grid(True, ls="--", alpha=0.25)

    # Legend
    h1 = mpatches.Patch(color="#1565C0", label="NightMean (14-day avg)")
    h2 = mpatches.Patch(color="#E65100", label="GAF Anomaly Score (14-day avg)")
    h3 = mpatches.Patch(color="#c62828", label="Structural Change Point (2025-12-13)", alpha=0.6)
    ax1.legend(handles=[h1, h2, h3], fontsize=9, loc="upper left", framealpha=0.9)

    ax1.set_title("Convergence of Two Independent Signals: NightMean Decline and GAF Score Elevation\naround the Structural Change Point (December 2025)",
                  fontweight="bold", fontsize=12)
    ax1.set_xlabel("Date", fontweight="bold", fontsize=11)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_bulgu3_dual_signal.png")
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
