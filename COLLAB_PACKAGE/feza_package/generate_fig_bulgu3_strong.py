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
CP_DATE  = pd.Timestamp("2025-12-13")

def main():
    daily = pd.read_csv(os.path.join(DATA_DIR, "daily_features.csv"), parse_dates=["Date"])
    gaf   = pd.read_csv(os.path.join(DATA_DIR, "gaf_anomaly_scores.csv"), parse_dates=["Date"])

    # Merge and filter
    df = pd.merge(daily[["Date", "NightMean"]], gaf[["Date", "GAF_Anomaly_Score"]], on="Date")
    df = df[(df["Date"] <= CUTOFF)].sort_values("Date").dropna()

    # Calculate Z-scores
    df["NM_z"] = (df["NightMean"] - df["NightMean"].mean()) / df["NightMean"].std()
    df["GAF_z"] = (df["GAF_Anomaly_Score"] - df["GAF_Anomaly_Score"].mean()) / df["GAF_Anomaly_Score"].std()

    # Smooth Z-scores for clarity (14-day rolling)
    df["NM_z_roll"] = df["NM_z"].rolling(14, min_periods=7).mean()
    df["GAF_z_roll"] = df["GAF_z"].rolling(14, min_periods=7).mean()

    # Calculate Correlation before and after
    df_before = df[df["Date"] < CP_DATE]
    df_after = df[df["Date"] >= CP_DATE]
    corr_before = df_before["NightMean"].corr(df_before["GAF_Anomaly_Score"])
    corr_after = df_after["NightMean"].corr(df_after["GAF_Anomaly_Score"])

    # Setup Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # --- Plot Z-scored NightMean ---
    ax.plot(df["Date"], df["NM_z_roll"], color="#1565C0", lw=2.5, label="NightMean (Z-score, 14-day avg)")
    ax.fill_between(df["Date"], df["NM_z_roll"], 0, where=(df["NM_z_roll"] < 0), color="#1565C0", alpha=0.1)

    # --- Plot Z-scored GAF Score ---
    ax.plot(df["Date"], df["GAF_z_roll"], color="#E65100", lw=2.5, label="GAF Anomaly Score (Z-score, 14-day avg)", ls="--")

    # --- Vertical Line for Change Point ---
    ax.axvline(CP_DATE, color="#c62828", lw=3.0, ls="-", alpha=0.9, zorder=10)
    ax.text(CP_DATE, 2.5, f"  STRUCTURAL SHIFT\n  {CP_DATE.strftime('%Y-%m-%d')}", 
            fontsize=10, color="#c62828", fontweight="bold", va="top")

    # --- Correlation Boxes ---
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#bdbdbd')
    ax.text(pd.Timestamp("2025-06-01"), -1.5, f"Correlation: {corr_before:.3f}\n(Pre-Shift)", 
            transform=ax.transData, fontsize=10, color="#616161", bbox=props, ha="center")
    ax.text(pd.Timestamp("2026-01-15"), 1.0, f"Correlation: {corr_after:.3f}\n(Post-Shift)", 
            transform=ax.transData, fontsize=10, color="#c62828", bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.9, edgecolor='#c62828'), ha="center")

    # --- Seasonal Comparison Annotation ---
    ax.annotate(
        "Non-Seasonal Drop:\nJan-Feb 2026 is 6.1% lower\nthan same period in 2025",
        xy=(pd.Timestamp("2026-01-15"), -1.0),
        xytext=(pd.Timestamp("2025-09-01"), -2.5),
        fontsize=10, color="#2e7d32", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.5, connectionstyle="arc3,rad=.2"),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f5e9", edgecolor="#2e7d32", alpha=0.9)
    )

    # Styling
    ax.set_title("Standardized Comparison (Z-Scores): NightMean Decline vs. GAF Anomaly Growth", fontweight="bold", fontsize=14, pad=20)
    ax.set_ylabel("Standard Deviations from Mean (Z-Score)", fontweight="bold", fontsize=12)
    ax.set_xlabel("Timeline", fontweight="bold", fontsize=12)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)
    
    # Horizontal line at zero
    ax.axhline(0, color="black", lw=1, alpha=0.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=0, ha="center")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig_bulgu3_final_strong.png")
    fig.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
