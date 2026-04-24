import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(DATA_DIR, "report_figures")
CUTOFF = pd.Timestamp("2026-02-13")

def main():
    daily = pd.read_csv(os.path.join(DATA_DIR, "daily_features.csv"), parse_dates=["Date"])
    daily = daily[daily["Date"] <= CUTOFF].sort_values("Date")

    # Exclude data_quality_low and AllDayZero
    daily = daily[(daily["data_quality_flag"] != "data_quality_low") & (~daily["AllDayZero"])]

    # Load changepoints — only Persistent=True NightMean to keep graph readable
    cp_path = os.path.join(DATA_DIR, "changepoints_filtered.csv")
    changepoints = []
    if os.path.exists(cp_path):
        cp = pd.read_csv(cp_path, parse_dates=["ChangePoint_Date"])
        cp = cp[(cp["ChangePoint_Date"] <= CUTOFF) &
                (cp["Persistent"] == True) &
                (cp["Series"] == "NightMean")]
        changepoints = list(zip(cp["ChangePoint_Date"], cp["Change_kWh"]))

    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot NightMean
    ax.plot(daily["Date"], daily["NightMean"], color="#90CAF9",
            linewidth=0.8, alpha=0.7, label="Daily NightMean")

    # 14-day rolling mean
    daily["NightMean_14d"] = daily["NightMean"].rolling(14, min_periods=7).mean()
    ax.plot(daily["Date"], daily["NightMean_14d"], color="#1565C0",
            linewidth=2.2, label="14-day Rolling Mean")

    # Change points — annotate with date and change amount
    for i, (cp_date, change_kwh) in enumerate(changepoints):
        ax.axvline(cp_date, color="#c62828", linewidth=2.0,
                   linestyle="--", alpha=0.85,
                   label="NightMean Change Point" if i == 0 else "_")
        direction = "▼" if change_kwh < 0 else "▲"
        label_y = daily["NightMean"].max() * 0.95 if i % 2 == 0 else daily["NightMean"].max() * 0.82
        ax.text(cp_date, label_y,
                f"{direction}{abs(change_kwh):.2f} kWh\n{cp_date.strftime('%Y-%m-%d')}",
                fontsize=8.5, color="#b71c1c", rotation=0, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#c62828", alpha=0.85))

    # Global median line
    global_median = daily["NightMean"].median()
    ax.axhline(global_median, color="#388e3c", linewidth=1.2,
               linestyle=":", alpha=0.8, label=f"Global Median ({global_median:.2f} kWh)")

    ax.set_title("NightMean Consumption Over Time with Change Points",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Date", fontweight="bold", fontsize=11)
    ax.set_ylabel("NightMean (kWh)", fontweight="bold", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.set_ylim(0, daily["NightMean"].max() * 1.1)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_nightmean_changepoints.png")
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
