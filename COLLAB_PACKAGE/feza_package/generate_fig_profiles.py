import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(DATA_DIR, "report_figures")
CUTOFF = pd.Timestamp("2026-02-13")

# Days to plot: (date, label, color, linestyle)
DAYS = [
    ("2025-09-28", "Normal Day (Score: 0.000)",    "#2e7d32", "-",  2.5),
    ("2025-01-06", "Anomalous #2 (Score: 0.720)",  "#e65100", "--", 2.0),
    ("2025-01-04", "Anomalous #1 (Score: 1.000)",  "#b71c1c", "-",  2.5),
]

def main():
    master = pd.read_csv(os.path.join(DATA_DIR, "master_imputed.csv"), parse_dates=["Date"])
    master = master[master["Date"] <= CUTOFF]

    fig, ax = plt.subplots(figsize=(12, 5))
    hours = np.arange(24)

    for date_str, label, color, ls, lw in DAYS:
        dt = pd.Timestamp(date_str)
        day_data = master[master["Date"] == dt].sort_values("Hour")
        if len(day_data) == 24:
            ax.plot(hours, day_data["Consumption_kWh"].values,
                    color=color, linestyle=ls, linewidth=lw, label=label, marker="o",
                    markersize=4, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.5)

    ax.set_title("Daily Load Profile Comparison: Normal vs Anomalous Days",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Hour of Day", fontweight="bold", fontsize=11)
    ax.set_ylabel("Consumption (kWh)", fontweight="bold", fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(-0.3, 9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    # Shade night hours
    ax.axvspan(-0.5, 5.5, alpha=0.06, color="#1565C0", label="_Night")
    ax.axvspan(19.5, 23.5, alpha=0.06, color="#1565C0", label="_Night2")
    ax.axvspan(8.5, 16.5, alpha=0.06, color="#FFF9C4", label="_Day")

    ax.text(2.5, 8.5, "Night", fontsize=8, color="#1565C0", ha="center", style="italic")
    ax.text(12.5, 0.3, "Daytime (Off)", fontsize=8, color="#F57F17", ha="center", style="italic")
    ax.text(21.5, 8.5, "Night", fontsize=8, color="#1565C0", ha="center", style="italic")

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_profile_comparison.png")
    fig.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
