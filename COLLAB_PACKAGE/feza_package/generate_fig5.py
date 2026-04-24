import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(DATA_DIR, "report_figures")
os.makedirs(FIG_DIR, exist_ok=True)
CUTOFF = pd.Timestamp("2026-02-13")

def main():
    gaf = pd.read_csv(os.path.join(DATA_DIR, "gaf_anomaly_scores.csv"), parse_dates=["Date"])
    deg = pd.read_csv(os.path.join(DATA_DIR, "predictive_risk_scores.csv"), parse_dates=["Date"])

    # Filter with valid period CUTOFF
    gaf = gaf[gaf["Date"] <= CUTOFF].copy()
    deg = deg[deg["Date"] <= CUTOFF].copy()

    # Merge on Date
    df = pd.merge(gaf, deg, on="Date", how="inner")
    
    # Scatter plot showing GAF Anomaly Score vs Degradation Score
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color points based on RiskLevel
    risk_colors = {
        "HIGH_RISK": "#d32f2f",      # Red
        "MEDIUM_RISK": "#f57c00",    # Orange
        "LOW_RISK": "#fbc02d",       # Yellow
        "NORMAL": "#388e3c",         # Green
    }
    
    for risk, color in risk_colors.items():
        subset = df[df["RiskLevel"] == risk]
        if not subset.empty:
            ax.scatter(subset["DegradationScore"], subset["GAF_Anomaly_Score"], 
                       c=color, label=risk, s=60, alpha=0.7, edgecolor='white')

    ax.set_title("GAF Anomaly Score vs. Degradation Score", fontweight="bold")
    ax.set_xlabel("Degradation Score (Based on NightMean & Flags)", fontweight="bold")
    ax.set_ylabel("GAF Anomaly Score", fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Risk Level", loc="upper left")

    out_path = os.path.join(FIG_DIR, "fig5_gaf_vs_degradation_updated.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Updated figure saved to {out_path}")

if __name__ == "__main__":
    main()
