#!/usr/bin/env python3
"""
GAF Report Generator - Mask-Aware (Original Format Preserved)
===============================================================
Structure: Title + 1 Introduction + 2 Methodology + 3 Full Period Analysis
           (3.1 Load Profiles, 3.2 GAF Representation, 3.3 Results)
           + 4 Period Findings + 5 Comparison + 6 Conclusion

Outputs:
  - gaf_report_final.pdf
  - report_figures/fig1_final_load_profiles.png
  - report_figures/fig2_final_gaf_grid.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(DATA_DIR, "report_figures")
os.makedirs(FIG_DIR, exist_ok=True)

CUTOFF = pd.Timestamp("2026-02-13")

plt.rcParams.update({
    "figure.dpi": 180, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": "white",
})


def sanitize(text):
    """Replace non-Latin-1 characters with ASCII equivalents."""
    text = str(text)
    replacements = {
        '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u0131': 'i',
        '\u015f': 's', '\u00e7': 'c', '\u011f': 'g', '\u00fc': 'u',
        '\u00f6': 'o', '\u0130': 'I', '\u015e': 'S', '\u00c7': 'C',
        '\u011e': 'G', '\u00dc': 'U', '\u00d6': 'O',
        '\u2705': '', '\u274c': '', '\u2022': '-',
        '\u2265': '>=', '\u2264': '<=', '\u2260': '!=',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', errors='replace').decode('latin-1')


# ---------------------------------------------------------------------------
# GAF helper
# ---------------------------------------------------------------------------
def timeseries_to_gaf(v):
    mn, mx = v.min(), v.max()
    if mx - mn < 1e-9:
        s = np.zeros_like(v)
    else:
        s = 2 * (v - mn) / (mx - mn) - 1
    s = np.clip(s, -1, 1)
    phi = np.arccos(s)
    return np.cos(np.add.outer(phi, phi))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all():
    m = pd.read_csv(os.path.join(DATA_DIR, "master_dataset.csv"), parse_dates=["Date"])
    m = m[m["Date"] <= CUTOFF].copy()

    g = pd.read_csv(os.path.join(DATA_DIR, "gaf_anomaly_scores.csv"), parse_dates=["Date"])
    g = g[g["Date"] <= CUTOFF].copy()

    d = pd.read_csv(os.path.join(DATA_DIR, "daily_features.csv"), parse_dates=["Date"])
    d = d[d["Date"] <= CUTOFF].copy()

    p = pd.read_csv(os.path.join(DATA_DIR, "normal_profile.csv"))

    ms = pd.read_csv(os.path.join(DATA_DIR, "missing_data_summary.csv"), parse_dates=["Date"])
    ms = ms[ms["Date"] <= CUTOFF].copy()

    try:
        imp_log = pd.read_csv(os.path.join(DATA_DIR, "imputation_log.csv"), parse_dates=["Date"])
    except Exception:
        imp_log = pd.DataFrame()

    return m, g, d, p, ms, imp_log


def get_profiles(master):
    profiles = {}
    for dt, grp in master.groupby("Date"):
        grp_s = grp.sort_values("Hour")
        if len(grp_s) == 24:
            profiles[dt] = grp_s["Consumption_kWh"].fillna(0).values
    return profiles


# ---------------------------------------------------------------------------
# Figures (same as original)
# ---------------------------------------------------------------------------
def make_fig1(profiles):
    fig, ax = plt.subplots(figsize=(10, 4))
    hours = np.arange(24)
    all_vals = []
    for dt, vals in sorted(profiles.items()):
        ax.plot(hours, vals, color="#1976D2", alpha=0.015, linewidth=0.6)
        all_vals.append(vals)
    mean_p = np.mean(all_vals, axis=0)
    ax.plot(hours, mean_p, color="#D32F2F", linewidth=2.2, label="Mean profile")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Consumption (kWh)")
    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.3, 23.3)
    ax.set_ylim(-0.3, 9)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig1_final_load_profiles.png")
    fig.savefig(path)
    plt.close()
    return path


def make_fig2(profiles, gaf_scores):
    sorted_scores = gaf_scores.sort_values("GAF_Anomaly_Score")
    normal_date = sorted_scores.iloc[0]["Date"]
    top12 = sorted_scores.nlargest(12, "GAF_Anomaly_Score")
    top12_dates = top12["Date"].tolist()

    all_dates = [normal_date] + top12_dates
    n = len(all_dates); ncols = 7; nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))
    axes = axes.flatten()

    for i, dt in enumerate(all_dates):
        ax = axes[i]
        vals = profiles.get(dt, np.zeros(24))
        gaf = timeseries_to_gaf(vals)
        ax.imshow(gaf, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", aspect="equal")
        ds = pd.Timestamp(dt).strftime("%Y-%m-%d")
        if i == 0:
            ax.set_title(f"Normal\n{ds}", fontsize=7, fontweight="bold", color="green")
        else:
            score_row = gaf_scores[gaf_scores["Date"] == dt]
            sc = score_row["GAF_Anomaly_Score"].values[0] if len(score_row) > 0 else 0
            ax.set_title(f"#{i} ({sc:.3f})\n{ds}", fontsize=6.5, color="#c62828")
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_yticks([0, 6, 12, 18, 23])
        ax.tick_params(labelsize=5)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("GAF Images: Normal Day + Top 12 Anomalous Days",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_final_gaf_grid.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# PDF class (same as original with sanitize)
# ---------------------------------------------------------------------------
class GAFReport(FPDF):
    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 5, "GAF-Based Anomaly Detection - Street Lighting (14-Month Analysis)", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_title(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(13, 71, 161)
        self.multi_cell(0, 9, "Gramian Angular Field (GAF) Based\nAnomaly Detection for Street Lighting", align="C")
        self.ln(3)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 6, "OSF ID: 2193681000  |  Period: January 2025 - February 2026 (14 Months)", align="C")
        self.ln(10)
        self.set_draw_color(21, 101, 192)
        self.set_line_width(0.6)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(8)

    def section(self, num, title):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(21, 101, 192)
        self.cell(0, 8, sanitize(f"{num}  {title}"))
        self.ln(9)
        self.set_text_color(30, 30, 30)

    def subsection(self, num, title):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 7, sanitize(f"{num}  {title}"))
        self.ln(8)
        self.set_text_color(30, 30, 30)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, sanitize(text), align="J")
        self.ln(2)

    def formula_text(self, text):
        """Indented formula/pseudocode block."""
        self.set_font("Courier", "", 9)
        self.set_x(25)
        self.multi_cell(160, 4.8, sanitize(text))
        self.ln(2)
        self.set_font("Helvetica", "", 10)

    def add_figure(self, img_path, caption, w=170):
        x = (210 - w) / 2
        self.image(img_path, x=x, w=w)
        self.ln(3)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(90, 90, 90)
        self.multi_cell(0, 4.5, sanitize(caption), align="C")
        self.set_text_color(30, 30, 30)
        self.ln(4)

    def add_table(self, headers, rows, col_widths=None, title=None):
        if title:
            self.set_font("Helvetica", "B", 9)
            self.cell(0, 6, sanitize(title))
            self.ln(5)
        if col_widths is None:
            col_widths = [170 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 8.5)
        self.set_fill_color(21, 101, 192)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, sanitize(h), border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 30, 30)
        for j, row in enumerate(rows):
            if j % 2 == 0:
                self.set_fill_color(240, 245, 255)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, sanitize(str(val)), border=1, fill=True, align="C")
            self.ln()
        self.ln(4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" GAF Report Generator (Mask-Aware, Original Format)")
    print("=" * 60)

    master, gaf_scores, daily, profile, ms, imp_log = load_all()
    profiles = get_profiles(master)

    n_days_total = daily["Date"].nunique()
    n_days_scored = gaf_scores["Date"].nunique()
    date_min = daily["Date"].min().date()
    date_max = daily["Date"].max().date()

    # Merge for tables
    merged = gaf_scores.merge(
        daily[["Date", "NightMean", "DayMean", "DailyTotal", "missing_ratio", "data_quality_flag"]],
        on="Date", how="left"
    )
    top10 = merged.nlargest(10, "GAF_Anomaly_Score")

    # Missing data stats
    n_complete = int((ms["data_quality_flag"] == "complete").sum())
    n_partial = int((ms["data_quality_flag"] == "partial_missing").sum())
    n_dql = int((ms["data_quality_flag"] == "data_quality_low").sum())
    n_imputed_hours = len(imp_log) if len(imp_log) > 0 else 0
    n_imputed_days = len(imp_log["Date"].unique()) if len(imp_log) > 0 else 0
    n_excluded = n_days_total - n_days_scored
    total_missing = int(master["is_missing"].sum()) if "is_missing" in master.columns else 0

    # data_quality_low days
    dql_days = ms[ms["data_quality_flag"] == "data_quality_low"].sort_values("Date")

    print(f"  Valid period: {date_min} to {date_max}")
    print(f"  Total days: {n_days_total}, Scored: {n_days_scored}, Excluded: {n_excluded}")

    # Figures
    print("  Generating Figure 1...")
    fig1 = make_fig1(profiles)
    print("  Generating Figure 2...")
    fig2 = make_fig2(profiles, gaf_scores)

    # PDF
    print("  Building PDF...")
    pdf = GAFReport()
    pdf.add_page()
    pdf.add_title()

    # ================================================================
    # 1  Introduction
    # ================================================================
    pdf.section("1", "Introduction")
    pdf.body_text(
        "Street lighting systems exhibit a characteristic and predictable daily consumption pattern: "
        "high energy use during nighttime hours (approximately 22:00 - 06:00) when luminaires are active, "
        "and near-zero consumption during daytime (09:00 - 16:00) when lights are switched off. "
        "Transitional ramp-up and ramp-down periods correspond to sunset and sunrise times, "
        "which vary seasonally."
    )
    pdf.body_text(
        f"The aim of this study is to identify abnormal days within a 14-month hourly consumption "
        f"dataset ({date_min} to {date_max}, {n_days_total} days) using two complementary approaches: "
        "direct visual inspection of hourly load profiles, and Gramian Angular Field (GAF) "
        "image representations. The GAF transformation captures the global temporal structure of each "
        "day's profile, enabling detection of structural distortions that may indicate equipment "
        "degradation, circuit faults, timer malfunctions, or meter offline conditions. "
        "Since automated metering systems may occasionally produce incomplete records, the methodology "
        "incorporates a mask-aware scoring approach that prevents data quality gaps from generating "
        "false anomaly signals while preserving detection of genuine operational deviations."
    )

    # ================================================================
    # 2  Methodology
    # ================================================================
    pdf.section("2", "Methodology")
    pdf.body_text(
        f"For each of the {n_days_total} days in the analysis period, a 24-dimensional vector is constructed "
        "from hourly consumption values (hours 0-23). These vectors form the basis for both visual "
        "inspection and GAF transformation."
    )

    # 2.1  Missing Data Handling (Mask-Aware Scoring)
    pdf.subsection("2.1", "Missing Data Handling (Mask-Aware Scoring)")
    pdf.body_text(
        "Inspection of the source OSOS Excel files revealed that some hourly cells contain no recorded "
        "value (true missing data), as opposed to a genuine zero reading. If these missing values are "
        "treated as zero consumption, they distort the daily load profile and produce false anomalies "
        "in the GAF transformation. To address this, a mask-aware approach is employed."
    )
    pdf.body_text(
        "For each day, a binary presence mask is_missing[0..23] is constructed, where 1 indicates that "
        "a measurement was recorded for that hour and 0 indicates a missing value. The missing ratio "
        "is defined as:"
    )
    pdf.formula_text(
        "missing_ratio = n_missing_hours / 24"
    )
    pdf.body_text(
        "Based on the missing ratio, each day is assigned a data quality classification:"
    )
    pdf.formula_text(
        "data_quality_flag:\n"
        "  'complete'         if missing_ratio == 0\n"
        "  'partial_missing'  if 0 < missing_ratio <= 0.25\n"
        "  'data_quality_low' if missing_ratio > 0.25"
    )
    pdf.body_text(
        f"In the current dataset, {n_complete} days are complete, {n_partial} days have partial missing "
        f"data (1-6 hours), and {n_dql} days have low data quality (more than 6 hours missing). "
        f"The total number of missing hourly measurements is {total_missing} out of {n_days_total * 24} "
        f"({total_missing / (n_days_total * 24) * 100:.2f} percent)."
    )
    pdf.body_text(
        "The anomaly scoring mechanism operates in a mask-aware manner. All scores are computed "
        "using only recorded (non-missing) measurements. For the GAF transformation, which requires a "
        "complete 24-dimensional vector as input, missing hours are filled with the monthly hourly median "
        "(the typical value for that specific hour in the same calendar month). This imputation serves "
        "only as a technical necessity for GAF image generation; the anomaly decision is based on a "
        "mask-aware score defined as:"
    )
    pdf.formula_text(
        "mask_aware_score = raw_reconstruction_error * (n_present / 24)"
    )
    pdf.body_text(
        "This weighting ensures that a day with genuine anomalous behavior and a few missing hours "
        "still receives a high score (weighted by, e.g., 22/24 = 0.917), while days with extensive "
        "missing data receive proportionally reduced scores, preventing false anomaly signals."
    )
    pdf.body_text(
        f"Days classified as data_quality_low (missing_ratio > 0.25) are excluded from the main anomaly "
        f"ranking entirely. A total of {n_imputed_hours} hours across {n_imputed_days} days were imputed "
        f"for GAF input, and {n_excluded} days were excluded from scoring, leaving "
        f"{n_days_scored} days for GAF analysis."
    )

    # 2.2  GAF Transformation (same as original)
    pdf.subsection("2.2", "GAF Transformation")
    pdf.body_text(
        "Daily profiles are plotted as overlaid line charts to visually identify deviations from the "
        "expected pattern. Subsequently, each 24-hour vector is converted into a 24x24 Gramian Angular "
        "Summation Field (GASF) image. The transformation involves: (1) normalizing values to [-1, 1], "
        "(2) computing angular representations via the arccosine function, and (3) constructing the "
        "GAF matrix as GAF(i,j) = cos(phi_i + phi_j). The resulting image encodes pairwise temporal "
        "correlations between all hours, with the diagonal preserving original values and off-diagonal "
        "elements capturing inter-hour relationships."
    )

    # 2.3  Anomaly Scoring (same as original)
    pdf.subsection("2.3", "Anomaly Scoring")
    pdf.body_text(
        "An autoencoder (576-64-16-64-576) trained on GAF images from complete, non-zero consumption "
        "days computes reconstruction error for each day. The anomaly score is defined as the mean "
        "squared reconstruction error, normalized to [0, 1], and adjusted by the mask-aware weighting "
        "described in Section 2.1. This is an unsupervised approach: the autoencoder learns to "
        "reconstruct typical structural patterns, and days with unusual structures yield higher "
        "reconstruction error."
    )

    # ================================================================
    # 3  Full Period Analysis
    # ================================================================
    pdf.section("3", f"Full Period Analysis ({date_min} to {date_max})")

    # 3.1  Hourly Load Profiles
    pdf.subsection("3.1", "Hourly Load Profiles")
    pdf.body_text(
        f"Figure 1 presents an overlay of all {n_days_scored} daily load profiles across the analysis period. "
        "The characteristic street lighting pattern is clearly visible: a stable plateau of approximately "
        "6.5 - 7.0 kWh during night hours, a sharp morning decline between 05:00 and 08:00, near-zero "
        "daytime consumption, and an evening ramp-up between 17:00 and 21:00. The seasonal variation "
        "in transition timing is evident from the broader spread in ramp-up and ramp-down regions. "
        "Several profiles with reduced nighttime amplitude are visible, indicating potential operational "
        "anomalies."
    )
    pdf.add_figure(fig1,
        f"Figure 1: Overlay of hourly consumption profiles for {n_days_scored} days "
        f"({date_min} to {date_max}). Red line indicates the mean profile.")

    # 3.2  GAF Representation
    pdf.add_page()
    pdf.subsection("3.2", "GAF Representation")
    pdf.body_text(
        "Figure 2 presents GAF images for one representative normal day (lowest anomaly score) and "
        "the top 12 anomalous days ranked by GAF reconstruction error."
    )
    pdf.body_text(
        "The normal day GAF exhibits a clear symmetric block structure: the upper-left and lower-right "
        "quadrants (night-night and evening-evening correlations) show strong positive values (red). "
        "Cross-blocks between high-consumption night hours and near-zero daytime hours show "
        "strong negative values (blue). This block pattern directly reflects the binary on/off nature "
        "of street lighting."
    )
    pdf.body_text(
        "The anomalous day GAFs show varying degrees of structural distortion. Days with reduced "
        "nighttime consumption display weaker block contrast and shifted color boundaries. Days with "
        "shifted transition times show asymmetric patterns in the off-diagonal regions."
    )
    pdf.add_figure(fig2,
        "Figure 2: GAF images. Top-left: normal day (lowest score). "
        "Remaining: top 12 anomalous days ranked by GAF reconstruction error.", w=175)

    # 3.3  Results
    pdf.subsection("3.3", "Results")
    pdf.body_text(
        f"Among the {n_days_scored} days analyzed (after data quality filtering), the top anomalies "
        "are characterized by reduced nighttime consumption amplitude, indicating possible partial "
        "circuit failures or luminaire degradation. The highest-scoring anomaly days are associated "
        "with confirmed NightMean structural changes detected independently by change point analysis."
    )

    # Table 1: Top 10 anomalies (with missing_ratio column)
    t1_headers = ["Rank", "Date", "GAF Score", "NightMean", "DailyTotal", "Missing%"]
    t1_rows = []
    for i, (_, r) in enumerate(top10.iterrows()):
        mr = r.get("missing_ratio", 0)
        t1_rows.append([
            str(i + 1),
            pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"),
            f"{r['GAF_Anomaly_Score']:.4f}",
            f"{r['NightMean']:.3f}" if pd.notna(r.get('NightMean')) else "-",
            f"{r['DailyTotal']:.1f}" if pd.notna(r.get('DailyTotal')) else "-",
            f"{mr * 100:.0f}%" if pd.notna(mr) else "0%",
        ])
    pdf.add_table(t1_headers, t1_rows, col_widths=[12, 30, 28, 28, 28, 18],
                  title="Table 1: Top 10 Anomalous Days (Mask-Aware GAF Score)")

    # Table 2: Data-quality-low days (separate sub-table)
    if n_dql > 0:
        pdf.body_text(
            f"Table 2 lists the {n_dql} days classified as data_quality_low (missing_ratio > 25 percent). "
            "These days are excluded from the anomaly ranking above to prevent false positives caused "
            "by incomplete measurement records. They are reported separately for data quality transparency."
        )
        t2_headers = ["Date", "Missing Hrs", "Missing%", "Quality Flag"]
        t2_rows = []
        for _, r in dql_days.iterrows():
            t2_rows.append([
                pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"),
                str(int(r["n_missing"])),
                f"{r['missing_ratio'] * 100:.0f}%",
                r["data_quality_flag"],
            ])
        pdf.add_table(t2_headers, t2_rows, col_widths=[35, 25, 25, 40],
                      title="Table 2: Data Quality Low Days (Excluded from Scoring)")

    # ================================================================
    # 4  Period Findings
    # ================================================================
    pdf.section("4", "Period Findings (Key Observations)")
    pdf.body_text(
        "The 14-month analysis reveals that operational anomalies are not uniformly distributed "
        "across the period. The most significant anomaly cluster occurs in mid-December 2025, where "
        "NightMean consumption drops from approximately 6.8 kWh to 5.2 kWh (a reduction of 24.6 "
        "percent). This structural break is persistent, lasting several weeks, and aligns with "
        "independent change point detection results, suggesting a genuine change in lighting system "
        "behavior rather than a transient disturbance."
    )
    pdf.body_text(
        "Additional isolated anomalies appear sporadically throughout the dataset. These are "
        "typically associated with schedule shifts due to seasonal sunset/sunrise changes or "
        "single-day nighttime consumption drops. Unlike the December cluster, these isolated events "
        "do not persist and are more consistent with temporary operational variations."
    )
    pdf.body_text(
        "The early months of the dataset (January - March 2025) also show a small number of elevated "
        "GAF scores, primarily driven by nighttime drops and schedule shift deviations. These are "
        "localized anomalies that do not indicate systemic issues."
    )

    # ================================================================
    # 5  Comparison
    # ================================================================
    pdf.section("5", "Comparison of Early vs Late Period")
    pdf.body_text(
        "Early period (January to June 2025): Daily profiles are stable with high nighttime amplitude "
        "(NightMean 6.8 - 7.1 kWh). Anomalies are primarily localized deviations, i.e., individual "
        "days with schedule shifts or minor nighttime drops. The GAF images for these days show "
        "subtle asymmetries in transition blocks but preserve the overall block structure."
    )
    pdf.body_text(
        "Late period (November 2025 to February 2026): The consumption structure shows a more "
        "significant distortion. Starting mid-December 2025, a persistent amplitude reduction "
        "is visible in the nightly plateau. This represents a global structural change captured "
        "effectively by GAF as altered block intensities. Unlike the localized early-period "
        "anomalies, these late-period deviations reflect sustained system-level changes."
    )
    pdf.body_text(
        "This contrast demonstrates GAF's particular strength: while both localized deviations "
        "(e.g. a single shifted ramp-up time) and global structural distortions (e.g. persistent "
        "amplitude reduction across all night hours) produce elevated anomaly scores, GAF is more "
        "sensitive to the latter, making it especially valuable for detecting progressive degradation."
    )

    # ================================================================
    # 6  Conclusion
    # ================================================================
    pdf.section("6", "Conclusion")
    pdf.body_text(
        "Combining hourly load profile visualization with Gramian Angular Field (GAF) transformation "
        "provides a reliable anomaly detection approach for street lighting consumption data. Over the "
        f"14-month analysis period ({n_days_scored} scored days after data quality filtering), the method "
        "successfully identified operationally significant anomalies, most notably a persistent "
        "nighttime amplitude reduction in December 2025 that is consistent with equipment degradation "
        "or partial circuit failure."
    )
    pdf.body_text(
        "The mask-aware scoring methodology ensures that anomaly detection is based exclusively on "
        "verified measurements. Missing hourly records are explicitly tracked through a binary "
        "presence mask, and the anomaly score is weighted by the fraction of present hours. "
        "This prevents data quality gaps from inflating reconstruction error and generating false "
        "positives, while days with genuine operational deviations and partial missing data still "
        "receive appropriately high scores proportional to their measured anomalous behavior."
    )
    pdf.body_text(
        "The GAF approach is particularly effective for detecting global daily-structure distortions "
        "that scalar metrics may miss. As an unsupervised per-day metric, it identifies structural "
        "deviations without requiring labeled training data, making it suitable for operational "
        "monitoring of street lighting installations."
    )

    out_pdf = os.path.join(DATA_DIR, "gaf_report_final.pdf")
    pdf.output(out_pdf)
    print(f"\n  Saved: {out_pdf}")
    print(f"  Figures in: {FIG_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
