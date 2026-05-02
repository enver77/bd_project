"""
BEDAS Predictive Maintenance - HTML Report Generator
Reads /tmp/report_data.json and writes outputs/anomaly_report.html
"""

import json
import os

# ── Load data ─────────────────────────────────────────────────────────────────
for path in ["C:/tmp/report_data.json", "/tmp/report_data.json"]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        print("Loaded from:", path)
        break
else:
    raise FileNotFoundError("report_data.json not found")

ts_data      = d["ts_data"]        # 424 items
gt_events    = d["gt_events"]      # 6 items
high_list    = d["high_list"]      # 23 items
monthly_list = d["monthly_list"]   # 14 items
cp_list      = d["cp_list"]        # 7 items
ev           = d["eval"]
imgs         = d.get("imgs", {})

# ── helpers ────────────────────────────────────────────────────────────────────
def js_list(lst):
    return json.dumps(lst)

def score_bar(score):
    """Return an inline score bar HTML snippet (0-1 value)."""
    pct = round(score * 100, 1)
    r = int(min(255, 50 + score * 205))
    g = int(max(50, 200 - score * 150))
    b = 80
    color = "#{:02x}{:02x}{:02x}".format(r, g, b)
    return (
        '<div class="score-bar-wrap">'
        '<div class="score-bar" style="width:{pct}%;background:{color}"></div>'
        '<span class="score-val">{raw:.3f}</span>'
        '</div>'
    ).format(pct=pct, color=color, raw=score)

def type_badge(t):
    cls_map = {
        "outage":        "badge-red",
        "partial_fault": "badge-orange",
        "spike":         "badge-yellow",
        "normal":        "badge-green",
    }
    cls = cls_map.get(t, "badge-default")
    label = t.replace("_", " ").title()
    return '<span class="badge {cls}">{label}</span>'.format(cls=cls, label=label)

def conf_badge(c):
    cls_map = {"high": "badge-red", "medium": "badge-orange", "low": "badge-green"}
    cls = cls_map.get(c, "badge-default")
    return '<span class="badge {cls}">{label}</span>'.format(cls=cls, label=c.title())

def detected_badge(detected):
    if detected:
        return '<span class="badge badge-green">Detected</span>'
    return '<span class="badge badge-red">Missed</span>'

# ── Build JS data arrays ───────────────────────────────────────────────────────
ts_dates    = [row["date"]  for row in ts_data]
ts_kwh      = [row["kwh"]   for row in ts_data]
ts_scores   = [row["score"] for row in ts_data]
ts_types    = [row["type"]  for row in ts_data]

# baseline = 7-day rolling mean (simple Python)
baseline = []
for i, row in enumerate(ts_data):
    w = ts_kwh[max(0, i-3):i+4]
    baseline.append(round(sum(w) / len(w), 3))

# anomaly markers
anomaly_pts = []
for row in ts_data:
    if row["conf"] in ("high", "medium"):
        anomaly_pts.append({"x": row["date"], "y": row["kwh"], "score": row["score"], "type": row["type"]})

# ensemble score bar chart data (23 high)
hl_dates  = [r["date"]  for r in high_list]
hl_scores = [r["score"] for r in high_list]

# monthly bar chart
mo_labels  = [r["month"]   for r in monthly_list]
mo_avgkwh  = [r["avg_kwh"] for r in monthly_list]
mo_flagged = [r["flagged"] for r in monthly_list]

# ── Bootstrap stability table ──────────────────────────────────────────────────
bstrap = ev.get("bootstrap_stability", {}).get("jaccard_pairs", {})
bstrap_rows = ""
for pair, val in bstrap.items():
    seeds = pair.replace("_vs_", " vs ")
    bstrap_rows += (
        "<tr><td>Seed {seeds}</td><td>{val:.4f}</td></tr>"
    ).format(seeds=seeds, val=val)

avg_j = ev.get("bootstrap_stability", {}).get("avg_jaccard", 0)

# ── Threshold table ────────────────────────────────────────────────────────────
thr_rows = ""
for thr, tv in sorted(ev["threshold_analysis"].items(), key=lambda x: float(x[0])):
    thr_rows += (
        "<tr>"
        "<td>{thr}</td>"
        "<td>{tp}</td><td>{fp}</td><td>{fn}</td><td>{tn}</td>"
        "<td>{prec:.4f}</td><td>{rec:.4f}</td><td>{f1:.4f}</td>"
        "<td>{nd}</td>"
        "</tr>"
    ).format(
        thr=thr, tp=tv["tp"], fp=tv["fp"], fn=tv["fn"], tn=tv["tn"],
        prec=tv["precision"], rec=tv["recall"], f1=tv["f1_score"],
        nd=tv["n_detected"]
    )

# ── High-confidence anomaly table rows ────────────────────────────────────────
high_rows = ""
for i, row in enumerate(high_list, 1):
    dev_color = "#d64040" if row["dev"] < 0 else "#2a7a2a"
    high_rows += (
        "<tr>"
        "<td>{i}</td>"
        "<td>{date}</td>"
        "<td>{sb}</td>"
        "<td>{tb}</td>"
        "<td style='color:{dc};font-weight:600'>{dev:+.1f}%</td>"
        "<td>{votes}</td>"
        "</tr>"
    ).format(
        i=i, date=row["date"],
        sb=score_bar(row["score"]),
        tb=type_badge(row["type"]),
        dc=dev_color, dev=row["dev"],
        votes=row["votes"]
    )

# ── PELT change-points table ───────────────────────────────────────────────────
cp_rows = ""
for row in cp_list:
    rel_color = "#d64040" if row["rel"] < 0 else "#2a7a2a"
    cp_rows += (
        "<tr>"
        "<td>{date}</td>"
        "<td>{tb}</td>"
        "<td style='color:{rc};font-weight:600'>{rel:+.1f}%</td>"
        "<td>{before:.1f}</td>"
        "<td>{after:.1f}</td>"
        "</tr>"
    ).format(
        date=row["date"],
        tb=type_badge(row["type"]),
        rc=rel_color, rel=row["rel"],
        before=row["before"], after=row["after"]
    )

# ── Ground-truth table ─────────────────────────────────────────────────────────
gt_rows = ""
for row in gt_events:
    gt_rows += (
        "<tr>"
        "<td>{date}</td>"
        "<td>{dur} min</td>"
        "<td>{window}</td>"
        "<td>{sb}</td>"
        "<td>{db}</td>"
        "</tr>"
    ).format(
        date=row["date"], dur=row["duration"],
        window=row["window"],
        sb=score_bar(row["score"]),
        db=detected_badge(row["detected"])
    )

# ── Embedded images ────────────────────────────────────────────────────────────
img_titles = {
    "01_stl_decomposition.png":  "STL Decomposition",
    "02_cusum_chart.png":        "CUSUM Chart",
    "03_iforest_pca_scatter.png":"Isolation Forest PCA",
    "04_changepoint_timeline.png":"Change-Point Timeline",
    "05_calendar_heatmap.png":   "Calendar Heatmap",
}
img_html = ""
for fname, b64 in imgs.items():
    title = img_titles.get(fname, fname)
    img_html += (
        '<div class="fig-card">'
        '<p class="fig-title">{title}</p>'
        '<img src="data:image/png;base64,{b64}" alt="{title}" loading="lazy">'
        '</div>'
    ).format(title=title, b64=b64)

# ── Summary eval values ────────────────────────────────────────────────────────
sil   = ev.get("silhouette", {}).get("silhouette_score", 0)
phys  = ev.get("physical_plausibility", {}).get("phys_support_rate", 0)
inter = ev.get("inter_method_agreement", {})
t1f   = inter.get("t1_flagged", 0)
t3f   = inter.get("t3_flagged", 0)
iunion = inter.get("union", 1)
ijac  = inter.get("agreement_rate_jaccard", 0)
det_note = ev.get("detectability_note", "")

# ── Build full HTML ─────────────────────────────────────────────────────────────
HTML_TOP = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>BEDAŞ · Predictive Maintenance Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#ffffff;
  --surface:#f9f9f9;
  --border:#e5e5e3;
  --accent:#d97757;
  --accent-dark:#b85c3a;
  --text:#1a1a19;
  --text2:#5e5e5b;
  --muted:#8e8e8b;
  --r:8px;
  --rl:12px;
  --shadow:0 1px 4px rgba(0,0,0,.06),0 4px 12px rgba(0,0,0,.04);
  --shadow-lg:0 2px 8px rgba(0,0,0,.08),0 8px 24px rgba(0,0,0,.06);
}
html{scroll-behavior:smooth;font-size:16px}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);line-height:1.6}

/* ── Header ─────────────────────────────────────────── */
header{
  position:sticky;top:0;z-index:100;
  background:rgba(255,255,255,.92);
  backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);
  padding:0 2rem;
  height:56px;
  display:flex;align-items:center;justify-content:space-between;
}
.logo{display:flex;align-items:center;gap:.6rem}
.logo-icon{
  width:32px;height:32px;border-radius:8px;
  background:linear-gradient(135deg,#d97757,#b85c3a);
  display:flex;align-items:center;justify-content:center;
  color:#fff;font-weight:800;font-size:.9rem;letter-spacing:-.5px;
}
.logo-text{font-weight:700;font-size:1rem;color:var(--text)}
.logo-sep{color:var(--muted);margin:0 .3rem}
.logo-sub{font-weight:400;font-size:.85rem;color:var(--text2)}
nav{display:flex;gap:1.4rem}
nav a{
  font-size:.82rem;font-weight:500;color:var(--text2);
  text-decoration:none;transition:color .15s;
}
nav a:hover{color:var(--accent)}

/* ── Layout ─────────────────────────────────────────── */
main{max-width:1200px;margin:0 auto;padding:2.5rem 2rem 6rem}
section{margin-bottom:4rem}

/* ── Section labels ──────────────────────────────────── */
.sec-label{
  font-size:.7rem;font-weight:700;letter-spacing:.12em;
  text-transform:uppercase;color:var(--accent);
  margin-bottom:.5rem;
}
.sec-title{
  font-size:1.75rem;font-weight:800;color:var(--text);
  line-height:1.2;margin-bottom:.5rem;
}
.sec-sub{font-size:.95rem;color:var(--text2);max-width:60ch;margin-bottom:2rem}

/* ── Hero ────────────────────────────────────────────── */
#hero{
  padding:3.5rem 0 1rem;
  border-bottom:1px solid var(--border);
  margin-bottom:3rem;
}
.hero-eyebrow{
  display:inline-block;
  background:linear-gradient(135deg,#d97757,#b85c3a);
  color:#fff;border-radius:6px;
  padding:.25rem .7rem;
  font-size:.75rem;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;margin-bottom:1.2rem;
}
.hero-title{font-size:2.5rem;font-weight:800;line-height:1.1;margin-bottom:.8rem}
.hero-title span{color:var(--accent)}
.hero-desc{font-size:1.05rem;color:var(--text2);max-width:65ch;margin-bottom:2.5rem}

/* ── KPI Cards ────────────────────────────────────────── */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin-bottom:2rem}
.kpi-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--rl);padding:1.25rem 1.5rem;
  box-shadow:var(--shadow);
}
.kpi-label{font-size:.75rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem}
.kpi-value{font-size:2.1rem;font-weight:800;color:var(--text);line-height:1}
.kpi-unit{font-size:.85rem;font-weight:500;color:var(--text2);margin-top:.2rem}
.kpi-card.accent{border-color:var(--accent);border-left:3px solid var(--accent)}

/* ── Cards & Grid ─────────────────────────────────────── */
.card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--rl);padding:1.75rem;
  box-shadow:var(--shadow);
}
.card-title{font-size:1rem;font-weight:700;color:var(--text);margin-bottom:1.25rem}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
.three-col{display:grid;grid-template-columns:repeat(3,1fr);gap:1.25rem}

/* ── Charts ──────────────────────────────────────────── */
.chart-wrap{position:relative;height:320px;width:100%}
.chart-wrap-sm{position:relative;height:240px;width:100%}

/* ── Tables ──────────────────────────────────────────── */
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.875rem}
thead th{
  background:var(--surface);color:var(--muted);
  font-size:.72rem;font-weight:700;letter-spacing:.07em;
  text-transform:uppercase;padding:.65rem 1rem;
  border-bottom:2px solid var(--border);text-align:left;
}
tbody tr{border-bottom:1px solid var(--border);transition:background .1s}
tbody tr:hover{background:#fafafa}
tbody td{padding:.6rem 1rem;color:var(--text);vertical-align:middle}
tbody tr:last-child{border-bottom:none}

/* ── Score bar ─────────────────────────────────────────── */
.score-bar-wrap{display:flex;align-items:center;gap:.5rem;min-width:120px}
.score-bar{height:6px;border-radius:3px;min-width:2px;transition:width .3s}
.score-val{font-size:.78rem;font-weight:600;color:var(--text2);white-space:nowrap}

/* ── Badges ───────────────────────────────────────────── */
.badge{
  display:inline-block;padding:.2rem .55rem;
  border-radius:4px;font-size:.72rem;font-weight:700;
  letter-spacing:.04em;text-transform:capitalize;white-space:nowrap;
}
.badge-red   {background:#fde8e8;color:#c0392b}
.badge-orange{background:#fef3e2;color:#c0661a}
.badge-yellow{background:#fefae2;color:#8a7a00}
.badge-green {background:#e8f5e9;color:#2a7a2a}
.badge-blue  {background:#e3f0fd;color:#1a5fa8}
.badge-default{background:var(--surface);color:var(--muted)}

/* ── Methodology cards ─────────────────────────────────── */
.method-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1.25rem}
.method-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--rl);padding:1.4rem;box-shadow:var(--shadow);
}
.method-icon{
  width:36px;height:36px;border-radius:8px;
  background:linear-gradient(135deg,#d97757,#b85c3a);
  display:flex;align-items:center;justify-content:center;
  color:#fff;font-size:1rem;margin-bottom:.9rem;
}
.method-name{font-size:.95rem;font-weight:700;color:var(--text);margin-bottom:.4rem}
.method-desc{font-size:.83rem;color:var(--text2);line-height:1.55}
.method-detail{font-size:.78rem;color:var(--muted);margin-top:.5rem;padding-top:.5rem;border-top:1px solid var(--border)}

/* ── Metric grid ──────────────────────────────────────── */
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}
.metric-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:1.1rem 1.25rem;
  box-shadow:var(--shadow);
}
.metric-name{font-size:.75rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}
.metric-val{font-size:1.5rem;font-weight:800;color:var(--text);margin:.25rem 0}
.metric-desc{font-size:.78rem;color:var(--text2)}

/* ── Figures ─────────────────────────────────────────── */
.fig-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:1.5rem}
.fig-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--rl);overflow:hidden;box-shadow:var(--shadow);
}
.fig-title{
  font-size:.8rem;font-weight:700;color:var(--text2);
  padding:.75rem 1rem;border-bottom:1px solid var(--border);
  text-transform:uppercase;letter-spacing:.06em;
}
.fig-card img{width:100%;display:block}

/* ── Info box ─────────────────────────────────────────── */
.info-box{
  background:#fff8f5;border:1px solid #f4d0be;
  border-left:3px solid var(--accent);border-radius:var(--r);
  padding:1rem 1.25rem;font-size:.85rem;color:#6b3a1f;line-height:1.6;
  margin-bottom:1.5rem;
}

/* ── References ──────────────────────────────────────── */
.ref-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}
.ref-item{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:1rem 1.25rem;
  font-size:.82rem;line-height:1.55;
}
.ref-num{font-weight:700;color:var(--accent);margin-right:.4rem}
.ref-title{font-weight:600;color:var(--text)}
.ref-detail{color:var(--text2);font-size:.78rem;margin-top:.2rem}

/* ── Footer ──────────────────────────────────────────── */
footer{
  border-top:1px solid var(--border);
  padding:1.5rem 2rem;text-align:center;
  font-size:.8rem;color:var(--muted);
}
footer strong{color:var(--text2)}

/* ── Responsive ──────────────────────────────────────── */
@media(max-width:768px){
  nav{display:none}
  .two-col,.three-col{grid-template-columns:1fr}
  .hero-title{font-size:1.9rem}
  main{padding:1.5rem 1rem 4rem}
}
</style>
</head>
<body>
"""

HTML_HEADER = """
<header>
  <div class="logo">
    <div class="logo-icon">B</div>
    <div>
      <span class="logo-text">BEDAŞ</span>
      <span class="logo-sep">·</span>
      <span class="logo-sub">Predictive Maintenance</span>
    </div>
  </div>
  <nav>
    <a href="#overview">Overview</a>
    <a href="#timeseries">Time Series</a>
    <a href="#anomalies">Anomalies</a>
    <a href="#changepoints">Change Points</a>
    <a href="#groundtruth">Ground Truth</a>
    <a href="#methodology">Methodology</a>
    <a href="#figures">Figures</a>
    <a href="#evaluation">Evaluation</a>
    <a href="#references">References</a>
  </nav>
</header>
<main>
"""

HTML_HERO = """
<section id="overview">
  <div id="hero">
    <div class="hero-eyebrow">Technical Report · 2025–2026</div>
    <h1 class="hero-title">BEDAŞ <span>Predictive Maintenance</span><br>Anomaly Detection System</h1>
    <p class="hero-desc">
      Ensemble-based unsupervised anomaly detection on 424 days of daily electricity
      consumption data (rpt-300 feeder). Combines STL decomposition, CUSUM control
      charts, Isolation Forest, and PELT change-point detection into a unified
      ensemble score.
    </p>
    <div class="kpi-grid">
      <div class="kpi-card accent">
        <div class="kpi-label">Analysis Window</div>
        <div class="kpi-value">424</div>
        <div class="kpi-unit">days (Jan 2025 – Feb 2026)</div>
      </div>
      <div class="kpi-card accent">
        <div class="kpi-label">High-Confidence Anomalies</div>
        <div class="kpi-value">23</div>
        <div class="kpi-unit">ensemble score ≥ 0.35</div>
      </div>
      <div class="kpi-card accent">
        <div class="kpi-label">Total Flagged Days</div>
        <div class="kpi-value">67</div>
        <div class="kpi-unit">score &gt; 0.10 threshold</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Ground Truth Events</div>
        <div class="kpi-value">6</div>
        <div class="kpi-unit">confirmed outage days</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Bootstrap Jaccard</div>
        <div class="kpi-value">0.922</div>
        <div class="kpi-unit">avg across 10 seed pairs</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Silhouette Score</div>
        <div class="kpi-value">0.509</div>
        <div class="kpi-unit">cluster separation quality</div>
      </div>
    </div>
  </div>
</section>
"""

HTML_TIMESERIES_START = """
<section id="timeseries">
  <div class="sec-label">Time Series Analysis</div>
  <h2 class="sec-title">Daily Consumption &amp; Anomaly Detection</h2>
  <p class="sec-sub">424-day daily kWh aggregates with 7-day rolling baseline and anomaly markers coloured by ensemble score.</p>
  <div class="card" style="margin-bottom:1.5rem">
    <div class="card-title">Daily kWh · Baseline · Anomaly Markers</div>
    <div class="chart-wrap">
      <canvas id="tsChart"></canvas>
    </div>
  </div>
  <div class="two-col">
    <div class="card">
      <div class="card-title">Ensemble Score · Top 23 Anomalies</div>
      <div class="chart-wrap-sm">
        <canvas id="scoreChart"></canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Monthly Average kWh &amp; Flagged Days</div>
      <div class="chart-wrap-sm">
        <canvas id="monthChart"></canvas>
      </div>
    </div>
  </div>
</section>
"""

HTML_ANOMALY_TABLE = """
<section id="anomalies">
  <div class="sec-label">Anomaly Catalogue</div>
  <h2 class="sec-title">23 High-Confidence Anomalies</h2>
  <p class="sec-sub">Days with ensemble score ≥ 0.35 and ≥ 2 method votes. Score bars indicate magnitude; deviation is relative to rolling baseline.</p>
  <div class="card">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Date</th><th>Ensemble Score</th><th>Type</th><th>Deviation</th><th>Votes</th>
          </tr>
        </thead>
        <tbody>
""" + high_rows + """
        </tbody>
      </table>
    </div>
  </div>
</section>
"""

HTML_CP_TABLE = """
<section id="changepoints">
  <div class="sec-label">Structural Breaks</div>
  <h2 class="sec-title">PELT Change Points</h2>
  <p class="sec-sub">7 structural breaks detected by Pruned Exact Linear Time (PELT) algorithm with RBF cost function (penalty=15).</p>
  <div class="card">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th><th>Type</th><th>Relative Shift</th><th>Before (kWh)</th><th>After (kWh)</th>
          </tr>
        </thead>
        <tbody>
""" + cp_rows + """
        </tbody>
      </table>
    </div>
  </div>
</section>
"""

HTML_GT_TABLE = (
"""
<section id="groundtruth">
  <div class="sec-label">Validation</div>
  <h2 class="sec-title">Ground Truth Comparison</h2>
  <p class="sec-sub">6 confirmed electricity outage events from BEDAŞ rpt-300 records, compared against ensemble detection results.</p>
  <div class="info-box">
    <strong>Detectability Note:</strong> """
+ det_note +
"""
  </div>
  <div class="card">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th><th>Duration (min)</th><th>Outage Window</th><th>Ensemble Score</th><th>Status</th>
          </tr>
        </thead>
        <tbody>
"""
+ gt_rows +
"""
        </tbody>
      </table>
    </div>
  </div>
</section>
"""
)

HTML_METHODOLOGY = """
<section id="methodology">
  <div class="sec-label">Methods</div>
  <h2 class="sec-title">Detection Methodology</h2>
  <p class="sec-sub">Six-stage pipeline combining classical statistical methods with modern machine learning for robust anomaly detection.</p>
  <div class="method-grid">
    <div class="method-card">
      <div class="method-icon">1</div>
      <div class="method-name">Preprocessing &amp; Aggregation</div>
      <div class="method-desc">15-minute interval meter data aggregated to daily kWh totals. Missing values filled via temporal interpolation. Outlier-resistant normalisation applied.</div>
      <div class="method-detail">Input: raw meter reads · Output: 424-day daily series</div>
    </div>
    <div class="method-card">
      <div class="method-icon">2</div>
      <div class="method-name">STL + Modified Z-Score</div>
      <div class="method-desc">Seasonal-Trend Decomposition via LOESS separates weekly seasonality and long-term trend. Modified Z-score applied to residuals detects statistical outliers.</div>
      <div class="method-detail">Period=7 (weekly), robust=True, MZS threshold=3.5</div>
    </div>
    <div class="method-card">
      <div class="method-icon">3</div>
      <div class="method-name">CUSUM Control Chart</div>
      <div class="method-desc">Cumulative Sum chart monitors shifts in the mean consumption level. Signals when cumulative deviations exceed a control limit, catching sustained drift.</div>
      <div class="method-detail">k=0.5·σ, h=5·σ (two-sided), reset on signal</div>
    </div>
    <div class="method-card">
      <div class="method-icon">4</div>
      <div class="method-name">Isolation Forest</div>
      <div class="method-desc">Unsupervised tree-based anomaly detection using feature space consisting of consumption, rolling statistics, and day-of-week indicators.</div>
      <div class="method-detail">n_estimators=200, contamination=auto, max_features=1.0</div>
    </div>
    <div class="method-card">
      <div class="method-icon">5</div>
      <div class="method-name">PELT Change-Point Detection</div>
      <div class="method-desc">Pruned Exact Linear Time algorithm detects abrupt structural breaks in the mean and variance of the time series, identifying equipment degradation events.</div>
      <div class="method-detail">Model=rbf, penalty=15 (BIC-calibrated), 7 breaks found</div>
    </div>
    <div class="method-card">
      <div class="method-icon">6</div>
      <div class="method-name">Ensemble Scoring</div>
      <div class="method-desc">Weighted combination of normalised anomaly signals from all methods. Final ensemble score ∈ [0,1]; days scoring ≥ 0.35 with ≥ 2 votes classified as high-confidence.</div>
      <div class="method-detail">Weights: STL×0.3, CUSUM×0.25, iForest×0.3, PELT×0.15</div>
    </div>
  </div>
</section>
"""

HTML_FIGURES = """
<section id="figures">
  <div class="sec-label">Visual Diagnostics</div>
  <h2 class="sec-title">Static Analysis Figures</h2>
  <p class="sec-sub">Embedded diagnostic visualisations generated during the analysis pipeline.</p>
  <div class="fig-grid">
""" + img_html + """
  </div>
</section>
"""

HTML_EVAL = (
"""
<section id="evaluation">
  <div class="sec-label">Performance Metrics</div>
  <h2 class="sec-title">Evaluation Results</h2>
  <p class="sec-sub">Comprehensive evaluation including threshold analysis, cluster quality, bootstrap stability, and physical plausibility.</p>

  <div class="metric-grid" style="margin-bottom:2rem">
    <div class="metric-card">
      <div class="metric-name">Bootstrap Jaccard (avg)</div>
      <div class="metric-val">{avg_j:.4f}</div>
      <div class="metric-desc">Stability across 5 random seeds, 10 pairs</div>
    </div>
    <div class="metric-card">
      <div class="metric-name">Silhouette Score</div>
      <div class="metric-val">{sil:.4f}</div>
      <div class="metric-desc">Cluster separation: normal vs anomaly</div>
    </div>
    <div class="metric-card">
      <div class="metric-name">Physical Support Rate</div>
      <div class="metric-val">{phys:.1%}</div>
      <div class="metric-desc">{n_phys} of {n_flag} flagged days physically plausible</div>
    </div>
    <div class="metric-card">
      <div class="metric-name">Inter-Method Jaccard</div>
      <div class="metric-val">{ijac:.4f}</div>
      <div class="metric-desc">T1({t1f}) ∩ T3({t3f}) / T1 ∪ T3 = {iunion}</div>
    </div>
    <div class="metric-card">
      <div class="metric-name">Best Recall (τ=0.1)</div>
      <div class="metric-val">0.3333</div>
      <div class="metric-desc">2 of 6 GT days detected at low threshold</div>
    </div>
    <div class="metric-card">
      <div class="metric-name">Best Precision (τ=0.5)</div>
      <div class="metric-val">N/A</div>
      <div class="metric-desc">No GT days captured at high threshold</div>
    </div>
  </div>

  <div class="card" style="margin-bottom:1.5rem">
    <div class="card-title">Threshold Analysis (Precision / Recall / F1)</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Threshold</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th>
            <th>Precision</th><th>Recall</th><th>F1</th><th>Detected</th>
          </tr>
        </thead>
        <tbody>
""" + thr_rows + """
        </tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Bootstrap Stability · Pairwise Jaccard Similarity</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Seed Pair</th><th>Jaccard Similarity</th></tr></thead>
        <tbody>
""" + bstrap_rows + """
        </tbody>
      </table>
    </div>
  </div>
</section>
"""
).format(
    avg_j=avg_j, sil=sil, phys=phys,
    n_phys=ev.get("physical_plausibility",{}).get("n_physically_supported",0),
    n_flag=ev.get("physical_plausibility",{}).get("n_flagged",0),
    ijac=ijac, t1f=t1f, t3f=t3f, iunion=iunion
)

HTML_REFERENCES = """
<section id="references">
  <div class="sec-label">Literature</div>
  <h2 class="sec-title">References</h2>
  <div class="ref-grid">
    <div class="ref-item">
      <span class="ref-num">[1]</span>
      <span class="ref-title">Cleveland et al. (1990) – STL: A Seasonal-Trend Decomposition Procedure Based on Loess</span>
      <div class="ref-detail">Journal of Official Statistics, 6(1), 3–73</div>
    </div>
    <div class="ref-item">
      <span class="ref-num">[2]</span>
      <span class="ref-title">Page (1954) – Continuous Inspection Schemes</span>
      <div class="ref-detail">Biometrika 41(1-2), 100–115 · CUSUM basis</div>
    </div>
    <div class="ref-item">
      <span class="ref-num">[3]</span>
      <span class="ref-title">Liu et al. (2008) – Isolation Forest</span>
      <div class="ref-detail">IEEE ICDM 2008 · doi:10.1109/ICDM.2008.17</div>
    </div>
    <div class="ref-item">
      <span class="ref-num">[4]</span>
      <span class="ref-title">Killick et al. (2012) – Optimal Detection of Changepoints with a Linear Computational Cost</span>
      <div class="ref-detail">JASA 107(500), 1590–1598 · PELT algorithm</div>
    </div>
    <div class="ref-item">
      <span class="ref-num">[5]</span>
      <span class="ref-title">Rousseeuw &amp; Croux (1993) – Alternatives to the Median Absolute Deviation</span>
      <div class="ref-detail">JASA 88(424), 1273–1283 · Modified Z-score</div>
    </div>
    <div class="ref-item">
      <span class="ref-num">[6]</span>
      <span class="ref-title">Truong et al. (2020) – Selective review of offline change point detection methods</span>
      <div class="ref-detail">Signal Processing 167 · ruptures library</div>
    </div>
  </div>
</section>
"""

HTML_FOOTER = """
</main>
<footer>
  <strong>BEDAŞ · Predictive Maintenance Report</strong> &nbsp;·&nbsp;
  Generated 2026-03-16 &nbsp;·&nbsp;
  Kadir Has University – OSOS Data Project &nbsp;·&nbsp;
  Analysis period: 2025-01-01 → 2026-02-28
</footer>
"""

# ── JavaScript (Chart.js) ──────────────────────────────────────────────────────
js_ts_dates    = json.dumps(ts_dates)
js_ts_kwh      = json.dumps(ts_kwh)
js_baseline    = json.dumps(baseline)
js_anom_pts    = json.dumps(anomaly_pts)
js_hl_dates    = json.dumps(hl_dates)
js_hl_scores   = json.dumps(hl_scores)
js_mo_labels   = json.dumps(mo_labels)
js_mo_avgkwh   = json.dumps(mo_avgkwh)
js_mo_flagged  = json.dumps(mo_flagged)

# score-based colours for bar chart
bar_colors = []
for s in hl_scores:
    r = min(255, int(50  + s * 205))
    g = max(50,  int(200 - s * 150))
    bar_colors.append("rgba({r},{g},80,0.85)".format(r=r, g=g))
js_bar_colors = json.dumps(bar_colors)

HTML_SCRIPTS = """
<script>
(function(){
// ── 1. Time Series Chart ──────────────────────────────────────────────────────
var tsDates   = """ + js_ts_dates + """;
var tsKwh     = """ + js_ts_kwh + """;
var tsBase    = """ + js_baseline + """;
var tsAnom    = """ + js_anom_pts + """;

var ctx1 = document.getElementById('tsChart').getContext('2d');
new Chart(ctx1, {
  type: 'line',
  data: {
    labels: tsDates,
    datasets: [
      {
        label: 'Daily kWh',
        data: tsKwh,
        borderColor: 'rgba(100,149,237,0.85)',
        backgroundColor: 'rgba(100,149,237,0.07)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
        order: 3
      },
      {
        label: '7-day Baseline',
        data: tsBase,
        borderColor: 'rgba(90,180,100,0.9)',
        borderWidth: 2,
        borderDash: [5,3],
        pointRadius: 0,
        fill: false,
        tension: 0.4,
        order: 2
      },
      {
        label: 'Anomaly Marker',
        data: tsAnom.map(function(p){ return {x: p.x, y: p.y}; }),
        borderColor: 'rgba(217,119,87,1)',
        backgroundColor: 'rgba(217,119,87,0.9)',
        pointRadius: tsAnom.map(function(p){ return 4 + p.score * 6; }),
        pointHoverRadius: 8,
        showLine: false,
        type: 'scatter',
        order: 1
      }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    interaction: {mode:'index', intersect:false},
    plugins:{
      legend:{
        labels:{font:{family:'Inter',size:11}, color:'#5e5e5b', boxWidth:12}
      },
      tooltip:{
        backgroundColor:'rgba(26,26,25,0.92)',
        titleFont:{family:'Inter',size:11},
        bodyFont:{family:'Inter',size:11},
        padding:10
      }
    },
    scales:{
      x:{
        ticks:{
          maxTicksLimit:12, maxRotation:0,
          font:{family:'Inter',size:10}, color:'#8e8e8b'
        },
        grid:{color:'rgba(0,0,0,0.04)'}
      },
      y:{
        title:{display:true, text:'kWh', font:{family:'Inter',size:11}, color:'#8e8e8b'},
        ticks:{font:{family:'Inter',size:10}, color:'#8e8e8b'},
        grid:{color:'rgba(0,0,0,0.05)'}
      }
    }
  }
});

// ── 2. Ensemble Score Bar Chart ───────────────────────────────────────────────
var hlDates   = """ + js_hl_dates + """;
var hlScores  = """ + js_hl_scores + """;
var hlColors  = """ + js_bar_colors + """;

var ctx2 = document.getElementById('scoreChart').getContext('2d');
new Chart(ctx2, {
  type: 'bar',
  data: {
    labels: hlDates,
    datasets: [{
      label: 'Ensemble Score',
      data: hlScores,
      backgroundColor: hlColors,
      borderRadius: 4,
      borderSkipped: false
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins:{
      legend:{display:false},
      tooltip:{
        backgroundColor:'rgba(26,26,25,0.92)',
        callbacks:{
          label: function(c){ return 'Score: ' + c.raw.toFixed(3); }
        }
      }
    },
    scales:{
      x:{ticks:{maxRotation:55, font:{size:9}, color:'#8e8e8b'}, grid:{display:false}},
      y:{min:0, max:1, ticks:{font:{size:10}, color:'#8e8e8b'}, grid:{color:'rgba(0,0,0,0.05)'}}
    }
  }
});

// ── 3. Monthly Bar Chart ──────────────────────────────────────────────────────
var moLabels  = """ + js_mo_labels + """;
var moAvgKwh  = """ + js_mo_avgkwh + """;
var moFlagged = """ + js_mo_flagged + """;

var ctx3 = document.getElementById('monthChart').getContext('2d');
new Chart(ctx3, {
  type: 'bar',
  data: {
    labels: moLabels,
    datasets: [
      {
        label: 'Avg kWh',
        data: moAvgKwh,
        backgroundColor: 'rgba(100,149,237,0.75)',
        borderRadius: 4,
        yAxisID: 'y'
      },
      {
        label: 'Flagged Days',
        data: moFlagged,
        backgroundColor: 'rgba(217,119,87,0.85)',
        borderRadius: 4,
        type: 'line',
        fill: false,
        tension: 0.35,
        borderColor: 'rgba(217,119,87,1)',
        pointRadius: 4,
        yAxisID: 'y2'
      }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins:{
      legend:{labels:{font:{family:'Inter',size:10}, color:'#5e5e5b', boxWidth:10}}
    },
    scales:{
      x:{ticks:{maxRotation:55, font:{size:9}, color:'#8e8e8b'}, grid:{display:false}},
      y:{
        position:'left', title:{display:true,text:'Avg kWh',font:{size:10}},
        ticks:{font:{size:9}, color:'#8e8e8b'}, grid:{color:'rgba(0,0,0,0.05)'}
      },
      y2:{
        position:'right', title:{display:true,text:'Flagged Days',font:{size:10}},
        ticks:{font:{size:9}, color:'#8e8e8b'}, grid:{display:false}
      }
    }
  }
});
})();
</script>
"""

HTML_BOTTOM = """
</body>
</html>
"""

# ── Assemble ───────────────────────────────────────────────────────────────────
html = (
    HTML_TOP
    + HTML_HEADER
    + HTML_HERO
    + HTML_TIMESERIES_START
    + HTML_ANOMALY_TABLE
    + HTML_CP_TABLE
    + HTML_GT_TABLE
    + HTML_METHODOLOGY
    + HTML_FIGURES
    + HTML_EVAL
    + HTML_REFERENCES
    + HTML_FOOTER
    + HTML_SCRIPTS
    + HTML_BOTTOM
)

# ── Write output ───────────────────────────────────────────────────────────────
out_path = (
    r"C:\Users\Ali Gökay Bozok"
    r"\OneDrive - Kadir Has University"
    r"\Masaüstü\OSOS_Data_Project"
    r"\outputs\anomaly_report.html"
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(out_path) / 1024
print("Report written to:", out_path)
print("File size: {:.1f} KB".format(size_kb))
print("HTML length: {:,} characters".format(len(html)))
