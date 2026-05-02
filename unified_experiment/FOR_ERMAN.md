# BEDAS Unified Experiment -- The Full Picture

**Updated:** May 2, 2026

---

## What This Project Is About

Imagine you're responsible for thousands of street lights in Istanbul. They turn on at sunset, off at sunrise, and you have no idea if any are broken until someone calls to complain. Now imagine you had a smart meter on the electrical panel feeding those lights, recording how much electricity flows through every hour. Could you detect failures *before* they become complaints?

That's this project. We have 14 months of hourly electricity consumption data from a single BEDAS (Istanbul's electricity distribution company) OSF point (a transformer feeding street lights), and we're building anomaly detection systems to catch problems early.

---

## The Team's Five Approaches

Our team attacked this problem from five angles. Think of it like five detectives investigating the same crime scene, each using a different method:

### 1. Feza's Unsupervised Pipeline (Modules 1-7) -- "The Full Medical Check-up"
The most comprehensive approach. Feza built a **7-module pipeline** that works like a doctor doing a full examination:
- **Module 0** (Data Pipeline): Parses raw Excel files cell-by-cell, tracks *why* data is missing, cross-references with BEDAS outage reports. This is the preprocessing that everyone now uses.
- **Module 1** (Baseline Features): Extracts daily statistics -- night mean, day mean, transition sharpness (how quickly consumption ramps up at sunset). Uses only present hours, never fills gaps silently.
- **Module 2** (Rule-Based Flags): Hard-coded domain knowledge -- "if all hours show zero but data exists, that's suspicious." Four rules: AllDayZero, DaytimeNonZero, NightDrop, ScheduleShift.
- **Module 3** (Change Points): CUSUM algorithm to detect when the consumption pattern fundamentally shifts (like a sudden drop in night consumption that persists for weeks).
- **Module 4** (Forecast Residuals): Builds an "expected" consumption profile for each hour/month/weekday combination, then flags days where the actual consumption deviates too much.
- **Module 5** (GAF Autoencoder): The most creative module. Converts 24h consumption curves into **Gramian Angular Fields** (mathematical images!), trains an autoencoder to reconstruct "normal" images, and uses reconstruction error as an anomaly score. High error = unusual day.
- **Module 6** (Combined Score): Weighted fusion of all four signal types into a single maintenance priority. Like a committee of doctors voting on a diagnosis.
- **Module 7** (Degradation): Tracks gradual decline over time -- is the system slowly getting worse? Uses rolling slopes and volatility metrics.

**Why it's smart:** No labels needed. It learns "normal" and flags deviations. With only 1 usable labeled fault day, this is the most robust approach.

### 2. Sena's Isolation Forest -- "The Elegant Detector"
The simplest and most elegant approach. Isolation Forest works on a beautiful insight: **anomalies are easier to isolate than normal points**. Instead of defining what's "normal" (hard), it asks "how many random splits does it take to separate this point from the rest?" (easy). Fewer splits = more anomalous.

Sena ran two versions:
- **Full-dataset**: One model on all 14 months for exploratory analysis
- **Train/Test**: Trained on Jan-Nov 2025, tested on Dec 2025-Feb 2026

Her visualizations (monthly heatmaps, day-of-month heatmaps, hourly distributions, seasonal pie charts) tell a compelling story about *when* and *where* anomalies cluster. Winter dominates anomalies -- which makes physical sense, since street lights run longest in winter nights.

### 3. Enver's Supervised GBM with Heuristic Optimization -- "The Heavy Artillery"
A GradientBoosting classifier that predicts "will there be a failure in the next 24 hours?", optimized with three bio-inspired algorithms:
- **GA (Genetic Algorithm)**: Evolves a population of feature subsets to find the best combination. Think of it as natural selection for features -- weak features die off, strong ones survive.
- **PSO (Particle Swarm Optimization)**: A swarm of particles flies through hyperparameter space, each attracted to its personal best and the global best. Like a flock of birds searching for food.
- **SA (Simulated Annealing)**: Fine-tunes the decision threshold. Instead of the default 0.5, finds the threshold that maximizes F1. Inspired by metallurgy -- start hot (explore wildly), cool down (exploit the best region).

Also tested: merging energy data with current/voltage measurements (akim-gerilim raporu) to see if electrical domain features (phase imbalance, power factor, etc.) improve prediction.

### 4. Enver's Oversampled Model -- "Fighting the Imbalance"
Same classification task, but focused on the class imbalance problem. Only ~5-6% of hours precede a failure -- the model sees 17 "no failure" examples for every 1 "failure." Without help, it just predicts "no failure" for everything and gets 94% accuracy while being completely useless.

Compared four strategies:
- **No oversampling** (baseline)
- **SMOTE**: Creates synthetic minority samples by interpolating between existing ones
- **SMOTE+Tomek Links**: SMOTE + removes borderline ambiguous samples
- **ADASYN**: Like SMOTE but creates more synthetic samples in harder-to-learn regions

### 5. Ali's Three-Tier Ensemble -- "The Statistical Council"
The most academically rigorous approach. Ali built a **three-tier voting system** where each tier uses a fundamentally different mathematical lens, and the final verdict is a weighted vote. This is the approach the LaTeX paper formally describes as the unsupervised model.

- **Tier 1 -- Statistical (35% weight):** Three independent statistical detectors:
  - **STL Decomposition** (Cleveland 1990): Splits the time series into Trend + Seasonal + Residual. Then computes a **Modified Z-Score** (Iglewicz & Hoaglin 1993, MAD-based, robust to outliers) on the residual. If the residual is more than 3.5 MAD below zero, that day is anomalous.
  - **CUSUM** (Page 1954): A "cumulative sum" control chart that triggers an alarm when the running deficit crosses 5 sigma. Catches sustained drops that a single z-score might miss.
  - **Rolling-Baseline Deviation**: Compares each day to its own 28-day rolling median, with a no-look-ahead `shift(1)` to prevent leakage. Important because with only 14 months of data (~1.16 cycles), STL's seasonal component absorbs almost all variation, leaving uninformative residuals -- this third signal compensates.

- **Tier 2 -- Machine Learning (35% weight):** Two models on the same 14 engineered features:
  - **Isolation Forest**: Same algorithm Sena uses, but Ali adds two robustness checks:
    - **Sensitivity analysis**: Retrains at contamination = {0.02, 0.05, 0.10} and reports how flagged-day count changes
    - **Bootstrap stability**: Five different random seeds, then computes pairwise Jaccard similarity of flagged date sets. If the average Jaccard > 0.7, the model is stable
  - **LSTM Autoencoder**: A sequence-to-sequence neural network that learns to reconstruct 7-day windows. Trained on the first 10 months ("normal"), tested on the rest. Days where reconstruction error exceeds the 95th percentile of training error are flagged. Skipped automatically if TensorFlow isn't installed.
  - **PCA projection**: 14D feature space -> 2D for visualization. Lets you see if anomalies cluster in feature space.

- **Tier 3 -- Change-Point Detection (30% weight):** **PELT algorithm** (Killick 2012, via the `ruptures` library) finds points where the time series structurally changes. Each change point is classified by relative magnitude (sudden_drop, partial_fault, grid_increase, minor) and days within +/-2 days of a "drop" change point get high severity scores.

- **Ensemble Fusion**: Final score = 0.35 * S_T1 + 0.35 * S_T2 + 0.30 * S_T3, where S_T = max(component severities within tier). Days are also classified by **vote count** (how many tiers flagged them) into confidence levels: high (score >= 0.6 OR votes >= 2), moderate, low.

- **Evaluation against ground truth**: Unlike the other approaches, Ali explicitly evaluates against **rpt-300** (operator-verified hardware faults). Reports Precision/Recall/F1 across thresholds {0.10, 0.15, 0.20, 0.25, 0.35, 0.50}, plus silhouette score, inter-method Jaccard agreement (Tier1 vs Tier3), and physical plausibility rate (% of flagged days where consumption was actually >20% below baseline).

**Why it's the most rigorous:** Five separate detectors voting, statistical sensitivity analyses, bootstrap stability checks, multiple thresholds reported. It's the only approach that explicitly acknowledges the ground-truth limitation in code (the `detectability_note` in `evaluation_results.json`).

---

## The Preprocessing Problem -- Why We Built `unified_experiment/`

Here's the dirty secret of team projects: **everyone preprocesses data differently**, and nobody notices until it's too late. This is like three cooks each using a different recipe for the base sauce, then wondering why the dishes taste different.

### What went wrong:

| Aspect | Feza | Sena | Enver |
|---|---|---|---|
| Excel parser | openpyxl cell-by-cell | pandas `read_excel` | Pre-built CSV |
| Missing data | Keeps mask (`is_missing` flag) | Interpolates everything | `ffill().fillna(0)` |
| Outage awareness | Cross-refs BEDAS reports | None | None |
| Cutoff date | Feb 13, 2026 (data ends) | No cutoff (includes zeros to Feb 28) | Inherited from CSV |
| Total rows | ~9,816 | ~10,176 | ~9,780 |

Sena's extra ~400 rows were **ghost data** -- the meter stopped reporting on Feb 13, but her code reindexed to Feb 28 and `interpolate(method="time")` happily filled in values that never existed. Her Isolation Forest was training on fabricated consumption data for 15 days.

When you compare "Sena found 306 anomalies" vs "Feza flagged 15 high-priority days", you're not comparing methods -- you're comparing **methods + preprocessing + data ranges**. It's like comparing two cars' lap times when one is driving on a different track.

### The fix: `unified_experiment/`

```
unified_experiment/
├── data/                           # Raw files go here (copy once)
├── preprocessing/
│   ├── data_pipeline.py            # Feza's parser (THE canonical one)
│   └── unified_preprocessing.py    # Wraps Feza's parser + controlled imputation
├── models/
│   ├── feza_anomaly/               # Modules 1-7 (reads from results/)
│   │   ├── module1_baseline.py
│   │   ├── module2_rules.py
│   │   ├── module3b_changepoint_filtered.py
│   │   ├── module4_forecast.py
│   │   ├── module5_gaf.py
│   │   ├── module6v2_combined.py
│   │   └── module7_degradation.py
│   ├── ali_three_tier/             # Ali's Three-Tier Ensemble
│   │   ├── ali_preprocess.py       # Tier 0: builds Ali's daily features
│   │   ├── ali_ground_truth.py     # rpt-300 + rpt-301 -> daily labels
│   │   ├── tier1_statistical.py    # STL + Modified Z-Score + CUSUM
│   │   ├── tier2_ml.py             # Isolation Forest + LSTM AE + PCA
│   │   ├── tier3_changepoint.py    # PELT (ruptures library)
│   │   ├── ensemble_scorer.py      # Weighted fusion: 0.35/0.35/0.30
│   │   ├── evaluate.py             # Precision/Recall/F1 vs ground truth
│   │   ├── visualize_anomalies.py  # 5 publication-quality figures
│   │   └── ali_run_pipeline.py     # Standalone Ali runner
│   ├── sena_isolation_forest.py    # Sena's IF (reads unified data)
│   ├── enver_merged_gbm.py         # GBM + GA/PSO/SA
│   ├── enver_oversampled.py        # SMOTE comparison
│   ├── merged_data_on_feza.py      # Experiment: merged data on Feza's pipeline
│   └── heuristic_opt.py            # GA, PSO, SA implementations
├── results/                        # ALL outputs land here
└── run_all.py                      # One command runs everything
```

The preprocessing pipeline produces four key files:
1. **`master_dataset.csv`** -- raw data with `is_missing` flags (Feza's mask-aware format)
2. **`master_imputed.csv`** -- same but with time-interpolated missing values
3. **`hourly_data.csv`** -- flat hourly format with time features (for Sena/Enver/Ali)
4. **`hourly_data_labeled.csv`** -- same + failure labels from outage reports

Ali's pipeline runs an additional `ali_preprocess.py` step on top of `hourly_data.csv` to produce `ali_preprocessed_daily.csv` -- a daily-aggregated feature table tailored to his specific algorithms (STL, Isolation Forest, PELT). This isn't a competing preprocessor; it's a downstream aggregation that all reads from the same canonical hourly data Feza's pipeline produced.

Now `run_all.py` runs preprocessing ONCE, producing standardized CSVs, and every model reads from the same source. The only variable is the model itself. Fair comparison achieved.

---

## The Ground Truth Problem (28 vs 7 Events)

This is a subtle but critical issue that the lecturer caught.

We have three BEDAS outage reports:

| Report | Contents | Events | Role |
|---|---|---|---|
| **rpt-300** | Meter hardware outages | 7 events, 6 days | Paper says: "ground truth" |
| **rpt-301** | Modem comm interruptions (filtered) | 28 events | Paper says: "secondary validation" |
| **rpt-308** | Modem comm interruptions (all) | ~28 events | What the code *actually* uses for labels |

**The mismatch:** Section 3.4 of the paper says rpt-300 is the only ground truth. But all three supervised model scripts (`failure_labeling.py`, `merged_model_comparison.py`, `oversampled_model.py`) use **rpt-308** as their `failure_next_24h` target.

**Why this happened:** rpt-300 has only 6 unique outage days, and **5 out of 6 occurred during daylight hours** when the street lights are off and drawing zero electricity. There's literally no consumption signature to learn from. Only the May 12, 2025 event (4:47-6:46 AM, 118 minutes) overlapped with active nighttime hours. You can't train a supervised model on 1 example.

So the code fell back to rpt-308 (modem communication interruptions, ~28 events) which gives enough positive samples to train on. These aren't hardware faults -- they're communication dropouts -- but they're the best proxy we have.

**The fix for the paper:** Explicitly state that supervised approaches use rpt-308 as proxy labels, while rpt-300 remains the hardware-verified ground truth for unsupervised evaluation. This is academically honest and common in predictive maintenance where verified fault data is scarce.

---

## Technologies & Why We Chose Them

### openpyxl vs pandas for Excel parsing
The OSOS (BEDAS's billing system) Excel files have a non-standard layout: consumption data lives in rows 23-70, columns 3-33, with "Cekis" (draw) and "Veris" (feed) rows interleaved. Pandas `read_excel` treats row 0 as headers and gets confused by the multi-row structure. openpyxl lets us address cells by exact coordinates -- row 23, column 3 = first hour of first day.

### numpy autoencoder vs PyTorch/TensorFlow
Feza's GAF autoencoder is 576->64->16->64->576 neurons. At this scale (400 training samples, 576 dimensions), numpy matrix multiplication runs in seconds. PyTorch would add dependency complexity with zero speed benefit. On a GPU machine with larger models, you'd swap to PyTorch.

### Gramian Angular Fields
GAFs encode temporal correlations as spatial patterns. Each cell (i,j) in the 24x24 image captures the angular relationship between hour i and hour j. An autoencoder trained on "normal" GAF images will produce high reconstruction error on days that look structurally different. It's like showing a face recognizer a picture of a dog -- the "this doesn't look right" signal is the anomaly score.

### GradientBoosting vs neural networks for supervised classification
With ~9,800 rows and 5-6% positive class, this is firmly "tabular data, small dataset" territory. GBMs (specifically `GradientBoostingClassifier`) dominate here and have for decades. Neural networks need more data, more tuning, and more compute for typically worse results on tabular data.

---

## Bugs We Hit and How We Fixed Them

### 1. The Ghost Data Bug (Sena)
Sena's notebook reindexed the timestamp to `2026-02-28` but real data stopped at `2026-02-13`. The `interpolate(method="time")` happily filled in 15 days of fabricated data. The fix: apply Feza's cutoff before any interpolation. In the unified pipeline, `CUTOFF = pd.Timestamp("2026-02-13")` is enforced before the data ever reaches any model.

### 2. The Hour Shift Bug (Sena)
In Sena's Excel parsing, the "Saat/Gun" (Hour/Day) column needed a `.shift(1)` because the hour label belongs to the *previous* row's Cekis data in the Excel layout. Feza's openpyxl approach avoids this entirely by reading specific row/column coordinates.

### 3. The Label Source Mismatch (Paper)
The paper said rpt-300 (7 events) was ground truth, but the code used rpt-308 (28 events). Not a code bug per se, but a documentation/conceptual bug that matters for reproducibility and intellectual honesty.

### 4. scikit-learn version conflict
`imbalanced-learn` needed `scikit-learn==1.5.2` exactly, but the environment had a newer version. Pinning `scikit-learn==1.5.2` in requirements fixed it. Lesson: always pin your ML library versions.

### 5. STL Seasonal Absorption (Ali's Tier 1)
With only ~14 months of data, the STL decomposition (period=365) sees only ~1.16 cycles. The seasonal component fits the annual pattern almost perfectly, leaving near-zero residuals everywhere except at series boundaries. Ali's Tier 1a (z-score on STL residual) was therefore essentially mute. Fix: he supplemented with **Tier 1c** -- a rolling-baseline deviation signal that operates on the raw normalized kWh, sidestepping STL's short-series limitation. This is documented as the `note_stl_limitation` field in `ali_tier1_results.json`. Lesson: when applying classical time-series decomposition to short series, always have a non-decomposition fallback signal.

---

## How to Run on the GPU Machine

```bash
# 1. Copy unified_experiment/ folder to the GPU machine
# 2. Copy raw data into unified_experiment/data/:
#      osf_2193681000_01_2025_*.xlsx  (all 14 monthly files)
#      rpt-300_*.xlsx, rpt-301_*.xlsx, rpt-308_*.xlsx
#      akim-gerilim_raporu_Vkx3H.xlsx

# 3. Install dependencies
pip install numpy pandas openpyxl scikit-learn==1.5.2 imbalanced-learn \
            matplotlib seaborn joblib statsmodels ruptures
# Optional: tensorflow (for Ali's LSTM autoencoder; auto-skipped if absent)
pip install tensorflow

# 4. Run everything (preprocessing + all 5 approaches)
cd unified_experiment
python run_all.py

# Or run specific parts:
python run_all.py --only feza          # Just Feza's 7-module pipeline
python run_all.py --only sena          # Just Sena's Isolation Forest
python run_all.py --only enver         # Just Enver's supervised models
python run_all.py --only ali           # Just Ali's Three-Tier Ensemble
python run_all.py --only merged        # Just the merged-data experiment
python run_all.py --skip-preprocessing # Rerun models without re-parsing Excel
```

Ali's pipeline can also be run standalone with finer control:

```bash
cd unified_experiment/models/ali_three_tier
python ali_run_pipeline.py --tiers 1,2,3       # Run all tiers
python ali_run_pipeline.py --no-lstm           # Skip LSTM (fast)
python ali_run_pipeline.py --skip-viz          # No visualizations
```

All outputs (CSVs, PNGs, pickled models) land in `unified_experiment/results/`.

---

## Key Lessons

1. **Agree on preprocessing first.** Before anyone writes a model, the team should produce ONE canonical dataset and freeze it. "My model works on my laptop" is meaningless if your data went through different transforms.

2. **Track your ground truth.** If you have 3 different failure reports, decide upfront which one is "truth" and document it in both the paper AND the code. Don't let them drift apart.

3. **Cutoffs matter.** Real-world data has edges -- months where the meter wasn't installed, days after service ended. Blindly interpolating across these boundaries creates plausible-looking but fabricated data.

4. **Class imbalance is the norm in predictive maintenance.** Failures are rare -- that's the whole point. Budget time for SMOTE/threshold optimization, not just model selection.

5. **Unsupervised beats supervised when labels are scarce.** With only 1 usable labeled fault day, supervised learning runs on fumes. Feza's unsupervised pipeline doesn't need labels at all -- it learns "normal" and flags deviations. More robust, more honest.

6. **The mask matters.** Treating missing data as zero is wrong (that's a real value). Interpolating it away is also wrong (you lose the information *that* it was missing). Feza's approach -- keep the mask, compute features only from present values -- is the most principled.
