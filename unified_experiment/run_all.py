#!/usr/bin/env python3
"""
Unified Experiment Runner -- All 5 Team Approaches
====================================================
Runs preprocessing ONCE, then all team approaches on the same preprocessed data.
Ensures fair comparison -- only the method differs.

Approaches:
  1. Feza   - Modular unsupervised pipeline (Modules 1-7: rules, GAF AE, etc.)
  2. Sena   - Isolation Forest with seasonal/temporal visualizations
  3. Enver  - Supervised GBM + GA/PSO/SA optimization (uses rpt-308 labels)
  4. Enver  - Supervised GBM + SMOTE/ADASYN oversampling
  5. Ali    - Three-Tier Ensemble (STL+CUSUM, iForest+LSTM, PELT) + ground truth eval
  6. Merged - Enver's akim-gerilim data fed into Feza's GAF pipeline (experiment)

Usage:
  python run_all.py                       # run everything
  python run_all.py --skip-preprocessing  # skip if already done
  python run_all.py --only feza|sena|enver|ali|merged

Requirements:
  Copy raw data files into unified_experiment/data/:
    osf_*.xlsx                           (14 monthly consumption files)
    rpt-300_*.xlsx                       (meter outage report -- Ali's ground truth)
    rpt-301_*.xlsx                       (modem outage filtered -- Ali's secondary)
    rpt-308_*.xlsx                       (all modem outages -- Enver's labels)
    akim-gerilim_raporu_Vkx3H.xlsx       (current/voltage data)
"""

import os
import sys
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PREPROCESSING_DIR = os.path.join(SCRIPT_DIR, "preprocessing")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
FEZA_DIR = os.path.join(MODELS_DIR, "feza_anomaly")
ALI_DIR = os.path.join(MODELS_DIR, "ali_three_tier")

# Set environment variables so all modules find data/results
os.environ["BEDAS_DATA_DIR"] = DATA_DIR
os.environ["BEDAS_RESULTS_DIR"] = RESULTS_DIR

# Add to path
sys.path.insert(0, PREPROCESSING_DIR)
sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, FEZA_DIR)
sys.path.insert(0, ALI_DIR)


def run_step(name, func):
    print(f"\n{'#' * 70}")
    print(f"# {name}")
    print(f"{'#' * 70}")
    t0 = time.time()
    try:
        func()
        elapsed = time.time() - t0
        print(f"\n[OK] {name} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[FAIL] {name} failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_preprocessing():
    from unified_preprocessing import run
    run(DATA_DIR, RESULTS_DIR)


def run_feza():
    from feza_anomaly import module1_baseline
    module1_baseline.main()
    from feza_anomaly import module2_rules
    module2_rules.main()
    from feza_anomaly import module3b_changepoint_filtered
    module3b_changepoint_filtered.main()
    from feza_anomaly import module4_forecast
    module4_forecast.main()
    from feza_anomaly import module5_gaf
    module5_gaf.main()
    from feza_anomaly import module6v2_combined
    module6v2_combined.main()
    from feza_anomaly import module7_degradation
    module7_degradation.main()


def run_sena():
    import sena_isolation_forest
    sena_isolation_forest.main()


def run_enver():
    import enver_merged_gbm
    enver_merged_gbm.main()
    import enver_oversampled
    enver_oversampled.main()


def run_ali():
    """Run Ali's full Three-Tier Ensemble pipeline."""
    import pandas as pd

    # Tier 0: Ali's daily feature table (from unified hourly_data.csv)
    from ali_preprocess import main as preprocess_main
    daily = preprocess_main()

    # Ground truth (rpt-300 + rpt-301)
    from ali_ground_truth import main as gt_main
    gt = gt_main()

    # Tier 1: STL + Modified Z-Score + CUSUM
    from tier1_statistical import main as t1_main
    tier1_df = t1_main(daily.copy())

    # Tier 2: Isolation Forest + LSTM Autoencoder
    from tier2_ml import main as t2_main
    tier2_df = t2_main(daily.copy())

    # Tier 3: PELT change-point detection
    from tier3_changepoint import main as t3_main
    t3_result = t3_main(daily.copy())
    if isinstance(t3_result, tuple):
        tier3_df, cp_df = t3_result
    else:
        tier3_df, cp_df = t3_result, None

    # Ensemble fusion
    from ensemble_scorer import main as ens_main
    report = ens_main(tier1=tier1_df, tier2=tier2_df, tier3=tier3_df,
                      daily=daily, gt=gt)

    # Evaluation against ground truth
    from evaluate import main as eval_main
    eval_main(report)

    # Publication-quality visualizations
    from visualize_anomalies import main as viz_main
    viz_main(tier1=tier1_df, tier2=tier2_df, tier3=tier3_df,
             cp_df=cp_df, report=report, gt=gt)


def run_merged():
    import merged_data_on_feza
    merged_data_on_feza.main()


def main():
    parser = argparse.ArgumentParser(description="Run unified experiment")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip preprocessing (use existing results)")
    parser.add_argument("--only", choices=["feza", "sena", "enver", "ali", "merged"],
                        help="Run only a specific approach")
    args = parser.parse_args()

    print("=" * 70)
    print(" UNIFIED EXPERIMENT RUNNER -- 5 TEAM APPROACHES")
    print("=" * 70)
    print(f" Data dir   : {DATA_DIR}")
    print(f" Results dir: {RESULTS_DIR}")
    print("=" * 70)

    if not os.path.isdir(DATA_DIR):
        print(f"\nERROR: Data directory not found: {DATA_DIR}")
        print("\nPlease create unified_experiment/data/ and copy these files:")
        print("  - osf_*.xlsx (14 monthly consumption files)")
        print("  - rpt-300_*.xlsx, rpt-301_*.xlsx, rpt-308_*.xlsx (outage reports)")
        print("  - akim-gerilim_raporu_Vkx3H.xlsx (current/voltage data)")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}
    t_start = time.time()

    if args.only:
        if not args.skip_preprocessing:
            results["Preprocessing"] = run_step("Step 0: Preprocessing", run_preprocessing)
        dispatch = {
            "feza":   ("Feza Anomaly Pipeline (Modules 1-7)", run_feza),
            "sena":   ("Sena Isolation Forest", run_sena),
            "enver":  ("Enver Supervised Models", run_enver),
            "ali":    ("Ali Three-Tier Ensemble", run_ali),
            "merged": ("Merged Data on Feza's GAF Pipeline", run_merged),
        }
        name, func = dispatch[args.only]
        results[name] = run_step(name, func)
    else:
        if not args.skip_preprocessing:
            results["Preprocessing"] = run_step(
                "Step 0: Unified Preprocessing", run_preprocessing)

        results["Feza"] = run_step(
            "Step 1: Feza's Anomaly Pipeline (Modules 1-7)", run_feza)

        results["Sena"] = run_step(
            "Step 2: Sena's Isolation Forest", run_sena)

        results["Enver"] = run_step(
            "Step 3: Enver's Supervised Models (GBM + GA/PSO/SA + SMOTE)", run_enver)

        results["Ali"] = run_step(
            "Step 4: Ali's Three-Tier Ensemble (STL+CUSUM, iForest+LSTM, PELT)", run_ali)

        results["Merged"] = run_step(
            "Step 5: Merged Data on Feza's GAF Pipeline (experiment)", run_merged)

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(" EXPERIMENT SUMMARY")
    print("=" * 70)
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"   [{status:6s}] {name}")
    print(f"\n   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Results in: {RESULTS_DIR}")
    print("=" * 70)

    if all(results.values()):
        print("\nAll steps completed successfully!")
    else:
        print("\nSome steps failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
