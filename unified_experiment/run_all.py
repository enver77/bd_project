#!/usr/bin/env python3
"""
Unified Experiment Runner
==========================
Runs preprocessing ONCE, then all 4 team member approaches on the
same preprocessed data. Ensures fair comparison -- only the method differs.

Usage:
  python run_all.py                     # run everything
  python run_all.py --skip-preprocessing # skip preprocessing if already done
  python run_all.py --only feza         # run only Feza's pipeline
  python run_all.py --only sena         # run only Sena's approach
  python run_all.py --only enver        # run only Enver's approaches
  python run_all.py --only merged       # run only merged-data experiment

Requirements:
  - Copy raw data files into unified_experiment/data/:
      osf_*.xlsx                           (14 monthly consumption files)
      rpt-300_*.xlsx                       (meter outage report)
      rpt-301_*.xlsx                       (modem outage report)
      rpt-308_*.xlsx                       (all modem outages)
      akim-gerilim_raporu_Vkx3H.xlsx       (current/voltage data)
"""

import os
import sys
import time
import argparse

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PREPROCESSING_DIR = os.path.join(SCRIPT_DIR, "preprocessing")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
FEZA_DIR = os.path.join(MODELS_DIR, "feza_anomaly")

# Set environment variables so all modules find data/results
os.environ["BEDAS_DATA_DIR"] = DATA_DIR
os.environ["BEDAS_RESULTS_DIR"] = RESULTS_DIR

# Add to path
sys.path.insert(0, PREPROCESSING_DIR)
sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, FEZA_DIR)


def run_step(name, func):
    """Run a step with timing and error handling."""
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
    """Step 0: Unified preprocessing."""
    from unified_preprocessing import run
    run(DATA_DIR, RESULTS_DIR)


def run_feza():
    """Steps 1-6: Feza's full anomaly pipeline."""
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
    """Sena's Isolation Forest."""
    import sena_isolation_forest
    sena_isolation_forest.main()


def run_enver():
    """Enver's supervised approaches."""
    import enver_merged_gbm
    enver_merged_gbm.main()

    import enver_oversampled
    enver_oversampled.main()


def run_merged():
    """Merged data on Feza's pipeline experiment."""
    import merged_data_on_feza
    merged_data_on_feza.main()


def main():
    parser = argparse.ArgumentParser(description="Run unified experiment")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip preprocessing (use existing results)")
    parser.add_argument("--only", choices=["feza", "sena", "enver", "merged"],
                        help="Run only a specific approach")
    args = parser.parse_args()

    print("=" * 70)
    print(" UNIFIED EXPERIMENT RUNNER")
    print("=" * 70)
    print(f" Data dir   : {DATA_DIR}")
    print(f" Results dir: {RESULTS_DIR}")
    print("=" * 70)

    # Check data directory
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
        # Run specific approach only (assumes preprocessing is done)
        if not args.skip_preprocessing:
            results["Preprocessing"] = run_step("Step 0: Preprocessing", run_preprocessing)
        dispatch = {
            "feza": ("Feza Anomaly Pipeline", run_feza),
            "sena": ("Sena Isolation Forest", run_sena),
            "enver": ("Enver Supervised Models", run_enver),
            "merged": ("Merged Data Experiment", run_merged),
        }
        name, func = dispatch[args.only]
        results[name] = run_step(name, func)
    else:
        # Run everything
        if not args.skip_preprocessing:
            results["Preprocessing"] = run_step(
                "Step 0: Unified Preprocessing", run_preprocessing)

        results["Feza"] = run_step(
            "Step 1: Feza's Anomaly Pipeline (Modules 1-7)", run_feza)

        results["Sena"] = run_step(
            "Step 2: Sena's Isolation Forest", run_sena)

        results["Enver"] = run_step(
            "Step 3: Enver's Supervised Models", run_enver)

        results["Merged"] = run_step(
            "Step 4: Merged Data on Feza's Pipeline", run_merged)

    # ── Summary ──
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
