"""
Ali's Three-Tier Ensemble Pipeline -- Orchestrator
Reads from unified preprocessing output, runs Tier 0->1->2->3->Ensemble->Eval->Viz.

Usage (from ali_three_tier dir):
    python ali_run_pipeline.py
    python ali_run_pipeline.py --tiers 1,2,3
    python ali_run_pipeline.py --no-lstm
    python ali_run_pipeline.py --skip-viz
"""

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

RESULTS_DIR = os.environ.get("BEDAS_RESULTS_DIR",
    os.path.join(SCRIPT_DIR, "..", "..", "results"))


def run_step(name, func, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"  [OK] {name} completed in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="Ali's Three-Tier Ensemble Pipeline")
    parser.add_argument("--tiers", type=str, default="0,1,2,3",
                        help="Comma-separated tiers (0=preprocess, 1,2,3=detection)")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--no-lstm", action="store_true")
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",")]
    print(f"\nAli's Three-Tier Ensemble Pipeline")
    print(f"  Tiers: {tiers}")
    print(f"  Results dir: {RESULTS_DIR}")

    import pandas as pd

    DAILY_CSV = os.path.join(RESULTS_DIR, "ali_preprocessed_daily.csv")
    GT_CSV = os.path.join(RESULTS_DIR, "ali_ground_truth_labels.csv")

    # Tier 0 -- Preprocessing
    if "0" in tiers or not os.path.exists(DAILY_CSV):
        from ali_preprocess import main as preprocess_main
        daily = run_step("Tier 0 -- Preprocessing", preprocess_main)
    else:
        print("\n[Tier 0] Loading existing ali_preprocessed_daily.csv ...")
        daily = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)
        print(f"  Loaded {len(daily)} days")

    # Ground Truth
    if not os.path.exists(GT_CSV) or "0" in tiers:
        from ali_ground_truth import main as gt_main
        gt = run_step("Ground Truth Parsing", gt_main)
    else:
        gt = pd.read_csv(GT_CSV, index_col=0, parse_dates=True)
        print(f"\n[GT] Loaded {gt['outage'].sum()} ground truth outage days")

    tier1_df = tier2_df = tier3_df = cp_df = None

    if "1" in tiers:
        from tier1_statistical import main as t1_main
        tier1_df = run_step("Tier 1 -- STL + Z-Score + CUSUM", t1_main, daily.copy())

    if "2" in tiers:
        if args.no_lstm:
            import tier2_ml
            _orig = tier2_ml.run_lstm_autoencoder
            def _noop(X, index, **kwargs):
                import pandas as pd
                return pd.Series(0.0, index=index, name="t2b_lstm_severity"), \
                       pd.Series(False, index=index, name="t2b_flag")
            tier2_ml.run_lstm_autoencoder = _noop
            tier2_df = run_step("Tier 2 -- iForest (LSTM skipped)", tier2_ml.main, daily.copy())
            tier2_ml.run_lstm_autoencoder = _orig
        else:
            from tier2_ml import main as t2_main
            tier2_df = run_step("Tier 2 -- iForest + LSTM Autoencoder", t2_main, daily.copy())

    if "3" in tiers:
        from tier3_changepoint import main as t3_main
        t3_result = run_step("Tier 3 -- PELT Change-Point Detection", t3_main, daily.copy())
        if isinstance(t3_result, tuple):
            tier3_df, cp_df = t3_result
        else:
            tier3_df = t3_result

    if tier1_df is not None or tier2_df is not None or tier3_df is not None:
        import numpy as np

        if tier1_df is None:
            tier1_df = pd.DataFrame({
                "t1a_severity": 0.0, "t1b_severity": 0.0,
                "tier1_flag": False, "tier1_severity": 0.0,
            }, index=daily.index)
        if tier2_df is None:
            tier2_df = pd.DataFrame({
                "t2a_iforest_severity": 0.0, "t2b_lstm_severity": 0.0,
                "tier2_flag": False, "tier2_severity": 0.0,
                "pc1": 0.0, "pc2": 0.0,
            }, index=daily.index)
        if tier3_df is None:
            tier3_df = pd.DataFrame({
                "t3_severity": 0.0, "t3_flag": False,
            }, index=daily.index)

        from ensemble_scorer import main as ens_main
        report = run_step("Ensemble Scoring", ens_main,
                          tier1=tier1_df, tier2=tier2_df, tier3=tier3_df,
                          daily=daily, gt=gt)

        from evaluate import main as eval_main
        eval_results = run_step("Evaluation", eval_main, report)

        if not args.skip_viz:
            from visualize_anomalies import main as viz_main
            run_step("Visualization", viz_main,
                     tier1=tier1_df, tier2=tier2_df, tier3=tier3_df,
                     cp_df=cp_df, report=report, gt=gt)

        print(f"\n{'='*60}")
        print("ALI'S PIPELINE COMPLETE -- SUMMARY")
        print(f"{'='*60}")
        print(f"  Total days analyzed:      {len(daily)}")
        print(f"  Ground truth events:      {int(gt['outage'].sum())}")
        high_conf = (report["confidence"] == "high").sum()
        flagged = (report["ensemble_score"] >= 0.35).sum()
        print(f"  High-confidence anomalies:{high_conf}")
        print(f"  Total flagged (>=0.35):   {flagged}")

        best = eval_results.get("summary", {})
        print(f"  Best threshold: {best.get('best_threshold', '?')}")
        print(f"  Precision: {best.get('precision', 0):.3f}")
        print(f"  Recall:    {best.get('recall', 0):.3f}")
        print(f"  F1-Score:  {best.get('f1_score', 0):.3f}")

        OUT_CSV = os.path.join(RESULTS_DIR, "ali_anomaly_report.csv")
        print(f"\n  Main output: {OUT_CSV}")
    else:
        print("\nNo tiers ran -- only preprocessing done.")


if __name__ == "__main__":
    main()
