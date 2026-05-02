"""
Pipeline Orchestrator
Usage:
    python run_pipeline.py               # run all tiers
    python run_pipeline.py --tiers 1,2,3
    python run_pipeline.py --tiers 1     # only Tier1
    python run_pipeline.py --skip-viz    # skip visualization
    python run_pipeline.py --no-lstm     # skip LSTM (faster)
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

DAILY_CSV = PROJECT_ROOT / "data" / "processed" / "preprocessed_daily.csv"
GT_CSV    = PROJECT_ROOT / "data" / "processed" / "ground_truth_labels.csv"


def run_step(name: str, func, *args, **kwargs):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"  ✓ {name} completed in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="BEDAŞ Predictive Maintenance Pipeline")
    parser.add_argument(
        "--tiers", type=str, default="0,1,2,3",
        help="Comma-separated list of tiers to run (0=preprocess, 1,2,3=detection)"
    )
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization step")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM autoencoder")
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",")]
    print(f"\nBEDAŞ Predictive Maintenance Pipeline")
    print(f"  Tiers: {tiers}")
    print(f"  Skip viz: {args.skip_viz}")
    print(f"  No LSTM: {args.no_lstm}")

    import pandas as pd

    # -----------------------------------------------------------------------
    # Tier 0 — Preprocessing
    # -----------------------------------------------------------------------
    if "0" in tiers or not DAILY_CSV.exists():
        from preprocess import main as preprocess_main
        daily = run_step("Tier 0 — Preprocessing", preprocess_main)
    else:
        print("\n[Tier 0] Loading existing preprocessed_daily.csv …")
        daily = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)
        print(f"  Loaded {len(daily)} days")

    # Ground Truth
    if not GT_CSV.exists() or "0" in tiers:
        from ground_truth import main as gt_main
        gt = run_step("Ground Truth Parsing", gt_main)
    else:
        gt = pd.read_csv(GT_CSV, index_col=0, parse_dates=True)
        print(f"\n[GT] Loaded {gt['outage'].sum()} ground truth outage days")

    tier1_df = tier2_df = tier3_df = cp_df = None

    # -----------------------------------------------------------------------
    # Tier 1 — Statistical
    # -----------------------------------------------------------------------
    if "1" in tiers:
        from tier1_statistical import main as t1_main
        tier1_df = run_step("Tier 1 — STL + Z-Score + CUSUM", t1_main, daily.copy())

    # -----------------------------------------------------------------------
    # Tier 2 — Machine Learning
    # -----------------------------------------------------------------------
    if "2" in tiers:
        if args.no_lstm:
            # Monkey-patch to skip LSTM
            import tier2_ml
            _orig_lstm = tier2_ml.run_lstm_autoencoder
            def _noop_lstm(X, index, **kwargs):
                import pandas as pd
                dummy = pd.Series(0.0, index=index, name="t2b_lstm_severity")
                return dummy, pd.Series(False, index=index, name="t2b_flag")
            tier2_ml.run_lstm_autoencoder = _noop_lstm
            tier2_df = run_step("Tier 2 — iForest (LSTM skipped)", tier2_ml.main, daily.copy())
            tier2_ml.run_lstm_autoencoder = _orig_lstm
        else:
            from tier2_ml import main as t2_main
            tier2_df = run_step("Tier 2 — iForest + LSTM Autoencoder", t2_main, daily.copy())

    # -----------------------------------------------------------------------
    # Tier 3 — Change-Point Detection
    # -----------------------------------------------------------------------
    if "3" in tiers:
        from tier3_changepoint import main as t3_main
        t3_result = run_step("Tier 3 — PELT Change-Point Detection", t3_main, daily.copy())
        if isinstance(t3_result, tuple):
            tier3_df, cp_df = t3_result
        else:
            tier3_df = t3_result

    # -----------------------------------------------------------------------
    # Ensemble Scoring
    # -----------------------------------------------------------------------
    if tier1_df is not None or tier2_df is not None or tier3_df is not None:
        # Fill missing tiers with zero-score DataFrames
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
        report = run_step(
            "Ensemble Scoring",
            ens_main,
            tier1=tier1_df, tier2=tier2_df, tier3=tier3_df,
            daily=daily, gt=gt
        )

        # -----------------------------------------------------------------------
        # Evaluation
        # -----------------------------------------------------------------------
        from evaluate import main as eval_main
        eval_results = run_step("Evaluation", eval_main, report)

        # -----------------------------------------------------------------------
        # Visualization
        # -----------------------------------------------------------------------
        if not args.skip_viz:
            from visualize_anomalies import main as viz_main
            run_step(
                "Visualization",
                viz_main,
                tier1=tier1_df, tier2=tier2_df, tier3=tier3_df,
                cp_df=cp_df, report=report, gt=gt
            )

        # -----------------------------------------------------------------------
        # Final summary
        # -----------------------------------------------------------------------
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE — SUMMARY")
        print(f"{'='*60}")
        print(f"  Total days analyzed:      {len(daily)}")
        print(f"  Ground truth events:      {int(gt['outage'].sum())}")
        high_conf = (report["confidence"] == "high").sum()
        flagged   = (report["ensemble_score"] >= 0.35).sum()
        print(f"  High-confidence anomalies:{high_conf}")
        print(f"  Total flagged (≥0.35):    {flagged}")

        best = eval_results.get("summary", {})
        print(f"  Best threshold: {best.get('best_threshold', '?')}")
        print(f"  Precision: {best.get('precision', 0):.3f}")
        print(f"  Recall:    {best.get('recall',    0):.3f}")
        print(f"  F1-Score:  {best.get('f1_score',  0):.3f}")
        # Also show threshold=0.10 results
        thr010 = eval_results.get("threshold_analysis", {}).get("0.1", {})
        if thr010:
            print(f"  At thr=0.10: Detected={thr010.get('tp',0)}/{thr010.get('n_ground_truth_days',0)} GT days")

        OUT_CSV = PROJECT_ROOT / "outputs" / "anomaly_report.csv"
        print(f"\n  Main output: {OUT_CSV}")

    else:
        print("\nNo tiers ran — only preprocessing done.")
        print(f"  preprocessed_daily.csv: {DAILY_CSV}")


if __name__ == "__main__":
    main()
