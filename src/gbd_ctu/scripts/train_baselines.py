"""GBD-CTU baseline training script.

This script trains Random Forest and XGBoost baselines on serialized CTU-13 IP
graphs. Inputs are a graph directory and baseline hyperparameters; outputs are
CSV and JSON reports.
"""

from __future__ import annotations

import argparse

from gbd_ctu.training.trainer_baseline import train_baselines


def main() -> int:
    """Parse CLI arguments and train the configured baseline models."""

    parser = argparse.ArgumentParser(description="Train classical baselines on CTU-13 graphs.")
    parser.add_argument("--graph-dir", default="artifacts/graphs")
    parser.add_argument("--output-dir", default="artifacts/reports")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rf-n-estimators", type=int, default=400)
    parser.add_argument("--xgb-n-estimators", type=int, default=250)
    args = parser.parse_args()
    train_baselines(
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        random_state=args.random_state,
        rf_n_estimators=args.rf_n_estimators,
        xgb_n_estimators=args.xgb_n_estimators,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
