"""GBD-CTU evaluation script.

This script evaluates a trained GNN checkpoint across CTU-13 scenarios. Inputs
are a graph directory, checkpoint path, and output path; outputs are CSV and JSON
scenario reports.
"""

from __future__ import annotations

import argparse

from gbd_ctu.evaluation.scenario_eval import evaluate_gnn_checkpoint


def main() -> int:
    """Parse CLI arguments and run scenario-wise GNN evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a GNN checkpoint on CTU-13 graphs.")
    parser.add_argument("--graph-dir", default="artifacts/graphs")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/gnn_model.pt")
    parser.add_argument("--output", default="artifacts/reports/gnn_metrics.csv")
    args = parser.parse_args()
    evaluate_gnn_checkpoint(args.graph_dir, args.checkpoint, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
