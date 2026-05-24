"""GBD-CTU evaluation script.

This script evaluates a trained GNN checkpoint across CTU-13 scenarios. Inputs
are a graph directory, checkpoint path, and output path; outputs are a printed
results table plus CSV and JSON scenario reports.

Examples
--------
# Evaluate a trained graphsage checkpoint
python -m gbd_ctu.scripts.evaluate --checkpoint artifacts/checkpoints/graphsage_1.pt --model graphsage
"""

from __future__ import annotations

import argparse

from gbd_ctu.evaluation.scenario_eval import evaluate_gnn_checkpoint


def main() -> int:
    """Parse CLI arguments and run scenario-wise GNN evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a GNN checkpoint on CTU-13 graphs.")
    parser.add_argument("--graph-dir", default="artifacts/graphs",
                        help="Directory with serialised .pt graph files.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the trained GNN checkpoint (.pt).")
    parser.add_argument("--model", default=None,
                        help="Model family name override (graphsage | gat | hybrid). "
                             "Defaults to the value stored inside the checkpoint.")
    parser.add_argument("--output", default="artifacts/reports/gnn_metrics.csv",
                        help="Destination CSV path for the results table.")
    args = parser.parse_args()

    report = evaluate_gnn_checkpoint(
        graph_dir=args.graph_dir,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_name=args.model,
    )
    print(report.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
