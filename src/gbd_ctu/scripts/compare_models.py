"""GBD-CTU model comparison script.

This script merges a GNN report and a baseline report into a single comparison
table. Inputs are two report files; outputs are combined CSV and JSON tables.
"""

from __future__ import annotations

import argparse

from gbd_ctu.evaluation.compare import compare_reports


def main() -> int:
    """Parse CLI arguments and write a combined comparison report."""

    parser = argparse.ArgumentParser(description="Merge GNN and baseline CTU-13 reports.")
    parser.add_argument("--gnn-report", default="artifacts/reports/gnn_metrics.csv")
    parser.add_argument("--baseline-report", default="artifacts/reports/baseline_metrics.csv")
    parser.add_argument("--output", default="artifacts/reports/comparison.csv")
    args = parser.parse_args()
    compare_reports(args.gnn_report, args.baseline_report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
