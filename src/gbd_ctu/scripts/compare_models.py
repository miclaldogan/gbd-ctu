"""GBD-CTU model comparison script.

Loads all ``results_*.csv`` files from a results directory, builds wide-format
comparison tables (one per metric), runs a Wilcoxon signed-rank test of the
Hybrid GNN against the best baseline, saves figures, and prints a summary.

Examples
--------
python -m gbd_ctu.scripts.compare_models --results-dir artifacts/reports/
python -m gbd_ctu.scripts.compare_models --results-dir artifacts/reports/ --output-dir artifacts/comparison/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbd_ctu.evaluation.compare import run_comparison


def main() -> int:
    """Parse CLI arguments and generate comparison tables + figures."""

    parser = argparse.ArgumentParser(
        description="Generate GNN vs baseline comparison tables and figures."
    )
    parser.add_argument(
        "--results-dir",
        default="artifacts/reports",
        help="Directory containing results_*.csv files (default: artifacts/reports).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for tables and figures "
             "(default: same as --results-dir).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    result = run_comparison(results_dir=results_dir, output_dir=output_dir)

    # Print comparison tables
    import pandas as pd
    for metric, records in result["tables"].items():
        if records:
            print(f"\n=== {metric.upper()} Comparison ===")
            print(pd.DataFrame(records).to_string(index=False))

    # Print Wilcoxon result
    wx = result["wilcoxon"]
    print("\n=== Statistical Significance (Wilcoxon Signed-Rank Test) ===")
    print(f"  Hybrid GNN vs best baseline — n={wx['n_scenarios']} scenarios")
    if wx['p_value'] == wx['p_value']:  # not NaN
        sig = "YES" if wx['significant'] else "NO"
        print(f"  statistic={wx['statistic']:.4f}  p-value={wx['p_value']:.4f}  significant={sig}")
    print(f"  {wx['note']}")

    # Print saved artefacts
    print("\n=== Saved Artefacts ===")
    for k, v in result["csv_paths"].items():
        print(f"  [{k}] {v}")
    print(f"  [combined] {result['combined_csv']}")
    print(f"  [report]   {result['markdown_report']}")
    for fp in result["figures"]:
        print(f"  [figure]   {fp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
