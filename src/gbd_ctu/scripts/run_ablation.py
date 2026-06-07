"""GBD-CTU ablation study script.

Runs a suite of ablation experiments defined in a YAML config file and writes
summary results (mean ± std across scenarios) to CSV, JSON, and optionally
a LaTeX table.

Examples
--------
# Run all ablations using the bundled config:
    python -m gbd_ctu.scripts.run_ablation \\
        --graph-dir data/processed \\
        --output-dir results/ablation

# Dry-run: print the ablation plan without training:
    python -m gbd_ctu.scripts.run_ablation \\
        --graph-dir data/processed \\
        --dry-run

# Use a custom config:
    python -m gbd_ctu.scripts.run_ablation \\
        --config my_ablations.yaml \\
        --graph-dir data/processed

LaTeX export
------------
After the run, convert the output CSV to a LaTeX table::

    import pandas as pd
    df = pd.read_csv("results/ablation/ablation_results.csv")
    print(df.to_latex(index=False, float_format="%.4f"))

Or use the ``--latex`` flag to write the table directly::

    python -m gbd_ctu.scripts.run_ablation \\
        --graph-dir data/processed \\
        --output-dir results/ablation \\
        --latex
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)


def _print_plan(configs) -> None:
    """Print the ablation plan without running anything."""
    print(f"{'Name':<22} {'Model':<12} {'Graph transforms':<35} {'Loss'}")
    print("-" * 85)
    for cfg in configs:
        mk = cfg.model_kwargs
        gk = cfg.graph_kwargs
        tk = cfg.trainer_kwargs
        model_str = mk.get("type", "hybrid")
        if "num_layers" in mk:
            model_str += f"/{mk['num_layers']}L"
        if "heads" in mk:
            model_str += f"/{mk['heads']}H"
        graph_parts = []
        if not gk.get("use_edge_attr", True):
            graph_parts.append("no_edge_attr")
        if gk.get("node_features", "all") != "all":
            graph_parts.append(f"feats={gk['node_features']}")
        if gk.get("undirected", False):
            graph_parts.append("undirected")
        if gk.get("self_loops", False):
            graph_parts.append("self_loops")
        graph_str = ",".join(graph_parts) if graph_parts else "default"
        loss_kw = tk.get("loss", {})
        loss_str = loss_kw.get("type", "focal")
        print(f"{cfg.name:<22} {model_str:<12} {graph_str:<35} {loss_str}")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run GBD-CTU ablation study and write summary results."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to ablation YAML config. "
            "Defaults to the bundled configs/ablation.yaml."
        ),
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Directory containing serialized .pt graph files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ablation",
        help="Directory where result CSV / JSON are written (default: results/ablation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs per ablation (default: 20).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ablation plan without running any training.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Write a LaTeX table of results to <output-dir>/ablation_results.tex.",
    )
    args = parser.parse_args()

    try:
        from gbd_ctu.evaluation.ablation import AblationConfig, configs_from_yaml, run_ablation_suite
    except ImportError as exc:
        _logger.error("Import failed: %s", exc)
        return 1

    configs = configs_from_yaml(args.config)
    _logger.info("Loaded %d ablation configs from %s",
                 len(configs), args.config or "bundled ablation.yaml")

    if args.dry_run:
        _print_plan(configs)
        return 0

    # Propagate per-run epochs into trainer_kwargs if not already set
    for cfg in configs:
        cfg.trainer_kwargs.setdefault("epochs", args.epochs)

    result = run_ablation_suite(
        configs=configs,
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    if args.latex and not result.empty:
        tex_path = Path(args.output_dir) / "ablation_results.tex"
        tex = result.to_latex(index=False, float_format="%.4f", na_rep="-")
        tex_path.write_text(tex, encoding="utf-8")
        _logger.info("LaTeX table written to %s", tex_path)

    _logger.info("Ablation complete — %d rows in %s", len(result), args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
