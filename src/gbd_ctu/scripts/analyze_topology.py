"""GBD-CTU graph topology analysis script.

Loads serialized CTU-13 scenario graphs, computes topology statistics for each
scenario, and writes a CSV table, a LaTeX table, and one degree-distribution
PNG per scenario.

Examples
--------
# Analyse all graphs in the default location:
    python -m gbd_ctu.scripts.analyze_topology --graph-dir data/processed

# Custom output directory:
    python -m gbd_ctu.scripts.analyze_topology \\
        --graph-dir artifacts/graphs \\
        --output-dir results/topology
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute graph topology statistics for CTU-13 scenario graphs."
    )
    parser.add_argument(
        "--graph-dir", default="data/processed",
        help="Directory containing serialized .pt graph files.",
    )
    parser.add_argument(
        "--output-dir", default="results/topology",
        help="Destination directory for CSV, LaTeX, and PNG outputs.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip degree-distribution PNG generation.",
    )
    parser.add_argument(
        "--loglevel", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        from gbd_ctu.data.graph_builder import load_graphs
        from gbd_ctu.evaluation.graph_stats import (
            degree_distribution_plot,
            scenario_topology_stats,
            topology_stats_table,
        )
    except ImportError as exc:
        _logger.error("Import failed: %s", exc)
        return 1

    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _logger.info("Loading graphs from %s", graph_dir)
    try:
        graphs = load_graphs(graph_dir)
    except Exception as exc:
        _logger.error("Failed to load graphs: %s", exc)
        return 1

    if not graphs:
        _logger.error("No graphs found in %s", graph_dir)
        return 1

    _logger.info("Computing topology stats for %d graphs …", len(graphs))
    all_stats: list[dict] = []
    plots_dir = output_dir / "plots"

    for graph in graphs:
        sid = getattr(graph, "scenario_id", None)
        scenario = getattr(graph, "scenario", str(sid))
        try:
            stats = scenario_topology_stats(graph)
        except Exception as exc:
            _logger.warning("Skipping scenario %s: %s", sid, exc)
            continue

        stats["scenario_id"] = sid
        stats["scenario"] = scenario
        all_stats.append(stats)
        _logger.info(
            "  scenario %s: %d nodes, %d edges, botnet_frac=%.3f",
            sid, stats["n_nodes"], stats["n_edges"], stats.get("botnet_node_frac", float("nan")),
        )

        if not args.no_plots:
            try:
                png = degree_distribution_plot(graph, sid, plots_dir)
                _logger.debug("  Saved %s", png)
            except Exception as exc:
                _logger.warning("  Could not generate plot for scenario %s: %s", sid, exc)

    if not all_stats:
        _logger.error("No topology stats computed.")
        return 1

    df = topology_stats_table(all_stats)

    csv_path = output_dir / "topology_stats.csv"
    df.to_csv(csv_path, index=False)
    _logger.info("Wrote %s", csv_path)

    tex_path = output_dir / "topology_stats.tex"
    float_cols = [c for c in df.columns if df[c].dtype == float]
    float_fmt = {c: "%.4f" for c in float_cols}
    try:
        tex = df.to_latex(index=False, float_format="%.4f", na_rep="—")
    except TypeError:
        tex = df.to_latex(index=False, na_rep="—")
    tex_path.write_text(tex, encoding="utf-8")
    _logger.info("Wrote %s", tex_path)

    print(df.to_string(index=False))
    print(f"\nResults written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
