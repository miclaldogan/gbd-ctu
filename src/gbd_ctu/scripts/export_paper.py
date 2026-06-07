"""GBD-CTU paper artifact export CLI.

Generates all publication-ready LaTeX tables and 300-DPI figures from
populated results CSVs in a single command.

Expected directory layout
--------------------------
::

    results/
        results_hybrid.csv       # or any results_*.csv
        results_graphsage.csv
        baseline_cv_metrics.csv  # from train_baselines
        mlp_metrics.csv          # from train_mlp (optional)
        ablation_summary.csv     # from run_ablation_suite (optional)
        stats_significance.csv   # from full_significance_table (optional)

Outputs written to ``paper/figures/`` (or ``--output-dir``)
-------------------------------------------------------------
Tables (LaTeX .tex):
    table_results.tex           Main results (bold best)
    table_ablation.tex          Ablation results
    table_significance.tex      Wilcoxon + Bonferroni
    table_dataset.tex           Dataset statistics

Architecture diagrams (.dot):
    architecture_hybrid.dot
    architecture_gat.dot
    architecture_graphsage.dot
    architecture_mlp.dot

Figures (PNG, 300 DPI):
    heatmap_auc.png             Scenario × model AUC heatmap
    heatmap_f1.png
    ablation_bar.png            Ablation bar chart

Examples
--------
    python -m gbd_ctu.scripts.export_paper --results-dir results/

    python -m gbd_ctu.scripts.export_paper \\
        --results-dir results/ \\
        --output-dir paper/figures/ \\
        --skip-figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export all paper tables and figures from results CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir", default="results",
        help="Directory containing results_*.csv and optional extras.",
    )
    p.add_argument(
        "--output-dir", default="paper/figures",
        help="Destination directory for all outputs.",
    )
    p.add_argument(
        "--skip-figures", action="store_true",
        help="Skip figure generation (tables only).",
    )
    p.add_argument(
        "--skip-tables", action="store_true",
        help="Skip table generation (figures only).",
    )
    p.add_argument(
        "--loglevel", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _load_results(results_dir: Path) -> "pd.DataFrame | None":
    """Load and concatenate all results_*.csv, baseline_cv_metrics.csv, mlp_metrics.csv."""
    import pandas as pd
    frames = []
    for pattern in ("results_*.csv", "baseline_cv_metrics.csv", "mlp_metrics.csv"):
        for f in sorted(results_dir.glob(pattern)):
            try:
                df = pd.read_csv(f)
                if "model" in df.columns:
                    frames.append(df)
                    _logger.info("Loaded %s (%d rows)", f.name, len(df))
            except Exception as exc:
                _logger.warning("Skipping %s: %s", f.name, exc)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _load_csv_optional(path: Path) -> "pd.DataFrame | None":
    if not path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        _logger.info("Loaded %s (%d rows)", path.name, len(df))
        return df
    except Exception as exc:
        _logger.warning("Failed to load %s: %s", path.name, exc)
        return None


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        from gbd_ctu.evaluation.paper_artifacts import (
            export_results_table,
            export_ablation_table,
            export_significance_table,
            export_dataset_table,
            draw_model_architecture,
            scenario_heatmap_figure,
            ablation_bar_figure,
        )
    except ImportError as exc:
        _logger.error("Import failed: %s", exc)
        return 1

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        _logger.error("Results directory not found: %s", results_dir)
        return 1

    n_written = 0

    # ---- Tables -------------------------------------------------------------
    if not args.skip_tables:
        metrics_df = _load_results(results_dir)

        if metrics_df is not None:
            try:
                export_results_table(metrics_df, output_dir / "table_results.tex")
                n_written += 1
            except Exception as exc:
                _logger.warning("export_results_table failed: %s", exc)
        else:
            _logger.warning("No results CSVs found in %s; skipping results table.", results_dir)

        abl_df = _load_csv_optional(results_dir / "ablation_summary.csv")
        if abl_df is not None:
            try:
                export_ablation_table(abl_df, output_dir / "table_ablation.tex")
                n_written += 1
            except Exception as exc:
                _logger.warning("export_ablation_table failed: %s", exc)

        sig_df = _load_csv_optional(results_dir / "stats_significance.csv")
        if sig_df is None:
            # Also try the stats module output name
            sig_df = _load_csv_optional(results_dir / "significance_table.csv")
        if sig_df is not None:
            try:
                export_significance_table(sig_df, output_dir / "table_significance.tex")
                n_written += 1
            except Exception as exc:
                _logger.warning("export_significance_table failed: %s", exc)

        dataset_df = _load_csv_optional(results_dir / "dataset_stats.csv")
        if dataset_df is not None:
            try:
                export_dataset_table(dataset_df, output_dir / "table_dataset.tex")
                n_written += 1
            except Exception as exc:
                _logger.warning("export_dataset_table failed: %s", exc)

        # Architecture .dot files — always available
        for model_type in ("hybrid", "gat", "graphsage", "mlp"):
            try:
                draw_model_architecture(
                    model_type,
                    output_dir / f"architecture_{model_type}.dot",
                )
                n_written += 1
            except Exception as exc:
                _logger.warning("draw_model_architecture(%s) failed: %s", model_type, exc)

    # ---- Figures ------------------------------------------------------------
    if not args.skip_figures:
        metrics_df = _load_results(results_dir)

        if metrics_df is not None:
            for metric in ("auc", "f1"):
                if metric not in metrics_df.columns:
                    continue
                try:
                    scenario_heatmap_figure(metrics_df, metric=metric, output_dir=output_dir)
                    n_written += 1
                except Exception as exc:
                    _logger.warning("scenario_heatmap_figure(%s) failed: %s", metric, exc)

        abl_df = _load_csv_optional(results_dir / "ablation_summary.csv")
        if abl_df is not None:
            try:
                ablation_bar_figure(abl_df, output_dir=output_dir)
                n_written += 1
            except Exception as exc:
                _logger.warning("ablation_bar_figure failed: %s", exc)

    _logger.info("Done — wrote %d artifact(s) to %s", n_written, output_dir)
    if n_written == 0:
        _logger.warning(
            "No artifacts written. Check that %s contains results_*.csv files.",
            results_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
