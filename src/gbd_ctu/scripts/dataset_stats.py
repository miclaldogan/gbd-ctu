"""GBD-CTU dataset statistics script.

Loads all available CTU-13 scenario flow files, computes per-scenario
statistics, and writes a CSV table and a LaTeX table.

Examples
--------
    python -m gbd_ctu.scripts.dataset_stats --data-dir data/raw

    python -m gbd_ctu.scripts.dataset_stats \\
        --data-dir data/raw \\
        --output results/dataset_stats.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-scenario statistics for all CTU-13 flow files."
    )
    parser.add_argument(
        "--data-dir", default="data/raw",
        help="Root directory containing CTU-13 scenario subdirectories.",
    )
    parser.add_argument(
        "--output", default="results/dataset_stats.csv",
        help="Path for the output CSV file (default: results/dataset_stats.csv).",
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Also write a LaTeX table alongside the CSV.",
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
        from gbd_ctu.data.ctu13_loader import CTU13Loader, dataset_statistics_table
    except ImportError as exc:
        _logger.error("Import failed: %s", exc)
        return 1

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        _logger.error("Data directory not found: %s", data_dir)
        return 1

    _logger.info("Loading scenarios from %s …", data_dir)
    try:
        loader = CTU13Loader(data_root=data_dir)
        df = dataset_statistics_table(loader)
    except FileNotFoundError as exc:
        _logger.error("%s", exc)
        return 1
    except Exception as exc:
        _logger.error("Failed to build statistics table: %s", exc)
        return 1

    if df.empty:
        _logger.error("No scenarios found — check --data-dir.")
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _logger.info("Wrote %s", out_path)

    if args.latex:
        tex_path = out_path.with_suffix(".tex")
        float_cols = [c for c in df.columns if df[c].dtype == float]
        try:
            tex = df.to_latex(index=False, float_format="%.4f", na_rep="—")
        except TypeError:
            tex = df.to_latex(index=False, na_rep="—")
        tex_path.write_text(tex, encoding="utf-8")
        _logger.info("Wrote %s", tex_path)

    print(df.to_string(index=False))
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
