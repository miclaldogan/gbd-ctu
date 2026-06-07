"""GBD-CTU hyperparameter search CLI.

Run Optuna TPE search over GNN hyperparameters with MedianPruner.

Examples
--------
    # First run (50 trials, in-memory):
    python -m gbd_ctu.scripts.run_hparam_search \\
        --graph-dir results/graphs --model-type hybrid --n-trials 50

    # Persistent storage (resume-able):
    python -m gbd_ctu.scripts.run_hparam_search \\
        --graph-dir results/graphs \\
        --storage sqlite:///results/hparam/study.db \\
        --resume

    # Dry-run (tiny synthetic graph, no real data needed):
    python -m gbd_ctu.scripts.run_hparam_search --dry-run --n-trials 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for GBD-CTU GNN models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--graph-dir",
        default="results/graphs",
        help="Directory containing serialised .pt graph files.",
    )
    p.add_argument(
        "--model-type",
        default="hybrid",
        choices=["hybrid", "gat", "graphsage"],
        help="GNN architecture to tune.",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (each trial uses seed + trial_number).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs per trial.",
    )
    p.add_argument(
        "--storage",
        default=None,
        help=(
            "Optuna storage URL, e.g. sqlite:///results/hparam/study.db. "
            "When omitted, an in-memory storage is used."
        ),
    )
    p.add_argument(
        "--study-name",
        default="gbd_ctu_hparam",
        help="Optuna study name (used to identify the study in the storage).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing study from --storage (requires --storage).",
    )
    p.add_argument(
        "--output",
        default="results/hparam",
        help="Output directory for best_params.json.",
    )
    p.add_argument(
        "--wandb-project",
        default=None,
        help="Optional Weights & Biases project for trial logging.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on a tiny synthetic graph (no real data required).",
    )
    p.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        from gbd_ctu.training.hparam_search import run_hparam_search
    except ImportError as exc:
        _logger.error("Import failed: %s", exc)
        return 1

    if args.resume and not args.storage:
        _logger.error("--resume requires --storage.")
        return 1

    if not args.dry_run:
        graph_dir = Path(args.graph_dir)
        if not graph_dir.exists():
            _logger.error("Graph directory not found: %s", graph_dir)
            return 1
    else:
        graph_dir = None

    try:
        study = run_hparam_search(
            graph_dir=graph_dir,
            model_type=args.model_type,
            n_trials=args.n_trials,
            seed=args.seed,
            storage=args.storage,
            study_name=args.study_name,
            output_dir=args.output,
            wandb_project=args.wandb_project,
            dry_run=args.dry_run,
            epochs=args.epochs,
        )
    except ImportError as exc:
        _logger.error("%s", exc)
        return 1
    except Exception as exc:
        _logger.error("Search failed: %s", exc)
        return 1

    best = study.best_trial
    params_path = Path(args.output) / "best_params.json"
    print(f"\nBest trial: #{best.number}  val_auc={best.value:.4f}")
    print(json.dumps(best.params, indent=2))
    print(f"\nFull results written to {params_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
