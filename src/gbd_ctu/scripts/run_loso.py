"""GBD-CTU leave-one-scenario-out evaluation script.

Trains a GNN model on all CTU-13 scenarios except one, evaluates on the
held-out scenario, and repeats for every scenario.  Optionally also runs
cross-family generalisation evaluation.

Examples
--------
# LOSO with default hybrid model:
    python -m gbd_ctu.scripts.run_loso --graph-dir data/processed

# LOSO + cross-family with GAT, 30 training epochs:
    python -m gbd_ctu.scripts.run_loso \\
        --graph-dir data/processed \\
        --model-type gat \\
        --epochs 30 \\
        --output-dir results/loso \\
        --cross-family
"""

from __future__ import annotations

import argparse
import logging
import sys

_logger = logging.getLogger(__name__)


def _build_trainer(family: str, epochs: int, lr: float, seed: int):
    """Return a ``trainer(model, graphs) -> None`` closure."""
    try:
        import torch
        from gbd_ctu.training.losses import FocalLoss
        from gbd_ctu.training.trainer_gnn import _train_one_epoch
    except ImportError as exc:
        raise ImportError(
            "torch and torch-geometric are required for LOSO training."
        ) from exc

    def trainer(model, train_graphs):
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = FocalLoss(gamma=2.0, alpha=0.25)
        model.train()
        for _ in range(epochs):
            _train_one_epoch(model, train_graphs, criterion, optimizer,
                             batch_size=512, num_neighbors=[10, 5])

    return trainer


def _build_model_factory(family: str, hidden_channels: int, out_channels: int,
                         heads: int, num_layers: int, dropout: float):
    """Return a ``model_factory(in_channels) -> nn.Module`` closure."""
    try:
        from gbd_ctu.training.trainer_gnn import _build_model
    except ImportError as exc:
        raise ImportError(
            "torch and torch-geometric are required for LOSO evaluation."
        ) from exc

    def model_factory(in_channels: int):
        return _build_model(family, in_channels, hidden_channels, out_channels,
                            heads, num_layers, dropout)

    return model_factory


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Leave-one-scenario-out (LOSO) GNN evaluation on CTU-13 graphs."
    )
    parser.add_argument(
        "--graph-dir", required=True,
        help="Directory containing serialized .pt graph files.",
    )
    parser.add_argument(
        "--model-type", default="hybrid",
        choices=["graphsage", "gat", "hybrid"],
        help="GNN architecture to use (default: hybrid).",
    )
    parser.add_argument(
        "--output-dir", default="results/loso",
        help="Directory where result CSVs and summary JSON are written.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Training epochs per LOSO fold (default: 20).",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--hidden-channels", type=int, default=64,
        help="Hidden channel width (default: 64).",
    )
    parser.add_argument(
        "--heads", type=int, default=4,
        help="GAT attention heads (default: 4).",
    )
    parser.add_argument(
        "--num-layers", type=int, default=3,
        help="Number of GNN message-passing layers (default: 3).",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Dropout rate (default: 0.3).",
    )
    parser.add_argument(
        "--cross-family", action="store_true",
        help="Also run cross-malware-family evaluation after LOSO.",
    )
    args = parser.parse_args()

    try:
        from gbd_ctu.data.graph_builder import load_graphs
        from gbd_ctu.data.ctu13_loader import SCENARIO_FAMILY
        from gbd_ctu.evaluation.cross_scenario import (
            cross_family_eval,
            leave_one_scenario_out,
        )
    except ImportError as exc:
        _logger.error("Import failed: %s", exc)
        return 1

    graphs = load_graphs(args.graph_dir)
    if not graphs:
        _logger.error("No graphs found in %s", args.graph_dir)
        return 1

    _logger.info("Loaded %d graphs from %s", len(graphs), args.graph_dir)

    model_factory = _build_model_factory(
        family=args.model_type,
        hidden_channels=args.hidden_channels,
        out_channels=2,
        heads=args.heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    trainer = _build_trainer(
        family=args.model_type,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
    )

    _logger.info("Starting LOSO evaluation (%d folds, %d epochs each) ...",
                 len(graphs), args.epochs)
    loso_df = leave_one_scenario_out(
        graphs=graphs,
        model_factory=model_factory,
        trainer=trainer,
        output_dir=args.output_dir,
    )
    _logger.info("LOSO complete — %d rows written to %s", len(loso_df), args.output_dir)

    if args.cross_family:
        _logger.info("Starting cross-family evaluation ...")
        cross_df = cross_family_eval(
            graphs=graphs,
            scenario_family_map=SCENARIO_FAMILY,
            model_factory=model_factory,
            trainer=trainer,
            output_dir=args.output_dir,
        )
        _logger.info("Cross-family complete — %d rows", len(cross_df))

    return 0


if __name__ == "__main__":
    sys.exit(main())
