"""GBD-CTU GNN training script.

This script trains a configured GNN on serialized CTU-13 graphs. Inputs are a
graph directory, checkpoint path, and hyperparameters; outputs are a checkpoint
and a history JSON.

Examples
--------
# Dry-run (no real data needed) — 3 synthetic epochs
python -m gbd_ctu.scripts.train_gnn --model graphsage --scenario 1 --dry-run

# Full training run on scenario 1
python -m gbd_ctu.scripts.train_gnn --model graphsage --scenario 1 --epochs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbd_ctu.training.trainer_gnn import train_gnn


_FAMILIES = ["graphsage", "gat", "hybrid", "mlp"]


def main() -> int:
    """Parse CLI arguments and train the requested GNN model."""

    parser = argparse.ArgumentParser(description="Train a GNN on CTU-13 IP graphs.")
    parser.add_argument("--graph-dir", default="artifacts/graphs",
                        help="Directory with serialised .pt graph files.")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint output path. Defaults to "
                             "artifacts/checkpoints/{model}_{scenario}.pt")
    # --model is the canonical flag; --family is kept for backward-compat.
    _model_grp = parser.add_mutually_exclusive_group()
    _model_grp.add_argument("--model", dest="model", choices=_FAMILIES,
                            help="GNN architecture to train.")
    _model_grp.add_argument("--family", dest="model", choices=_FAMILIES,
                            help=argparse.SUPPRESS)  # legacy alias
    parser.set_defaults(model="hybrid")
    parser.add_argument("--scenario", type=int, default=None,
                        help="CTU-13 scenario ID to restrict training data.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default=None,
                        help="W&B project name (omit to disable W&B).")
    parser.add_argument("--early-stop-patience", type=int, default=10,
                        help="Epochs without val-AUC improvement before stopping.")
    parser.add_argument("--cosine-t0", type=int, default=20,
                        help="CosineAnnealingWarmRestarts first cycle length.")
    parser.add_argument("--cosine-t-mult", type=int, default=2,
                        help="CosineAnnealingWarmRestarts cycle multiplier.")
    parser.add_argument("--neighbor-sampling", action="store_true",
                        help="Use NeighborLoader mini-batch training (15 nbrs/layer).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip real data; run 3 epochs on a synthetic graph.")
    parser.add_argument("--pretrained-gnn", default=None, metavar="PATH",
                        help="Load pretrained GNN checkpoint and skip GNN training; "
                             "only retrain the XGBoost ensemble.")
    args = parser.parse_args()

    # Build default checkpoint path
    if args.checkpoint is None:
        scenario_tag = str(args.scenario) if args.scenario is not None else "all"
        args.checkpoint = f"artifacts/checkpoints/{args.model}_{scenario_tag}.pt"

    result = train_gnn(
        graph_dir=args.graph_dir,
        checkpoint_path=args.checkpoint,
        family=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        seed=args.seed,
        wandb_project=args.wandb_project,
        early_stop_patience=args.early_stop_patience,
        cosine_T0=args.cosine_t0,
        cosine_T_mult=args.cosine_t_mult,
        use_neighbor_sampling=args.neighbor_sampling,
        dry_run=args.dry_run,
        scenario_ids=[args.scenario] if args.scenario is not None else None,
        pretrained_gnn_path=args.pretrained_gnn,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
