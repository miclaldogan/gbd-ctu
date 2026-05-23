"""GBD-CTU GNN training script.

This script trains a configured GNN on serialized CTU-13 graphs. Inputs are a
graph directory, checkpoint path, and hyperparameters; outputs are a checkpoint
and a history JSON.
"""

from __future__ import annotations

import argparse
import json

from gbd_ctu.training.trainer_gnn import train_gnn


def main() -> int:
    """Parse CLI arguments and train the requested GNN family."""

    parser = argparse.ArgumentParser(description="Train a GNN on CTU-13 IP graphs.")
    parser.add_argument("--graph-dir", default="artifacts/graphs")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/gnn_model.pt")
    parser.add_argument("--family", default="hybrid", choices=["graphsage", "gat", "hybrid"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_gnn(
        graph_dir=args.graph_dir,
        checkpoint_path=args.checkpoint,
        family=args.family,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
