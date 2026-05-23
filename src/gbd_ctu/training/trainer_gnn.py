"""GBD-CTU GNN training.

This module trains graph neural networks on CTU-13 IP graphs. Inputs are a graph
directory and model/training hyperparameters; outputs are a saved checkpoint,
training history, and scenario-aware validation metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch_geometric.loader import DataLoader
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None
    DataLoader = None

try:  # pragma: no cover - optional during static inspection
    import wandb
except ImportError:
    wandb = None

from gbd_ctu.data.graph_builder import load_graphs
from gbd_ctu.evaluation.metrics import classification_metrics, metrics_frame
from gbd_ctu.models.gnn.gat import GATNodeClassifier
from gbd_ctu.models.gnn.graphsage import GraphSAGENodeClassifier
from gbd_ctu.models.gnn.hybrid import GraphSageGATHybridClassifier
from gbd_ctu.training.losses import FocalLoss


MODEL_FAMILIES = {
    "graphsage": GraphSAGENodeClassifier,
    "gat": GATNodeClassifier,
    "hybrid": GraphSageGATHybridClassifier,
}


def _require_torch() -> None:
    if torch is None or DataLoader is None:
        raise ImportError("torch and torch-geometric are required for GNN training.")


def _build_model(family: str, in_channels: int, hidden_channels: int, out_channels: int, heads: int, num_layers: int, dropout: float):
    model_class = MODEL_FAMILIES.get(family)
    if model_class is None:
        raise ValueError(f"Unsupported GNN family: {family}")
    if family == "graphsage":
        return model_class(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    if family == "gat":
        return model_class(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
        )
    return model_class(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads,
        dropout=dropout,
    )


def _evaluate_split(model, graphs, split: str) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for graph in graphs:
            logits = model(graph)
            probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels = graph.y.cpu().numpy()
            mask = getattr(graph, f"{split}_mask").cpu().numpy().astype(bool)
            if not mask.any():
                continue
            metrics = classification_metrics(labels[mask], probabilities[mask])
            metrics.update({"model": "gnn", "scenario": graph.scenario, "split": split})
            records.append(metrics)
    return records


def train_gnn(
    graph_dir: str | Path,
    checkpoint_path: str | Path,
    family: str = "hybrid",
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    hidden_channels: int = 64,
    out_channels: int = 2,
    heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = 0.75,
    seed: int = 42,
    wandb_project: str | None = None,
) -> dict[str, Any]:
    """Train a configured GNN family on serialized CTU-13 IP graphs."""

    _require_torch()
    torch.manual_seed(seed)
    graphs = load_graphs(graph_dir)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)
    in_channels = int(graphs[0].num_node_features)
    model = _build_model(
        family=family,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    run = None
    if wandb_project:
        if wandb is None:
            raise ImportError("wandb is requested but not installed.")
        run = wandb.init(project=wandb_project, config={
            "family": family,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "hidden_channels": hidden_channels,
            "heads": heads,
            "num_layers": num_layers,
            "dropout": dropout,
        })

    best_state: dict[str, Any] | None = None
    best_val_auc = float("-inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        val_records = _evaluate_split(model, graphs, split="val")
        mean_val_auc = float(np.nanmean([record["auc"] for record in val_records])) if val_records else float("nan")
        epoch_record = {"epoch": float(epoch), "loss": running_loss / max(len(loader), 1), "val_auc": mean_val_auc}
        history.append(epoch_record)
        if run is not None:
            run.log(epoch_record)
        if not np.isnan(mean_val_auc) and mean_val_auc >= best_val_auc:
            best_val_auc = mean_val_auc
            best_state = {
                "family": family,
                "model_state": model.state_dict(),
                "model_kwargs": {
                    "in_channels": in_channels,
                    "hidden_channels": hidden_channels,
                    "out_channels": out_channels,
                    "heads": heads,
                    "num_layers": num_layers,
                    "dropout": dropout,
                },
                "seed": seed,
            }

    if best_state is None:
        best_state = {
            "family": family,
            "model_state": model.state_dict(),
            "model_kwargs": {
                "in_channels": in_channels,
                "hidden_channels": hidden_channels,
                "out_channels": out_channels,
                "heads": heads,
                "num_layers": num_layers,
                "dropout": dropout,
            },
            "seed": seed,
        }

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint)
    history_path = checkpoint.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    report = metrics_frame(_evaluate_split(model, graphs, split="val"))
    if run is not None:
        run.finish()
    return {
        "checkpoint": str(checkpoint),
        "history_path": str(history_path),
        "validation_report": report.to_dict(orient="records"),
    }
