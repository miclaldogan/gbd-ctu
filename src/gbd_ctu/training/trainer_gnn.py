"""GBD-CTU GNN training.

This module trains graph neural networks on CTU-13 IP graphs. Inputs are a graph
directory and model/training hyperparameters; outputs are a saved checkpoint,
training history, and scenario-aware validation metrics.

Training uses ``NeighborLoader`` for mini-batch neighbourhood sampling,
``ReduceLROnPlateau`` for adaptive LR scheduling, and early stopping on
validation AUC.  Weights & Biases logging is skipped gracefully when
``WANDB_MODE=disabled`` is set in the environment or W&B is not installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None
    NeighborLoader = None
    Data = None

try:  # pragma: no cover - optional
    import wandb
except ImportError:
    wandb = None

from gbd_ctu.data.graph_builder import load_graphs
from gbd_ctu.evaluation.metrics import classification_metrics, metrics_frame
from gbd_ctu.models.gnn import build_gnn_from_config
from gbd_ctu.training.losses import FocalLoss

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_torch() -> None:
    if torch is None or NeighborLoader is None:
        raise ImportError("torch and torch-geometric are required for GNN training.")


def _build_model(family: str, in_channels: int, hidden_channels: int,
                 out_channels: int, heads: int, num_layers: int, dropout: float):
    """Instantiate a GNN model via the shared factory."""
    config = {
        "model_type": family,
        "graphsage": {"hidden_dim": hidden_channels, "num_layers": num_layers, "dropout": dropout, "out_channels": out_channels},
        "gat": {"hidden_channels": hidden_channels, "embed_channels": hidden_channels // 2, "heads": heads, "dropout": dropout, "out_channels": out_channels},
        "hybrid": {"embed_channels": hidden_channels // 2, "dropout": dropout, "out_channels": out_channels},
    }
    return build_gnn_from_config(in_channels=in_channels, config=config)


def _make_synthetic_graph(n_nodes: int = 50, n_features: int = 6, seed: int = 0):
    """Build a tiny synthetic PyG graph for dry-run / unit tests."""
    _require_torch()
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.standard_normal((n_nodes, n_features)), dtype=torch.float32)
    # Random directed edges (no self-loops)
    src = torch.randint(0, n_nodes, (n_nodes * 2,))
    dst = torch.randint(0, n_nodes, (n_nodes * 2,))
    mask = src != dst
    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
    y = torch.tensor(rng.integers(0, 2, n_nodes), dtype=torch.long)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:int(n_nodes * 0.6)] = True
    val_mask[int(n_nodes * 0.6):int(n_nodes * 0.8)] = True
    test_mask[int(n_nodes * 0.8):] = True
    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                scenario="synthetic-00", scenario_id=0)


def _evaluate_split(model, graphs, split: str) -> list[dict[str, Any]]:
    """Run full-graph inference and return per-scenario metric dicts."""
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
            metrics.update({"model": "gnn", "scenario": getattr(graph, "scenario", "unknown"), "split": split})
            records.append(metrics)
    return records


def _train_one_epoch(model, graphs, criterion, optimizer, batch_size: int,
                     num_neighbors: list[int]) -> float:
    """Run one training epoch using NeighborLoader mini-batches.

    Falls back transparently to full-graph mini-batching when ``pyg-lib`` /
    ``torch-sparse`` are not installed (``NeighborLoader`` raises
    ``ImportError`` on first iteration in that case).
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    for graph in graphs:
        train_mask = graph.train_mask
        if not train_mask.any():
            continue
        loader = NeighborLoader(
            graph,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=train_mask,
            shuffle=True,
        )
        try:
            for batch in loader:
                optimizer.zero_grad()
                logits = model(batch)
                # Seed nodes are the first batch.batch_size entries
                seed_logits = logits[:batch.batch_size]
                seed_labels = batch.y[:batch.batch_size]
                loss = criterion(seed_logits, seed_labels)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                n_batches += 1
        except ImportError:
            # pyg-lib / torch-sparse not available; fall back to full-graph
            # node-batched training on the complete graph.
            _logger.warning(
                "NeighborLoader requires 'pyg-lib' or 'torch-sparse'; "
                "falling back to full-graph mini-batch training."
            )
            train_indices = train_mask.nonzero(as_tuple=True)[0]
            for i in range(0, max(len(train_indices), 1), batch_size):
                bi = train_indices[i : i + batch_size]
                optimizer.zero_grad()
                logits = model(graph)
                loss = criterion(logits[bi], graph.y[bi])
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                n_batches += 1
    return running_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_gnn(
    graph_dir: str | Path | None = None,
    checkpoint_path: str | Path = "artifacts/checkpoints/gnn_model.pt",
    family: str = "hybrid",
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    hidden_channels: int = 64,
    out_channels: int = 2,
    heads: int = 4,
    num_layers: int = 3,
    dropout: float = 0.3,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    seed: int = 42,
    wandb_project: str | None = None,
    early_stop_patience: int = 10,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_min_lr: float = 1e-5,
    num_neighbors: list[int] | None = None,
    dry_run: bool = False,
    scenario_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Train a GNN on serialized CTU-13 IP graphs with focal loss.

    Parameters
    ----------
    graph_dir:
        Directory of serialized ``.pt`` graph files.  Ignored when
        ``dry_run=True``.
    checkpoint_path:
        Where to write the best-val-AUC checkpoint.
    family:
        Model variant: ``graphsage`` | ``gat`` | ``hybrid``.
    early_stop_patience:
        Number of epochs without val-AUC improvement before training stops.
    num_neighbors:
        Neighbourhood sizes for ``NeighborLoader`` layers (default ``[10, 5]``).
    dry_run:
        When ``True``, skip loading real data: generate a synthetic 50-node
        graph and run exactly 3 epochs.  Useful for CI / smoke tests.
    scenario_ids:
        Optional list of scenario IDs to restrict training.
    """
    _require_torch()
    torch.manual_seed(seed)

    if num_neighbors is None:
        num_neighbors = [10, 5]

    # ---- data ---------------------------------------------------------------
    if dry_run:
        graphs = [_make_synthetic_graph(n_features=6, seed=seed)]
        epochs = 3
        _logger.info("dry-run mode: using synthetic graph for %d epochs", epochs)
    else:
        if graph_dir is None:
            raise ValueError("graph_dir must be provided when dry_run=False.")
        graphs = load_graphs(graph_dir)
        if scenario_ids is not None:
            graphs = [g for g in graphs if getattr(g, "scenario_id", None) in scenario_ids]
            if not graphs:
                raise ValueError(f"No graphs found for scenario_ids={scenario_ids}.")

    in_channels = int(graphs[0].num_node_features)
    model = _build_model(family, in_channels, hidden_channels, out_channels,
                         heads, num_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=lr_scheduler_patience,
        factor=lr_scheduler_factor, min_lr=lr_scheduler_min_lr,
    )
    criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    # ---- W&B init -----------------------------------------------------------
    run = None
    if wandb_project and wandb is not None:
        import os
        if os.environ.get("WANDB_MODE") != "disabled":
            run = wandb.init(project=wandb_project, config={
                "family": family, "epochs": epochs, "batch_size": batch_size,
                "learning_rate": learning_rate, "weight_decay": weight_decay,
                "hidden_channels": hidden_channels, "heads": heads,
                "num_layers": num_layers, "dropout": dropout,
                "focal_gamma": focal_gamma, "focal_alpha": focal_alpha,
            })

    # ---- training loop ------------------------------------------------------
    best_state: dict[str, Any] | None = None
    best_val_auc = float("-inf")
    epochs_no_improve = 0
    history: list[dict[str, float]] = []
    stopped_epoch: int = epochs

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, graphs, criterion, optimizer,
                                      batch_size=batch_size, num_neighbors=num_neighbors)

        val_records = _evaluate_split(model, graphs, split="val")
        val_auc = float(np.nanmean([r["auc"] for r in val_records])) if val_records else float("nan")
        val_f1 = float(np.nanmean([r["f1"] for r in val_records])) if val_records else float("nan")

        # val loss (full-graph cross-entropy for monitoring)
        val_loss_vals: list[float] = []
        model.eval()
        with torch.no_grad():
            for graph in graphs:
                if not graph.val_mask.any():
                    continue
                logits = model(graph)
                vl = float(criterion(logits[graph.val_mask], graph.y[graph.val_mask]).item())
                val_loss_vals.append(vl)
        val_loss = float(np.nanmean(val_loss_vals)) if val_loss_vals else float("nan")

        scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)

        current_lr = float(optimizer.param_groups[0]["lr"])
        record: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_f1": val_f1,
            "lr": current_lr,
        }
        history.append(record)
        _logger.info("epoch %d | train_loss=%.4f val_loss=%.4f val_auc=%.4f val_f1=%.4f",
                     epoch, train_loss, val_loss, val_auc, val_f1)

        # W&B per-epoch logging
        if run is not None:
            run.log(record, step=epoch)
            if epoch % 10 == 0:
                # Confusion-matrix summary every 10 epochs
                from sklearn.metrics import confusion_matrix as sk_cm
                all_preds, all_labels = [], []
                model.eval()
                with torch.no_grad():
                    for graph in graphs:
                        logits = model(graph)
                        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                        preds = (probs >= 0.5).astype(int)
                        mask = graph.val_mask.cpu().numpy().astype(bool)
                        if mask.any():
                            all_preds.extend(preds[mask].tolist())
                            all_labels.extend(graph.y.cpu().numpy()[mask].tolist())
                if all_labels:
                    cm = sk_cm(all_labels, all_preds, labels=[0, 1]).tolist()
                    run.log({"confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None, y_true=all_labels, preds=all_preds,
                        class_names=["benign", "botnet"])}, step=epoch)

        # early stopping
        if not np.isnan(val_auc) and val_auc >= best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "family": family,
                "model_state": model.state_dict(),
                "model_kwargs": {
                    "in_channels": in_channels, "hidden_channels": hidden_channels,
                    "out_channels": out_channels, "heads": heads,
                    "num_layers": num_layers, "dropout": dropout,
                },
                "best_val_auc": best_val_auc,
                "epoch": epoch,
                "seed": seed,
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                _logger.info("Early stopping at epoch %d (no improvement for %d epochs)",
                             epoch, early_stop_patience)
                stopped_epoch = epoch
                break

    # ---- checkpoint ---------------------------------------------------------
    if best_state is None:
        best_state = {
            "family": family,
            "model_state": model.state_dict(),
            "model_kwargs": {
                "in_channels": in_channels, "hidden_channels": hidden_channels,
                "out_channels": out_channels, "heads": heads,
                "num_layers": num_layers, "dropout": dropout,
            },
            "best_val_auc": float("nan"),
            "epoch": epochs,
            "seed": seed,
        }

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint)
    history_path = checkpoint.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Per-scenario checkpoints: {model_type}_scenario{sid}.pt
    for graph in graphs:
        sid = getattr(graph, "scenario_id", 0)
        scenario_ckpt = checkpoint.parent / f"{family}_scenario{sid}.pt"
        torch.save(best_state, scenario_ckpt)

    report = metrics_frame(_evaluate_split(model, graphs, split="val"))
    if run is not None:
        run.finish()

    return {
        "checkpoint": str(checkpoint),
        "history_path": str(history_path),
        "best_val_auc": best_val_auc,
        "epochs_trained": len(history),
        "stopped_epoch": stopped_epoch,
        "history": history,
        "validation_report": report.to_dict(orient="records"),
    }

