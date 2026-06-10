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
from gbd_ctu.training.seed import seed_everything

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_torch() -> None:
    if torch is None or NeighborLoader is None:
        raise ImportError("torch and torch-geometric are required for GNN training.")


class _MLPClassifier(torch.nn.Module if torch is not None else object):
    """3-layer MLP that ignores edge_index — pure node-feature baseline."""
    def __init__(self, in_channels: int, hidden: int, out_channels: int, dropout: float):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden), torch.nn.BatchNorm1d(hidden), torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, out_channels),
        )
    def forward(self, data):
        return self.net(data.x)


def _build_model(family: str, in_channels: int, hidden_channels: int,
                 out_channels: int, heads: int, num_layers: int, dropout: float):
    """Instantiate a GNN model via the shared factory."""
    if family == "mlp":
        return _MLPClassifier(in_channels, hidden_channels, out_channels, dropout)
    config = {
        "model_type": family,
        "graphsage": {"hidden_dim": hidden_channels, "num_layers": num_layers, "dropout": dropout, "out_channels": out_channels},
        "gat": {"hidden_channels": hidden_channels, "embed_channels": hidden_channels // 2, "heads": heads, "dropout": dropout, "out_channels": out_channels},
        "hybrid": {"embed_channels": hidden_channels // 2, "dropout": dropout, "out_channels": out_channels},
    }
    return build_gnn_from_config(in_channels=in_channels, config=config)


def _scenario_focal_alpha(graph) -> float:
    """Compute focal alpha from actual botnet rate: alpha = 1 - botnet_rate.

    This gives higher weight to the minority (botnet) class automatically,
    adapting to each scenario's imbalance rather than using a fixed global value.
    Clamped to [0.5, 0.999] so balanced scenarios still get some weighting.
    """
    labels = graph.y[graph.train_mask]
    if labels.numel() == 0:
        return 0.75
    botnet_rate = float(labels.float().mean().item())
    return float(np.clip(1.0 - botnet_rate, 0.5, 0.999))


def _find_optimal_threshold(model, graph, device) -> float:
    """Find the F1-maximising classification threshold on the val split.

    Sweeps 99 candidate thresholds and returns the one with the highest
    macro-F1 on the validation mask, defaulting to 0.5 if val is empty.
    """
    model.eval()
    with torch.no_grad():
        g = graph.to(device)
        probs = torch.softmax(model(g), dim=-1)[:, 1].cpu().numpy()
        labels = g.y.cpu().numpy()
        mask = g.val_mask.cpu().numpy().astype(bool)
    if not mask.any():
        return 0.5
    p, y = probs[mask], labels[mask]
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (p >= t).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def _make_synthetic_graph(n_nodes: int = 50, n_features: int = 35, seed: int = 0):
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


def _evaluate_split(model, graphs, split: str, device: "torch.device | None" = None,
                    thresholds: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Run full-graph inference and return per-scenario metric dicts.

    When ``thresholds`` is provided each scenario uses its val-tuned threshold
    instead of the default 0.5, which significantly improves F1 on rare-botnet
    scenarios.
    """
    model.eval()
    records: list[dict[str, Any]] = []
    use_amp = device is not None and device.type == "cuda"
    ctx = torch.amp.autocast(device_type="cuda") if use_amp else torch.amp.autocast(device_type="cpu", enabled=False)
    with torch.no_grad(), ctx:
        for graph in graphs:
            if device is not None:
                graph = graph.to(device)
            logits = model(graph)
            probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels = graph.y.cpu().numpy()
            mask = getattr(graph, f"{split}_mask").cpu().numpy().astype(bool)
            if not mask.any():
                continue
            scenario_name = getattr(graph, "scenario", "unknown")
            threshold = (thresholds or {}).get(scenario_name, 0.5)
            metrics = classification_metrics(labels[mask], probabilities[mask],
                                             threshold=threshold)
            metrics.update({"model": "gnn", "scenario": scenario_name,
                            "split": split, "threshold": threshold})
            records.append(metrics)
    return records


def _train_one_epoch(model, graphs, criterion, optimizer, batch_size: int,
                     num_neighbors: list[int], device: "torch.device | None" = None,
                     focal_gamma: float = 2.0,
                     scaler: "torch.amp.GradScaler | None" = None) -> float:
    """Run one training epoch with per-scenario adaptive focal alpha.

    Each graph gets a FocalLoss whose alpha = 1 - botnet_rate, so rare-botnet
    scenarios (0.1%) receive alpha~0.999 and balanced ones (50%) receive ~0.5.
    Graphs are moved to device one at a time to avoid GPU OOM.
    AMP (autocast + GradScaler) halves activation memory, allowing 1.5M-edge
    graphs to fit in 5.64 GB VRAM.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None and device is not None and device.type == "cuda"
    for graph in graphs:
        if device is not None:
            graph = graph.to(device)
        train_mask = graph.train_mask
        if not train_mask.any():
            continue
        alpha = _scenario_focal_alpha(graph)
        per_graph_criterion = FocalLoss(gamma=focal_gamma, alpha=alpha)
        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(graph)
                loss = per_graph_criterion(logits[train_mask], graph.y[train_mask])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(graph)
            loss = per_graph_criterion(logits[train_mask], graph.y[train_mask])
            loss.backward()
            optimizer.step()
        running_loss += float(loss.item())
        n_batches += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return running_loss / max(n_batches, 1)


def _train_one_epoch_sampled(
    model,
    graphs,
    optimizer,
    device: "torch.device | None",
    focal_gamma: float = 2.0,
    num_neighbors_per_layer: int = 15,
    batch_size: int = 2048,
) -> float:
    """Mini-batch training via per-graph NeighborLoader.

    Processes one graph at a time to avoid the C++ sampling-structure OOM
    that occurs when all 13 CTU-13 graphs are loaded simultaneously.
    NeighborLoader already limits each node to num_neighbors_per_layer
    neighbors per hop, so hub nodes (gateway, DNS) are naturally capped.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0
    n_layers = 3  # SAGE branch depth — keep in sync with _SageBranch

    for graph in graphs:
        train_idx = graph.train_mask.nonzero(as_tuple=False).view(-1)
        if train_idx.numel() == 0:
            continue

        alpha = _scenario_focal_alpha(graph)
        per_graph_criterion = FocalLoss(gamma=focal_gamma, alpha=alpha)

        # NeighborLoader naturally caps each node to num_neighbors_per_layer
        # neighbors per hop — hub nodes (degree > 100) are already limited to
        # at most that many neighbors during sampling without explicit pre-filtering.
        loader = NeighborLoader(
            graph,
            num_neighbors=[num_neighbors_per_layer] * n_layers,
            input_nodes=train_idx,
            batch_size=batch_size,
            shuffle=True,
        )

        for batch in loader:
            batch    = batch.to(device)
            batch_sz = batch.batch_size

            optimizer.zero_grad()
            logits     = model(batch)[:batch_sz]
            labels     = batch.y[:batch_sz]
            train_mask = batch.train_mask[:batch_sz]

            if not train_mask.any():
                continue

            loss = per_graph_criterion(logits[train_mask], labels[train_mask])
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches    += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return running_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _train_gnn_xgb_ensemble(model, graphs, device, checkpoint_path: Path) -> float:
    """Train XGBoost on [raw | GNN_emb | Platt_prob | uncertainty | botnet_ratio | burst | sid].

    Phase 2 improvements:
    - Platt scaling: LogisticRegression(C=1) fitted on val GNN outputs calibrates raw probs
    - GNN uncertainty: binary entropy of softmax (-p log2 p - (1-p) log2(1-p))
    - Scenario botnet ratio: per-scenario train-split botnet rate as a node feature
    - Flow burst indicator: fraction of incident edges that are temporal (type=1)
    - XGBoost: 1500 trees, lr=0.01, depth=7, min_child_weight=5, subsample=0.8, col=0.7
    """
    try:
        import xgboost as xgb  # type: ignore
    except ImportError:
        raise ImportError("xgboost is required for the ensemble step")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    model.eval()

    # --- Pass 1: collect val GNN probs to fit Platt calibrator ---
    _val_gnn_probs, _val_labels = [], []
    with torch.no_grad():
        for graph in graphs:
            g = graph.to(device)
            probs = torch.softmax(model(g), dim=-1)[:, 1].cpu().numpy()
            vm = g.val_mask.cpu().numpy().astype(bool)
            if vm.any():
                _val_gnn_probs.append(probs[vm])
                _val_labels.append(g.y.cpu().numpy()[vm])

    platt = LogisticRegression(C=1.0, max_iter=1000)
    platt.fit(np.concatenate(_val_gnn_probs).reshape(-1, 1), np.concatenate(_val_labels))

    # --- Pass 2: build full feature matrix ---
    X_train_parts, y_train_parts = [], []
    X_val_parts,   y_val_parts   = [], []

    with torch.no_grad():
        for graph in graphs:
            g = graph.to(device)
            n = g.num_nodes
            raw = g.x.cpu().numpy()

            try:
                emb = model.get_embeddings(g).cpu().numpy()
            except AttributeError:
                emb = np.zeros((n, 1), dtype=np.float32)

            probs = torch.softmax(model(g), dim=-1)[:, 1].cpu().numpy()

            # Platt-calibrated probability
            calib = platt.predict_proba(probs.reshape(-1, 1))[:, 1].reshape(-1, 1).astype(np.float32)

            # Binary entropy uncertainty
            p = np.clip(probs, 1e-7, 1 - 1e-7)
            uncertainty = (-p * np.log2(p) - (1 - p) * np.log2(1 - p)).reshape(-1, 1).astype(np.float32)

            # Scenario botnet ratio (train split)
            train_mask = g.train_mask.cpu().numpy().astype(bool)
            br = float(g.y.cpu().numpy()[train_mask].mean()) if train_mask.any() else 0.0
            botnet_ratio_col = np.full((n, 1), br, dtype=np.float32)

            # Flow burst: fraction of incident edges that are temporal edges (type=1)
            is_temporal = (g.edge_type == 1).float()
            ones_e = torch.ones(g.edge_index.shape[1], device=device)
            t_deg = torch.zeros(n, device=device)
            t_deg.scatter_add_(0, g.edge_index[0], is_temporal)
            t_deg.scatter_add_(0, g.edge_index[1], is_temporal)
            tot_deg = torch.zeros(n, device=device)
            tot_deg.scatter_add_(0, g.edge_index[0], ones_e)
            tot_deg.scatter_add_(0, g.edge_index[1], ones_e)
            burst = (t_deg / tot_deg.clamp(min=1.0)).cpu().numpy().reshape(-1, 1).astype(np.float32)

            sid = float(getattr(g, "scenario_id", 0))
            sid_col = np.full((n, 1), sid, dtype=np.float32)

            combined = np.concatenate([
                raw, emb, probs.reshape(-1, 1), calib, uncertainty,
                botnet_ratio_col, burst, sid_col,
            ], axis=1)

            labels = g.y.cpu().numpy()
            val_mask = g.val_mask.cpu().numpy().astype(bool)
            if train_mask.any():
                X_train_parts.append(combined[train_mask])
                y_train_parts.append(labels[train_mask])
            if val_mask.any():
                X_val_parts.append(combined[val_mask])
                y_val_parts.append(labels[val_mask])

    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    X_val   = np.concatenate(X_val_parts)
    y_val   = np.concatenate(y_val_parts)

    scale_pos = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    clf = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=7,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        scale_pos_weight=scale_pos,
        eval_metric="auc",
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_probs_final = clf.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_probs_final))

    ensemble_path = checkpoint_path.with_suffix(".ensemble.json")
    clf.save_model(str(ensemble_path))
    _logger.info("GNN+XGBoost ensemble saved to %s  val_auc=%.4f", ensemble_path, val_auc)
    return val_auc


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
    cosine_T0: int = 20,
    cosine_T_mult: int = 2,
    use_neighbor_sampling: bool = False,
    num_neighbors: list[int] | None = None,
    dry_run: bool = False,
    scenario_ids: list[int] | None = None,
    pretrained_gnn_path: str | Path | None = None,
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
    seed_everything(seed)

    if num_neighbors is None:
        num_neighbors = [10, 5]

    # ---- data ---------------------------------------------------------------
    if dry_run:
        graphs = [_make_synthetic_graph(n_features=35, seed=seed)]
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("training device: %s", device)
    # Graphs stay on CPU; each is moved to device only during its forward pass.
    # AMP GradScaler: halves activation memory so 1.5M-edge graphs fit in 5.64 GB VRAM.
    amp_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    in_channels = int(graphs[0].num_node_features)
    model = _build_model(family, in_channels, hidden_channels, out_channels,
                         heads, num_layers, dropout).to(device)

    # Load pretrained weights if provided — skips GNN training entirely.
    if pretrained_gnn_path is not None:
        ckpt = torch.load(pretrained_gnn_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        _logger.info("loaded pretrained GNN from %s — skipping training", pretrained_gnn_path)
        epochs = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Cosine annealing with warm restarts: escapes local minima that ReduceLROnPlateau
    # gets stuck in after the first step-down. cosine_T0=30, T_mult=2 gives restarts at
    # epochs 30, 90, 210 — wider cycles give longer exploitation phases.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cosine_T0, T_mult=cosine_T_mult, eta_min=lr_scheduler_min_lr,
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
        if use_neighbor_sampling:
            train_loss = _train_one_epoch_sampled(
                model, graphs, optimizer, device,
                focal_gamma=focal_gamma,
                num_neighbors_per_layer=15,
                batch_size=batch_size,
            )
        else:
            train_loss = _train_one_epoch(model, graphs, criterion, optimizer,
                                          batch_size=batch_size, num_neighbors=num_neighbors,
                                          device=device, focal_gamma=focal_gamma,
                                          scaler=amp_scaler)

        val_records = _evaluate_split(model, graphs, split="val", device=device)
        val_auc = float(np.nanmean([r["auc"] for r in val_records])) if val_records else float("nan")
        val_f1 = float(np.nanmean([r["f1"] for r in val_records])) if val_records else float("nan")

        # val loss (full-graph cross-entropy for monitoring)
        val_loss_vals: list[float] = []
        model.eval()
        _val_ctx = (torch.amp.autocast(device_type="cuda")
                    if device.type == "cuda"
                    else torch.amp.autocast(device_type="cpu", enabled=False))
        with torch.no_grad(), _val_ctx:
            for graph in graphs:
                g = graph.to(device)
                if not g.val_mask.any():
                    continue
                logits = model(g)
                vl = float(criterion(logits[g.val_mask], g.y[g.val_mask]).item())
                val_loss_vals.append(vl)
        val_loss = float(np.nanmean(val_loss_vals)) if val_loss_vals else float("nan")

        scheduler.step(epoch)

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

    # ---- Step 1: per-scenario threshold tuning on val set -------------------
    thresholds: dict[str, float] = {}
    for graph in graphs:
        scenario_name = getattr(graph, "scenario", str(getattr(graph, "scenario_id", "unknown")))
        thresholds[scenario_name] = _find_optimal_threshold(model, graph, device)
    _logger.info("optimal val thresholds: %s", thresholds)
    if best_state is not None:
        best_state["thresholds"] = thresholds

    torch.save(best_state, checkpoint)
    history_path = checkpoint.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Per-scenario checkpoints
    for graph in graphs:
        sid = getattr(graph, "scenario_id", 0)
        scenario_ckpt = checkpoint.parent / f"{family}_scenario{sid}.pt"
        torch.save(best_state, scenario_ckpt)

    # ---- Final eval with tuned thresholds -----------------------------------
    val_records = _evaluate_split(model, graphs, split="val", device=device,
                                  thresholds=thresholds)
    report = metrics_frame(val_records)
    if run is not None:
        run.finish()

    # ---- Step 5: GNN + XGBoost ensemble ------------------------------------
    ensemble_auc: float | None = None
    try:
        ensemble_auc = _train_gnn_xgb_ensemble(model, graphs, device, checkpoint)
        _logger.info("GNN+XGBoost ensemble val AUC: %.4f", ensemble_auc)
    except Exception as exc:
        _logger.warning("ensemble training skipped: %s", exc)

    return {
        "checkpoint": str(checkpoint),
        "history_path": str(history_path),
        "best_val_auc": best_val_auc,
        "epochs_trained": len(history),
        "stopped_epoch": stopped_epoch,
        "history": history,
        "validation_report": report.to_dict(orient="records"),
        "thresholds": thresholds,
        "ensemble_val_auc": ensemble_auc,
    }

