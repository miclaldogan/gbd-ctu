"""GBD-CTU baseline training.

This module trains classical baselines on node-level IP graph features. Inputs
are serialized graphs and baseline hyperparameters; outputs are saved reports and
scenario-wise evaluation records.

Public API
----------
``train_baselines``
    Train Random Forest and XGBoost with 5-fold cross-validation.

``train_mlp``
    Train a deep MLP (FocalLoss, ReduceLROnPlateau, early stopping) on the
    same node features — the graph-structure-free ablation baseline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from gbd_ctu.data.graph_builder import load_graphs
from gbd_ctu.evaluation.metrics import classification_metrics, metrics_frame
from gbd_ctu.models.baselines.random_forest_clf import RandomForestBaseline
from gbd_ctu.models.baselines.xgboost_clf import XGBoostBaseline
from gbd_ctu.training.seed import seed_everything

_logger = logging.getLogger(__name__)


def _stack_split(graphs, split: str) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for graph in graphs:
        mask = getattr(graph, f"{split}_mask").cpu().numpy().astype(bool)
        if not mask.any():
            continue
        features.append(graph.x.cpu().numpy()[mask])
        labels.append(graph.y.cpu().numpy()[mask])
    if not features:
        width = int(graphs[0].num_node_features) if graphs else 0
        return np.empty((0, width), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.vstack(features), np.concatenate(labels)


def _evaluate_by_scenario(graphs, estimator, model_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for graph in graphs:
        mask = graph.test_mask.cpu().numpy().astype(bool)
        if not mask.any():
            continue
        features = graph.x.cpu().numpy()[mask]
        labels = graph.y.cpu().numpy()[mask]
        probabilities = estimator.predict_proba(features)[:, 1]
        metrics = classification_metrics(labels, probabilities)
        metrics.update({
            "model": model_name,
            "scenario": getattr(graph, "scenario", f"scenario-{getattr(graph, 'scenario_id', 0)}"),
            "scenario_id": int(getattr(graph, "scenario_id", 0)),
            "split": "test",
            "n_botnet_nodes": int((labels == 1).sum()),
            "n_total_nodes": int(len(labels)),
        })
        records.append(metrics)
    return records


def train_baselines(
    graph_dir: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    rf_n_estimators: int = 400,
    rf_max_depth: int | None = None,
    xgb_n_estimators: int = 250,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.05,
    xgb_subsample: float = 0.8,
    xgb_colsample_bytree: float = 0.8,
    xgb_tree_method: str = "hist",
    scenario_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Train Random Forest and XGBoost baselines on serialized CTU-13 graphs.

    Parameters
    ----------
    scenario_ids:
        Optional list of scenario IDs to restrict training/evaluation.
        When ``None`` all graphs in ``graph_dir`` are used.
    """

    seed_everything(random_state)
    graphs = load_graphs(graph_dir)
    if scenario_ids is not None:
        graphs = [g for g in graphs if getattr(g, "scenario_id", None) in scenario_ids]
        if not graphs:
            raise ValueError(f"No graphs found for scenario_ids={scenario_ids} in {graph_dir}.")
    x_train, y_train = _stack_split(graphs, split="train")
    if x_train.size == 0:
        raise ValueError("No training samples were found in the graph artifacts.")
    models = [
        (
            "random_forest",
            RandomForestBaseline(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                random_state=random_state,
            ),
        )
    ]
    try:
        models.append(
            (
                "xgboost",
                XGBoostBaseline(
                    n_estimators=xgb_n_estimators,
                    max_depth=xgb_max_depth,
                    learning_rate=xgb_learning_rate,
                    subsample=xgb_subsample,
                    colsample_bytree=xgb_colsample_bytree,
                    random_state=random_state,
                    tree_method=xgb_tree_method,
                ),
            )
        )
    except ImportError:
        pass

    fold_records: list[dict[str, Any]] = []
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for model_name, estimator in models:
        for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
            x_fold_train, y_fold_train = x_train[train_idx], y_train[train_idx]
            x_fold_val, y_fold_val = x_train[val_idx], y_train[val_idx]

            estimator.fit(x_fold_train, y_fold_train)

            probabilities = estimator.predict_proba(x_fold_val)[:, 1]
            metrics = classification_metrics(y_fold_val, probabilities)
            metrics["model"] = model_name
            metrics["fold"] = fold
            fold_records.append(metrics)

        joblib.dump(estimator, destination / f"{model_name}.joblib")

    scenario_records: list[dict[str, Any]] = []
    for model_name, estimator in models:
        scenario_records.extend(_evaluate_by_scenario(graphs, estimator, model_name))

    if scenario_records:
        scenario_report = pd.DataFrame(scenario_records)
        scenario_report.to_csv(destination / "baseline_metrics.csv", index=False)
        for model_name in scenario_report["model"].unique():
            sub = scenario_report[scenario_report["model"] == model_name]
            sub.to_csv(destination / f"results_{model_name}.csv", index=False)

    report = pd.DataFrame(fold_records)
    report.to_csv(destination / "baseline_cv_metrics.csv", index=False)

    metric_cols = [c for c in report.columns if c not in ("model", "fold")]
    summary = (
        report.groupby("model")[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(destination / "baseline_cv_summary.csv", index=False)

    (destination / "baseline_cv_metrics.json").write_text(
        json.dumps(report.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    return report


# ---------------------------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------------------------

def _mlp_evaluate_by_scenario(
    graphs: list,
    model: Any,
    split: str,
) -> list[dict[str, Any]]:
    """Per-scenario metric dicts for the trained MLP model."""
    import torch

    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for graph in graphs:
            mask = getattr(graph, f"{split}_mask").cpu().numpy().astype(bool)
            if not mask.any():
                continue
            x = graph.x.cpu().numpy()[mask]
            x_t = torch.tensor(x, dtype=torch.float32)
            logits = model(x_t)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels = graph.y.cpu().numpy()[mask]
            metrics = classification_metrics(labels, probs)
            metrics.update({
                "model": "mlp",
                "scenario": getattr(graph, "scenario", "unknown"),
                "split": split,
            })
            records.append(metrics)
    return records


def train_mlp(
    graph_dir: str | Path | None = None,
    output_dir: str | Path = "results/mlp",
    hidden: list[int] | None = None,
    dropout: float = 0.3,
    epochs: int = 50,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    seed: int = 42,
    early_stop_patience: int = 10,
    lr_scheduler_patience: int = 5,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_min_lr: float = 1e-5,
    dry_run: bool = False,
    scenario_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Train an MLP on node features from serialised CTU-13 graphs.

    The MLP uses the same focal loss, LR scheduler, and early stopping as
    the GNN trainer so that comparisons are head-to-head.  Node features
    (``graph.x``) come from ``FlowFeatureExtractor``.  No graph structure is
    used: this is a pure tabular ablation.

    Parameters
    ----------
    graph_dir:
        Directory of serialised ``.pt`` graph files.  Ignored when
        ``dry_run=True``.
    output_dir:
        Directory where ``mlp_metrics.csv``, ``mlp_metrics.json``, and
        ``mlp_model.pt`` are written.
    hidden:
        Hidden layer widths.  Defaults to ``[128, 64, 32]``.
    dropout:
        Dropout probability applied after all hidden layers except the last.
    epochs:
        Maximum training epochs (overridden to 3 in dry-run mode).
    batch_size:
        Mini-batch size for the DataLoader.
    learning_rate:
        Initial Adam learning rate.
    weight_decay:
        Adam L2 weight decay.
    focal_gamma, focal_alpha:
        Focal loss hyper-parameters.  Defaults match the GNN trainer.
    seed:
        Random seed forwarded to :func:`seed_everything`.
    early_stop_patience:
        Epochs without val-AUC improvement before stopping.
    lr_scheduler_patience:
        Epochs without val-AUC improvement before LR decay.
    lr_scheduler_factor:
        LR multiplicative decay factor.
    lr_scheduler_min_lr:
        Floor for the learning rate scheduler.
    dry_run:
        When ``True``, skip real data and run 3 epochs on a synthetic graph.
    scenario_ids:
        Optional list of scenario IDs to restrict training/evaluation.

    Returns
    -------
    pd.DataFrame — per-scenario test evaluation records with
    columns ``["model", "scenario", "split", ...]``.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("torch is required for train_mlp.") from exc

    from gbd_ctu.models.baselines.mlp_clf import MLPClassifier
    from gbd_ctu.training.losses import FocalLoss

    seed_everything(seed)

    if hidden is None:
        hidden = [128, 64, 32]

    # ---- data ---------------------------------------------------------------
    if dry_run:
        from gbd_ctu.training.trainer_gnn import _make_synthetic_graph
        graphs = [_make_synthetic_graph(n_nodes=80, n_features=22, seed=seed)]
        epochs = 3
        _logger.info("dry-run mode: using synthetic graph for %d epochs", epochs)
    else:
        if graph_dir is None:
            raise ValueError("graph_dir must be provided when dry_run=False.")
        graphs = load_graphs(graph_dir)
        if scenario_ids is not None:
            graphs = [g for g in graphs if getattr(g, "scenario_id", None) in scenario_ids]
            if not graphs:
                raise ValueError(f"No graphs found for scenario_ids={scenario_ids} in {graph_dir}.")

    x_train_np, y_train_np = _stack_split(graphs, "train")
    x_val_np, y_val_np = _stack_split(graphs, "val")

    if x_train_np.size == 0:
        raise ValueError("No training samples found in graphs.")

    in_channels = x_train_np.shape[1]
    model = MLPClassifier(in_channels=in_channels, hidden=hidden, dropout=dropout)
    criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        patience=lr_scheduler_patience,
        factor=lr_scheduler_factor,
        min_lr=lr_scheduler_min_lr,
    )

    x_train_t = torch.tensor(x_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.long)
    x_val_t = torch.tensor(x_val_np, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_np, dtype=torch.long)
    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    # ---- training loop ------------------------------------------------------
    best_val_auc = float("-inf")
    best_state: dict[str, Any] | None = None
    epochs_no_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        train_loss = running_loss / max(len(loader), 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_probs = torch.softmax(val_logits, dim=-1)[:, 1].numpy()
        from gbd_ctu.evaluation.metrics import classification_metrics as _cm
        val_metrics = _cm(y_val_np, val_probs) if len(y_val_np) > 0 else {"auc": float("nan"), "f1": float("nan")}
        val_auc = float(val_metrics.get("auc", float("nan")))
        val_f1 = float(val_metrics.get("f1", float("nan")))

        scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)
        current_lr = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_auc": val_auc,
            "val_f1": val_f1,
            "lr": current_lr,
        }
        history.append(record)
        _logger.info(
            "epoch %d | train_loss=%.4f val_auc=%.4f val_f1=%.4f",
            epoch, train_loss, val_auc, val_f1,
        )

        if not np.isnan(val_auc) and val_auc >= best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() if hasattr(v, "clone") else v
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                _logger.info("Early stopping at epoch %d", epoch)
                break

    # ---- restore best weights & save ----------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(),
         "in_channels": in_channels,
         "hidden": hidden,
         "dropout": dropout,
         "best_val_auc": best_val_auc,
         "seed": seed},
        dest / "mlp_model.pt",
    )

    # ---- per-scenario test evaluation ---------------------------------------
    test_records = _mlp_evaluate_by_scenario(graphs, model, split="test")
    report = pd.DataFrame(test_records)
    report.to_csv(dest / "mlp_metrics.csv", index=False)
    (dest / "mlp_metrics.json").write_text(
        json.dumps(test_records, indent=2), encoding="utf-8"
    )
    (dest / "mlp_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    _logger.info(
        "MLP training done. best_val_auc=%.4f | wrote results to %s",
        best_val_auc, dest,
    )
    return report
