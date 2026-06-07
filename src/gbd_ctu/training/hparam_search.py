"""GBD-CTU hyperparameter search with Optuna.

Public API
----------
``SEARCH_SPACE``
    Dict describing the hyperparameter search space.

``sample_params``
    Draw one hyperparameter configuration from an Optuna trial.

``objective``
    Optuna objective function: train the model, return best val-AUC.
    Supports per-epoch intermediate reporting for MedianPruner.

``run_hparam_search``
    Create (or resume) an Optuna study and run *n_trials* trials.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore[assignment]
    _HAS_OPTUNA = False

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

SEARCH_SPACE: dict[str, Any] = {
    "hidden_channels": [32, 64, 128, 256],
    "dropout": [0.1, 0.2, 0.3, 0.5],
    "learning_rate": ("log_uniform", 1e-4, 1e-2),
    "weight_decay": ("log_uniform", 1e-5, 1e-3),
    "heads": [1, 2, 4, 8],
    "focal_gamma": [0.5, 1.0, 2.0, 4.0],
    "focal_alpha": [0.5, 0.65, 0.75, 0.9],
    "num_layers": [2, 3, 4],
}


def sample_params(trial: Any, model_type: str = "hybrid") -> dict[str, Any]:
    """Draw one hyperparameter configuration from an Optuna trial.

    Parameters
    ----------
    trial:
        An ``optuna.Trial`` object.
    model_type:
        GNN family; controls which params are sampled
        (``heads`` only for GAT/Hybrid, ``num_layers`` only for SAGE/Hybrid).

    Returns
    -------
    dict of hyperparameter name → sampled value.
    """
    params: dict[str, Any] = {
        "hidden_channels": trial.suggest_categorical(
            "hidden_channels", [32, 64, 128, 256]
        ),
        "dropout": trial.suggest_categorical(
            "dropout", [0.1, 0.2, 0.3, 0.5]
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-4, 1e-2, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-5, 1e-3, log=True
        ),
        "focal_gamma": trial.suggest_categorical(
            "focal_gamma", [0.5, 1.0, 2.0, 4.0]
        ),
        "focal_alpha": trial.suggest_categorical(
            "focal_alpha", [0.5, 0.65, 0.75, 0.9]
        ),
    }
    if model_type in ("gat", "hybrid"):
        params["heads"] = trial.suggest_categorical("heads", [1, 2, 4, 8])
    if model_type in ("graphsage", "hybrid"):
        params["num_layers"] = trial.suggest_categorical("num_layers", [2, 3, 4])
    return params


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(
    trial: Any,
    graph_dir: Path | str | None = None,
    model_type: str = "hybrid",
    seed: int = 42,
    epochs: int = 20,
    dry_run: bool = False,
) -> float:
    """Optuna objective: train model, return best val-AUC.

    Parameters
    ----------
    trial:
        An ``optuna.Trial``.
    graph_dir:
        Directory of serialised ``.pt`` graph files.  Ignored when
        ``dry_run=True``.
    model_type:
        GNN architecture (``"hybrid"``, ``"gat"``, ``"graphsage"``).
    seed:
        Base random seed.  Each trial uses ``seed + trial.number`` for
        full reproducibility.
    epochs:
        Training epochs per trial.  Set to 3 in dry-run mode.
    dry_run:
        When ``True`` bypass real data; run on a tiny synthetic graph.
        Useful for CI and unit tests.

    Returns
    -------
    float — best validation AUC across epochs (higher = better).
    Per-epoch values are reported to Optuna so MedianPruner can fire.

    Raises
    ------
    optuna.TrialPruned
        Raised internally when the pruner decides to stop the trial early.
    """
    from gbd_ctu.training.seed import seed_everything

    trial_seed = seed + trial.number
    seed_everything(trial_seed)

    params = sample_params(trial, model_type)
    _logger.debug("Trial %d params: %s", trial.number, params)

    if dry_run:
        return _objective_dry_run(trial, model_type, params, trial_seed)
    return _objective_full(trial, graph_dir, model_type, params, trial_seed, epochs)


def _objective_dry_run(
    trial: Any, model_type: str, params: dict, seed: int
) -> float:
    """Run 3 epochs on a tiny synthetic graph — no real data required."""
    import tempfile

    from gbd_ctu.training.trainer_gnn import train_gnn

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt = Path(f.name)

    try:
        result = train_gnn(
            family=model_type,
            checkpoint_path=ckpt,
            dry_run=True,
            seed=seed,
            hidden_channels=params["hidden_channels"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            focal_gamma=params["focal_gamma"],
            focal_alpha=params["focal_alpha"],
            heads=params.get("heads", 4),
            num_layers=params.get("num_layers", 3),
        )
    finally:
        ckpt.unlink(missing_ok=True)
        ckpt.with_suffix(".history.json").unlink(missing_ok=True)

    history = result.get("history", [])
    for epoch_rec in history:
        val_auc = float(epoch_rec.get("val_auc", 0.0))
        epoch = int(epoch_rec.get("epoch", 0))
        trial.report(val_auc, step=epoch)
        if optuna is not None and trial.should_prune():
            raise optuna.TrialPruned()

    return float(result.get("best_val_auc", 0.0))


def _objective_full(
    trial: Any,
    graph_dir: Path | str | None,
    model_type: str,
    params: dict,
    seed: int,
    epochs: int,
) -> float:
    """Full training loop with per-epoch pruner reporting."""
    if graph_dir is None:
        raise ValueError("graph_dir must be provided when dry_run=False.")

    try:
        import torch
        from gbd_ctu.data.graph_builder import load_graphs
        from gbd_ctu.training.losses import FocalLoss
        from gbd_ctu.training.trainer_gnn import (
            _build_model,
            _evaluate_split,
            _train_one_epoch,
        )
    except ImportError as exc:
        raise ImportError(
            "torch and torch-geometric are required for hparam search."
        ) from exc

    graphs = load_graphs(graph_dir)
    if not graphs:
        raise ValueError(f"No graphs found in {graph_dir}.")

    in_channels = int(graphs[0].num_node_features)
    model = _build_model(
        family=model_type,
        in_channels=in_channels,
        hidden_channels=params["hidden_channels"],
        out_channels=2,
        heads=params.get("heads", 4),
        num_layers=params.get("num_layers", 3),
        dropout=params["dropout"],
    )

    criterion = FocalLoss(
        gamma=float(params["focal_gamma"]),
        alpha=float(params["focal_alpha"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    best_auc = 0.0
    for epoch in range(epochs):
        _train_one_epoch(
            model, graphs, criterion, optimizer,
            batch_size=512, num_neighbors=[10, 5],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_records = _evaluate_split(model, graphs, split="val")
        val_auc = (
            float(np.nanmean([r["auc"] for r in val_records]))
            if val_records else 0.0
        )
        best_auc = max(best_auc, val_auc)

        trial.report(val_auc, step=epoch)
        if optuna is not None and trial.should_prune():
            raise optuna.TrialPruned()

    return best_auc


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_hparam_search(
    graph_dir: Path | str | None = None,
    model_type: str = "hybrid",
    n_trials: int = 50,
    seed: int = 42,
    storage: str | None = None,
    study_name: str = "gbd_ctu_hparam",
    output_dir: Path | str = "results/hparam",
    wandb_project: str | None = None,
    dry_run: bool = False,
    epochs: int = 20,
) -> Any:
    """Create (or resume) an Optuna study and run *n_trials* trials.

    Parameters
    ----------
    graph_dir:
        Directory of serialised ``.pt`` graph files.
    model_type:
        GNN architecture to tune.
    n_trials:
        Number of Optuna trials to run.
    seed:
        Base random seed passed to each trial.
    storage:
        Optuna storage URL (e.g. ``"sqlite:///results/hparam/study.db"``).
        When ``None`` an in-memory storage is used (not resume-able).
    study_name:
        Name used to identify the study in the storage backend.
    output_dir:
        Directory where ``best_params.json`` is written.
    wandb_project:
        When set, log each trial's params and result to Weights & Biases.
    dry_run:
        Pass ``True`` to run each trial on a tiny synthetic graph (for CI).
    epochs:
        Training epochs per trial.

    Returns
    -------
    ``optuna.Study`` — the completed study object.

    Raises
    ------
    ImportError
        When ``optuna`` is not installed.
    """
    if not _HAS_OPTUNA:
        raise ImportError(
            "optuna >= 3.0 is required. Install with: pip install optuna>=3.0"
        )

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,  # enables --resume
    )

    _logger.info(
        "Starting Optuna study '%s': model=%s, n_trials=%d, storage=%s",
        study_name, model_type, n_trials, storage or "in-memory",
    )

    def _wrapped_objective(trial: Any) -> float:
        value = objective(
            trial,
            graph_dir=graph_dir,
            model_type=model_type,
            seed=seed,
            epochs=epochs,
            dry_run=dry_run,
        )
        if wandb_project is not None:
            _log_to_wandb(trial, value, wandb_project)
        return value

    study.optimize(
        _wrapped_objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    # Write best params
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    best = {
        "params": study.best_params,
        "value": study.best_value,
        "trial_number": study.best_trial.number,
        "model_type": model_type,
        "n_trials": n_trials,
        "seed": seed,
    }
    params_path = dest / "best_params.json"
    params_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    _logger.info("Best params written to %s (val_auc=%.4f)", params_path, study.best_value)

    return study


def _log_to_wandb(trial: Any, value: float, project: str) -> None:
    try:
        import wandb
        import os
        if os.environ.get("WANDB_MODE") == "disabled":
            return
        wandb.log({
            "trial": trial.number,
            "val_auc": value,
            **{f"param/{k}": v for k, v in trial.params.items()},
        })
    except Exception:
        pass
