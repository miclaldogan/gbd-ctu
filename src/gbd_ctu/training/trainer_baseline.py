"""GBD-CTU baseline training.

This module trains classical baselines on node-level IP graph features. Inputs
are serialized graphs and baseline hyperparameters; outputs are saved reports and
scenario-wise evaluation records.
"""

from __future__ import annotations

import json
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
        metrics.update({"model": model_name, "scenario": graph.scenario, "split": "test"})
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
