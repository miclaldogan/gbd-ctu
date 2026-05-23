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

from gbd_ctu.data.graph_builder import load_graphs
from gbd_ctu.evaluation.metrics import classification_metrics, metrics_frame
from gbd_ctu.models.baselines.random_forest_clf import RandomForestBaseline
from gbd_ctu.models.baselines.xgboost_clf import XGBoostBaseline


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
) -> pd.DataFrame:
    """Train Random Forest and XGBoost baselines on serialized CTU-13 graphs."""

    graphs = load_graphs(graph_dir)
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

    records: list[dict[str, Any]] = []
    for model_name, estimator in models:
        estimator.fit(x_train, y_train)
        records.extend(_evaluate_by_scenario(graphs, estimator, model_name=model_name))
    report = metrics_frame(records)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "baseline_metrics.csv"
    report.to_csv(report_path, index=False)
    (destination / "baseline_metrics.json").write_text(
        json.dumps(report.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return report
