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

    records: list[dict[str, Any]] = []
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    
    # Stratified K-Fold ayarı (5 parçaya bölüyoruz)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    for model_name, estimator in models:
        fold_metrics = [] # Her fold için sonuçları tutacağımız geçici liste
        
        # Eğitim verisini 5'e bölüp her biri için eğitiyoruz
        for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
            x_fold_train, y_fold_train = x_train[train_idx], y_train[train_idx]
            x_fold_val, y_fold_val = x_train[val_idx], y_train[val_idx]
            
            # Modeli eğit
            estimator.fit(x_fold_train, y_fold_train)
            
            # Doğrulama metriklerini hesapla (val_idx üzerinden)
            probabilities = estimator.predict_proba(x_fold_val)[:, 1]
            metrics = classification_metrics(y_fold_val, probabilities)
            metrics.update({"model": model_name, "fold": fold})
            fold_metrics.append(metrics)
            
        # Modelin eğitilmiş son halini kaydet (Serialization)
        model_path = destination / f"{model_name}.joblib"
        joblib.dump(estimator, model_path)
        
        # Tüm fold'ların ortalama ve standart sapmasını hesaplayıp ana kayda ekle
        df_folds = pd.DataFrame(fold_metrics)
        mean_metrics = df_folds.mean().to_dict()
        std_metrics = df_folds.std().to_dict()
        
        final_record = {"model": model_name}
        for k in mean_metrics:
            if k not in ["model", "fold"]:
                final_record[f"{k}_mean"] = mean_metrics[k]
                final_record[f"{k}_std"] = std_metrics[k]
        
        records.append(final_record)
        
    # Raporlama kısmı
    report = pd.DataFrame(records)
    report_path = destination / "baseline_cv_metrics.csv"
    report.to_csv(report_path, index=False)
    
    (destination / "baseline_cv_metrics.json").write_text(
        json.dumps(report.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    
    return report
