"""GBD-CTU evaluation metrics.

This module computes binary classification metrics used in CTU-13 experiments.
Inputs are true labels and predicted scores; outputs are metric dictionaries and
sorted reporting DataFrames.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true: np.ndarray | list[int], y_score: np.ndarray | list[float], threshold: float = 0.5) -> dict[str, Any]:
    """Compute AUC, F1, precision, recall, FPR, and supporting metrics."""

    true_array = np.asarray(y_true, dtype=int)
    score_array = np.asarray(y_score, dtype=float)
    pred_array = (score_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true_array, pred_array, labels=[0, 1]).ravel()
    metrics = {
        "auc": float("nan"),
        "f1": f1_score(true_array, pred_array, zero_division=0),
        "precision": precision_score(true_array, pred_array, zero_division=0),
        "recall": recall_score(true_array, pred_array, zero_division=0),
        "fpr": float(fp / max(fp + tn, 1)),
        "support": int(true_array.shape[0]),
        "botnet_rate": float(np.mean(true_array)) if true_array.size else 0.0,
    }
    if np.unique(true_array).shape[0] > 1:
        metrics["auc"] = roc_auc_score(true_array, score_array)
    return metrics


def metrics_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert metric dictionaries into a stable reporting DataFrame."""

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    ordered_columns = [
        "model",
        "scenario",
        "split",
        "auc",
        "f1",
        "precision",
        "recall",
        "fpr",
        "support",
        "botnet_rate",
    ]
    available_columns = [column for column in ordered_columns if column in frame.columns]
    return frame[available_columns].sort_values(["model", "scenario", "split"]).reset_index(drop=True)
