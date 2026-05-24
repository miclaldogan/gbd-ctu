"""GBD-CTU evaluation metrics.

This module computes binary classification metrics used in CTU-13 experiments.
Inputs are true labels and predicted scores; outputs are metric dictionaries and
sorted reporting DataFrames.

Standalone helpers
------------------
``auc_roc``, ``f1_botnet``, ``precision_recall``, ``false_positive_rate``, and
``confusion_matrix_plot`` expose single-value interfaces that compose well with
external scripts and notebook cells.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover - optional
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Standalone metric functions
# ---------------------------------------------------------------------------

def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return ROC-AUC for binary predictions.

    Returns ``float('nan')`` when only one class is present in *y_true*.
    """
    true_arr = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(y_score, dtype=float)
    if np.unique(true_arr).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(true_arr, score_arr))


def f1_botnet(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return F1-score for the botnet class (class 1).

    Parameters
    ----------
    y_true:
        Integer ground-truth labels (0 = benign, 1 = botnet).
    y_pred:
        Integer predicted labels.
    """
    return float(f1_score(np.asarray(y_true, dtype=int),
                          np.asarray(y_pred, dtype=int),
                          pos_label=1, zero_division=0))


def precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Return ``(precision, recall)`` for the botnet class (class 1)."""
    true_arr = np.asarray(y_true, dtype=int)
    pred_arr = np.asarray(y_pred, dtype=int)
    prec = float(precision_score(true_arr, pred_arr, pos_label=1, zero_division=0))
    rec = float(recall_score(true_arr, pred_arr, pos_label=1, zero_division=0))
    return prec, rec


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return false-positive rate: FP / (FP + TN).

    This is the critical security metric — it quantifies how often benign
    traffic is incorrectly flagged as botnet.
    """
    true_arr = np.asarray(y_true, dtype=int)
    pred_arr = np.asarray(y_pred, dtype=int)
    tn, fp, _fn, _tp = confusion_matrix(true_arr, pred_arr, labels=[0, 1]).ravel()
    return float(fp / max(fp + tn, 1))


def confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
) -> "plt.Figure":
    """Return a ``matplotlib.figure.Figure`` showing the confusion matrix.

    Parameters
    ----------
    y_true:
        Integer ground-truth labels.
    y_pred:
        Integer predicted labels.
    title:
        Figure title string.

    Raises
    ------
    ImportError
        If *matplotlib* is not installed.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for confusion_matrix_plot.")
    cm = confusion_matrix(np.asarray(y_true, dtype=int),
                          np.asarray(y_pred, dtype=int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    classes = ["benign", "botnet"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=classes, yticklabels=classes,
        xlabel="Predicted label", ylabel="True label",
        title=title,
    )
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composite helpers (used by the trainer and scenario evaluator)
# ---------------------------------------------------------------------------

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
