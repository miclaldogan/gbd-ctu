"""GBD-CTU evaluation plots.

This module generates publication-quality figures for the CTU-13 evaluation:

- :func:`confusion_matrix_plot` — normalised confusion matrix PNG per scenario
- :func:`roc_pr_curves` — multi-scenario ROC and PR curves per model family
- :func:`scenario_comparison_heatmap` — heatmap of a metric across scenarios × models

All plots use 300 DPI, ``tight_layout()``, and the shared :data:`PALETTE` colour
map (one colour per model family).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _HAS_MPL = True
except ImportError:  # pragma: no cover - optional
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Shared palette
# ---------------------------------------------------------------------------

#: One colour per model family — used consistently across all plots.
PALETTE: dict[str, str] = {
    "GraphSAGE": "#1f77b4",
    "GAT": "#ff7f0e",
    "Hybrid": "#2ca02c",
    "XGBoost": "#d62728",
    "RandomForest": "#9467bd",
}

_DPI = 300


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for evaluation plots.")


# ---------------------------------------------------------------------------
# Confusion-matrix plot
# ---------------------------------------------------------------------------

def confusion_matrix_plot(
    y_true: "np.ndarray | list",
    y_pred: "np.ndarray | list",
    scenario_id: int | str,
    output_dir: "Path | str",
) -> Path:
    """Save a normalised confusion matrix PNG for a single scenario.

    Parameters
    ----------
    y_true:
        Integer ground-truth labels (0 = benign, 1 = botnet).
    y_pred:
        Integer predicted labels.
    scenario_id:
        Scenario identifier used in the filename and figure title.
    output_dir:
        Directory where the PNG will be written.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.
    """
    _require_mpl()
    from sklearn.metrics import confusion_matrix as _cm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_arr = np.asarray(y_true, dtype=int)
    pred_arr = np.asarray(y_pred, dtype=int)
    cm = _cm(true_arr, pred_arr, labels=[0, 1])
    # Normalise row-wise (true-label denominator)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["benign", "botnet"]
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=classes, yticklabels=classes,
        xlabel="Predicted label", ylabel="True label",
        title=f"Scenario {scenario_id} — Confusion Matrix",
    )
    thresh = 0.5
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n(n={cm[i, j]})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9,
            )
    fig.tight_layout()
    out_path = output_dir / f"confusion_matrix_scenario_{scenario_id}.png"
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# ROC + PR curves
# ---------------------------------------------------------------------------

def roc_pr_curves(
    records: list[dict[str, Any]],
    output_dir: "Path | str",
) -> Path:
    """Save one figure with ROC curves and one with PR curves, all scenarios overlaid.

    Each record must contain at least:
    ``model`` (str), ``y_true`` (array-like), ``y_score`` (array-like),
    and optionally ``scenario_id`` (int/str).

    Parameters
    ----------
    records:
        List of dicts, one per (model, scenario) evaluation run.
    output_dir:
        Directory where PNGs will be saved.

    Returns
    -------
    Path
        Directory containing the saved ``roc_curves.png`` and ``pr_curves.png``.
    """
    _require_mpl()
    from sklearn.metrics import roc_curve, precision_recall_curve, auc as _auc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- ROC ---
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")

    # --- PR ---
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))

    seen_labels: set[str] = set()
    for rec in records:
        model_name = str(rec.get("model", "unknown"))
        y_true = np.asarray(rec["y_true"], dtype=int)
        y_score = np.asarray(rec["y_score"], dtype=float)
        scenario = rec.get("scenario_id", "?")
        colour = PALETTE.get(model_name, "#333333")
        label = model_name if model_name not in seen_labels else "_nolegend_"
        seen_labels.add(model_name)

        if np.unique(y_true).shape[0] < 2:
            continue

        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
        roc_auc = _auc(fpr_arr, tpr_arr)
        ax_roc.plot(
            fpr_arr, tpr_arr,
            color=colour, alpha=0.6, lw=1.0,
            label=f"{label} (AUC={roc_auc:.2f})" if label != "_nolegend_" else "_nolegend_",
        )

        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_score)
        pr_auc = _auc(rec_arr, prec_arr)
        ax_pr.plot(
            rec_arr, prec_arr,
            color=colour, alpha=0.6, lw=1.0,
            label=f"{label} (AUPRC={pr_auc:.2f})" if label != "_nolegend_" else "_nolegend_",
        )

    ax_roc.set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
    ax_roc.legend(loc="lower right", fontsize=7)
    fig_roc.tight_layout()
    roc_path = output_dir / "roc_curves.png"
    fig_roc.savefig(roc_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig_roc)

    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax_pr.legend(loc="upper right", fontsize=7)
    fig_pr.tight_layout()
    pr_path = output_dir / "pr_curves.png"
    fig_pr.savefig(pr_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig_pr)

    return output_dir


# ---------------------------------------------------------------------------
# Scenario × model heatmap
# ---------------------------------------------------------------------------

def scenario_comparison_heatmap(
    metrics_df: "pd.DataFrame",
    metric: str,
    output_dir: "Path | str",
) -> Path:
    """Save a heatmap: rows = scenarios, columns = model families, colour = metric.

    Parameters
    ----------
    metrics_df:
        Long-format DataFrame with at least columns
        ``scenario_id``, ``model``, and *metric*.
    metric:
        Column name of the metric to visualise (e.g. ``"auc"``, ``"auprc"``).
    output_dir:
        Directory where the PNG will be saved.

    Returns
    -------
    Path
        Absolute path to the saved PNG file.
    """
    _require_mpl()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metric not in metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found in metrics_df columns: {list(metrics_df.columns)}")

    pivot = metrics_df.pivot_table(
        index="scenario_id", columns="model", values=metric, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(max(4, len(pivot.columns) * 1.2), max(3, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
    fig.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"S{s}" for s in pivot.index], fontsize=8)
    ax.set(title=f"{metric.upper()} by Scenario and Model", xlabel="Model", ylabel="Scenario")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="black")

    fig.tight_layout()
    out_path = output_dir / f"heatmap_{metric}.png"
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path
