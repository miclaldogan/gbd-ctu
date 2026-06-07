"""GBD-CTU model comparison report.

This module merges per-scenario result CSVs from multiple models (GNN variants
and baselines) into a wide-format comparison table, generates bar-chart and ROC
visualisations, and reports statistical significance via a Wilcoxon signed-rank
test of the hybrid GNN against the best classical baseline.

Expected input CSV schema (as produced by ``evaluate_all_scenarios`` or
``train_baselines``)::

    model, scenario_id, scenario, auc, f1, precision, recall, fpr, ...

Public API
----------
``build_comparison_table``
    Wide pivot table: one row per scenario, one column per (model × metric).

``generate_figures``
    Bar chart + ROC placeholders → PNG + PDF in ``figures/`` sub-directory.

``compare_reports``
    Legacy convenience wrapper: merges two CSVs and writes combined output.

``run_comparison``
    Main entry-point called by ``scripts/compare_models.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False

try:
    from scipy.stats import wilcoxon
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GNN_MODELS = {"graphsage", "gat", "hybrid"}
_BASELINE_MODELS = {"xgboost", "random_forest", "randomforest", "rf"}
_METRICS = ["auc", "auprc", "mcc", "f1", "fpr", "fpr_at_tpr95"]

# Column display names used in the wide table
_MODEL_DISPLAY = {
    "graphsage": "GraphSAGE",
    "gat": "GAT",
    "hybrid": "Hybrid",
    "xgboost": "XGBoost",
    "random_forest": "RandomForest",
    "randomforest": "RandomForest",
    "rf": "RandomForest",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results_dir(results_dir: Path) -> pd.DataFrame:
    """Load all ``results_*.csv`` files found in *results_dir* and concatenate."""
    csvs = list(results_dir.glob("results_*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No 'results_*.csv' files found in {results_dir}. "
            "Run evaluate.py and train_baselines.py first."
        )
    frames = [pd.read_csv(p) for p in sorted(csvs)]
    return pd.concat(frames, ignore_index=True)


def _normalise_model_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


# ---------------------------------------------------------------------------
# Wide pivot table
# ---------------------------------------------------------------------------

def build_comparison_table(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return one wide ``pd.DataFrame`` per metric.

    Each DataFrame has columns::

        scenario_id | scenario | <ModelA> | <ModelB> | ... | Best Model

    Parameters
    ----------
    df:
        Long-format DataFrame with at minimum columns
        ``model``, ``scenario_id``, ``scenario``, ``auc``, ``f1``, ``fpr``.
    """
    df = df.copy()
    df["model"] = df["model"].apply(_normalise_model_name)
    df["model_display"] = df["model"].map(lambda m: _MODEL_DISPLAY.get(m, m.title()))

    tables: dict[str, pd.DataFrame] = {}
    for metric in _METRICS:
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(
            index=["scenario_id", "scenario"],
            columns="model_display",
            values=metric,
            aggfunc="mean",
        ).reset_index()
        pivot.columns.name = None

        # Determine best model per row (for AUC/F1 highest = best; for FPR lowest = best)
        model_cols = [c for c in pivot.columns if c not in ("scenario_id", "scenario")]
        if model_cols:
            if metric in ("fpr", "fpr_at_tpr95"):
                pivot["Best Model"] = pivot[model_cols].idxmin(axis=1)
            else:
                pivot["Best Model"] = pivot[model_cols].idxmax(axis=1)

        tables[metric] = pivot.sort_values("scenario_id").reset_index(drop=True)

    return tables


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def wilcoxon_hybrid_vs_baseline(df: pd.DataFrame) -> dict[str, Any]:
    """Wilcoxon signed-rank test: Hybrid GNN AUC vs best classical baseline AUC.

    Returns a dict with keys ``statistic``, ``p_value``, ``significant``,
    ``n_scenarios``, and ``note``.

    A ``p_value < 0.05`` indicates the difference is statistically significant.
    """
    df = df.copy()
    df["model"] = df["model"].apply(_normalise_model_name)

    hybrid_rows = df[df["model"] == "hybrid"][["scenario_id", "auc"]].set_index("scenario_id")
    baseline_rows = df[df["model"].isin(_BASELINE_MODELS)].copy()

    if hybrid_rows.empty or baseline_rows.empty:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant": None,
            "n_scenarios": 0,
            "note": "Insufficient data for Wilcoxon test.",
        }

    # Best baseline AUC per scenario
    best_baseline = (
        baseline_rows.groupby("scenario_id")["auc"].max().rename("baseline_auc")
    )
    combined = hybrid_rows.join(best_baseline, how="inner")
    combined = combined.dropna()

    if len(combined) < 2:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant": None,
            "n_scenarios": len(combined),
            "note": "Not enough paired scenarios for Wilcoxon test (need ≥ 2).",
        }

    if not _HAS_SCIPY:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "significant": None,
            "n_scenarios": len(combined),
            "note": "scipy not installed; Wilcoxon test skipped.",
        }

    stat, p_val = wilcoxon(combined["auc"].values, combined["baseline_auc"].values,
                           alternative="two-sided", zero_method="wilcox")
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
        "n_scenarios": len(combined),
        "note": "Wilcoxon signed-rank test (two-sided): Hybrid GNN vs best baseline.",
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figures(
    df: pd.DataFrame,
    figures_dir: Path,
    dpi: int = 300,
) -> list[Path]:
    """Generate and save bar chart and ROC figure.

    Produces two figures each saved as PNG and PDF:

    1. ``auc_by_scenario.png/pdf`` — grouped bar chart: AUC per scenario ×
       model.
    2. ``roc_placeholder.png/pdf`` — ROC curve panel (placeholder when raw
       score arrays are unavailable; shows a note instead).

    Parameters
    ----------
    df:
        Long-format results DataFrame.
    figures_dir:
        Destination directory (created if missing).
    dpi:
        Resolution for PNG output (default 300).

    Returns
    -------
    List of saved file paths.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for generate_figures.")

    figures_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    df = df.copy()
    df["model"] = df["model"].apply(_normalise_model_name)
    df["model_display"] = df["model"].map(lambda m: _MODEL_DISPLAY.get(m, m.title()))

    # ---- 1. AUC bar chart ---------------------------------------------------
    if "auc" in df.columns and not df.empty:
        fig, ax = plt.subplots(figsize=(max(8, len(df["scenario_id"].unique()) * 0.8), 5))

        pivot = df.pivot_table(
            index="scenario_id", columns="model_display", values="auc", aggfunc="mean"
        )
        pivot.plot(kind="bar", ax=ax, width=0.75)

        ax.set_title("AUC per Scenario by Model", fontsize=13)
        ax.set_xlabel("Scenario ID")
        ax.set_ylabel("AUC")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        for ext in ("png", "pdf"):
            path = figures_dir / f"auc_by_scenario.{ext}"
            fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)

    # ---- 2. F1 bar chart ----------------------------------------------------
    if "f1" in df.columns and not df.empty:
        fig, ax = plt.subplots(figsize=(max(8, len(df["scenario_id"].unique()) * 0.8), 5))

        pivot_f1 = df.pivot_table(
            index="scenario_id", columns="model_display", values="f1", aggfunc="mean"
        )
        pivot_f1.plot(kind="bar", ax=ax, width=0.75)

        ax.set_title("F1 per Scenario by Model", fontsize=13)
        ax.set_xlabel("Scenario ID")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        for ext in ("png", "pdf"):
            path = figures_dir / f"f1_by_scenario.{ext}"
            fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)

    # ---- 3. ROC placeholder -------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — run evaluate.py to populate")
    ax.legend(loc="lower right")
    ax.text(0.5, 0.5, "ROC curves require\nper-node score arrays.\nRun evaluate.py first.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="gray",
            bbox=dict(boxstyle="round", fc="white", ec="gray"))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = figures_dir / f"roc_curves.{ext}"
        fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)

    return saved


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def run_comparison(
    results_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Load all result CSVs, build comparison tables, generate figures.

    Parameters
    ----------
    results_dir:
        Directory containing ``results_*.csv`` files.
    output_dir:
        Root output directory; tables go to ``output_dir/``, figures to
        ``output_dir/figures/``.

    Returns
    -------
    Dict with keys ``tables``, ``wilcoxon``, ``figures``, ``csv_paths``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_results_dir(results_dir)

    # Wide comparison tables
    tables = build_comparison_table(df)
    csv_paths: dict[str, str] = {}
    md_lines: list[str] = []
    for metric, table in tables.items():
        csv_path = output_dir / f"comparison_{metric}.csv"
        table.to_csv(csv_path, index=False)
        csv_paths[metric] = str(csv_path)
        md_lines.append(f"\n### {metric.upper()} Comparison\n")
        try:
            md_lines.append(table.to_markdown(index=False))
        except ImportError:
            md_lines.append(table.to_string(index=False))

    # Markdown table file
    md_path = output_dir / "comparison_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Combined long-format CSV
    combined_path = output_dir / "comparison_all.csv"
    df.to_csv(combined_path, index=False)
    combined_path.with_suffix(".json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2), encoding="utf-8"
    )

    # Statistical significance
    wx_result = wilcoxon_hybrid_vs_baseline(df)

    # Figures
    figures_dir = output_dir / "figures"
    fig_paths: list[str] = []
    if _HAS_MPL:
        try:
            saved = generate_figures(df, figures_dir)
            fig_paths = [str(p) for p in saved]
        except Exception as exc:  # pragma: no cover
            fig_paths = [f"error: {exc}"]

    return {
        "tables": {k: v.to_dict(orient="records") for k, v in tables.items()},
        "wilcoxon": wx_result,
        "figures": fig_paths,
        "csv_paths": csv_paths,
        "combined_csv": str(combined_path),
        "markdown_report": str(md_path),
    }


# ---------------------------------------------------------------------------
# Legacy convenience wrapper (kept for backward-compat with __init__.py)
# ---------------------------------------------------------------------------

def compare_reports(
    gnn_report_path: str | Path,
    baseline_report_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Merge GNN and baseline report tables into a sorted comparison frame.

    This is a thin convenience wrapper retained for backward compatibility.
    For the full comparison pipeline use ``run_comparison``.
    """
    gnn_report = pd.read_csv(gnn_report_path)
    baseline_report = pd.read_csv(baseline_report_path)
    combined = pd.concat([gnn_report, baseline_report], ignore_index=True)
    combined = combined.sort_values(["scenario", "model"]).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    destination.with_suffix(".json").write_text(
        json.dumps(combined.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    return combined

