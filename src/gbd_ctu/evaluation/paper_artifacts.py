"""GBD-CTU paper artifact export.

Generates publication-ready LaTeX tables (with booktabs and bold-best) and
300-DPI figures from evaluation result CSVs.  All outputs are reproducible:
given the same input CSVs the outputs are bit-for-bit identical.

Public API
----------
``apply_paper_style``
    Set matplotlib rcParams for publication-quality figures.

``bold_best``
    Wrap the best value in each metric column with ``\\textbf{}``.

``export_results_table``
    LaTeX ``tabular`` for main results (AUC/F1/MCC/AUPRC/FPR), bold best.

``export_ablation_table``
    LaTeX ``tabular`` for ablation results.

``export_significance_table``
    LaTeX ``tabular`` for Wilcoxon + Bonferroni significance, with stars.

``export_dataset_table``
    LaTeX ``tabular`` for per-scenario dataset statistics.

``draw_model_architecture``
    Write a Graphviz ``.dot`` file describing the model layer graph.

``scenario_heatmap_figure``
    Heatmap: rows = scenarios, columns = models, colour = metric value.

``ablation_bar_figure``
    Grouped bar chart comparing ablation variants across metrics.

``per_scenario_roc_pr_figure``
    Grid of per-scenario ROC / PR subplots, one curve per model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

#: matplotlib rcParams applied by :func:`apply_paper_style`.
STYLE_RC: dict[str, Any] = {
    "font.size": 11,
    "font.family": "serif",
    "text.usetex": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
}

#: Colour palette — one colour per model family (lower-case keys).
PALETTE: dict[str, str] = {
    "hybrid": "#1f77b4",
    "graphsage": "#ff7f0e",
    "gat": "#2ca02c",
    "xgboost": "#d62728",
    "random_forest": "#9467bd",
    "mlp": "#8c564b",
}

#: Metrics where lower is better.
LOWER_BETTER: frozenset[str] = frozenset({"fpr", "fpr_at_tpr95"})

_DPI = 300


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for paper figures.")


def apply_paper_style() -> None:
    """Apply :data:`STYLE_RC` to ``matplotlib.rcParams``."""
    _require_mpl()
    plt.rcParams.update(STYLE_RC)


# ---------------------------------------------------------------------------
# bold_best
# ---------------------------------------------------------------------------

def bold_best(
    df: pd.DataFrame,
    metric_cols: list[str],
    lower_better: list[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of *df* with the best value per column wrapped in ``\\textbf{}``.

    For each column in *metric_cols*:
    - If the column name is in *lower_better* (e.g. FPR), the **minimum** row
      value is bolded.
    - Otherwise the **maximum** row value is bolded.

    Non-numeric values and NaNs are left unchanged.  The output cells are
    formatted as strings suitable for direct LaTeX insertion.

    Parameters
    ----------
    df:
        Input DataFrame.  Numeric values in *metric_cols* will be formatted
        as ``"%.4f"`` strings after bolding.
    metric_cols:
        Column names to consider.  Columns not present in *df* are skipped.
    lower_better:
        Columns where smaller is better.  Defaults to ``list(LOWER_BETTER)``.

    Returns
    -------
    pd.DataFrame — same shape as *df*, with *metric_cols* as string columns.
    """
    if lower_better is None:
        lower_better = list(LOWER_BETTER)
    lower_set = set(lower_better)

    out = df.copy()
    for col in metric_cols:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        if series.isna().all():
            out[col] = out[col].astype(str)
            continue

        if col in lower_set:
            best_idx = series.idxmin()
        else:
            best_idx = series.idxmax()

        formatted: list[str] = []
        for idx, val in series.items():
            if np.isnan(val):
                formatted.append("—")
            elif idx == best_idx:
                formatted.append(r"\textbf{" + f"{val:.4f}" + r"}")
            else:
                formatted.append(f"{val:.4f}")
        out[col] = formatted

    return out


# ---------------------------------------------------------------------------
# LaTeX table helpers
# ---------------------------------------------------------------------------

_BOOKTABS_HEADER = r"\usepackage{booktabs}"

def _to_latex_booktabs(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "",
    float_format: str = "%.4f",
    na_rep: str = "—",
    column_format: str | None = None,
) -> str:
    """Render *df* as a booktabs LaTeX tabular string.

    Replaces the default LaTeX ``\\hline`` rules with ``\\toprule``,
    ``\\midrule``, and ``\\bottomrule``.
    """
    n_cols = len(df.columns)
    col_fmt = column_format or ("l" + "r" * (n_cols - 1))

    try:
        raw = df.to_latex(
            index=False,
            escape=False,
            float_format=float_format,
            na_rep=na_rep,
            column_format=col_fmt,
            caption=caption if caption else None,
            label=label if label else None,
        )
    except TypeError:
        raw = df.to_latex(
            index=False,
            escape=False,
            na_rep=na_rep,
            column_format=col_fmt,
        )

    raw = raw.replace(r"\hline", "")
    lines = raw.splitlines()
    out_lines: list[str] = []
    rule_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped == r"\toprule":
            out_lines.append(r"\toprule")
            rule_count += 1
        elif stripped == r"\midrule":
            out_lines.append(r"\midrule")
            rule_count += 1
        elif stripped == r"\bottomrule":
            out_lines.append(r"\bottomrule")
        else:
            out_lines.append(line)

    if r"\toprule" not in "\n".join(out_lines):
        fixed: list[str] = []
        inserted = {"top": False, "mid": False, "bot": False}
        for i, line in enumerate(out_lines):
            if r"\begin{tabular}" in line and not inserted["top"]:
                fixed.append(line)
                fixed.append(r"\toprule")
                inserted["top"] = True
            elif line.strip().startswith(df.columns[0]) and not inserted["mid"]:
                fixed.append(line)
                fixed.append(r"\midrule")
                inserted["mid"] = True
            elif r"\end{tabular}" in line and not inserted["bot"]:
                fixed.append(r"\bottomrule")
                fixed.append(line)
                inserted["bot"] = True
            else:
                fixed.append(line)
        out_lines = fixed

    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Table exports
# ---------------------------------------------------------------------------

def export_results_table(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    metrics: list[str] | None = None,
) -> str:
    """Export the main results table as a booktabs LaTeX string.

    The table has one row per (model, scenario) pair with metrics as columns.
    The best value in each metric column is wrapped in ``\\textbf{}``.

    Parameters
    ----------
    metrics_df:
        Long-format DataFrame with columns including ``model``, ``scenario``,
        and metric columns (``auc``, ``f1``, ``mcc``, ``auprc``, ``fpr``).
    output_path:
        Destination ``.tex`` file.
    metrics:
        Metric columns to include.  Defaults to available metrics from
        ``["auc", "auprc", "mcc", "f1", "fpr"]``.

    Returns
    -------
    str — the LaTeX table string.
    """
    if metrics is None:
        metrics = [m for m in ["auc", "auprc", "mcc", "f1", "fpr"] if m in metrics_df.columns]

    id_cols = [c for c in ["model", "scenario", "scenario_id"] if c in metrics_df.columns]
    keep_cols = id_cols + [m for m in metrics if m in metrics_df.columns]
    table = metrics_df[keep_cols].copy()

    # Pivot so each row = scenario, each model is its own set of cols, or keep long
    table = bold_best(table, metrics)

    display_cols = {
        "model": "Model",
        "scenario": "Scenario",
        "scenario_id": "S-ID",
        "auc": "AUC",
        "auprc": "AUPRC",
        "mcc": "MCC",
        "f1": "F1",
        "fpr": "FPR",
    }
    table = table.rename(columns={k: v for k, v in display_cols.items() if k in table.columns})

    tex = _to_latex_booktabs(
        table,
        caption="Main results: AUC, AUPRC, MCC, F1 and FPR per model and scenario.",
        label="tab:results",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex, encoding="utf-8")
    _logger.info("Wrote results table → %s", output_path)
    return tex


def export_ablation_table(
    ablation_df: pd.DataFrame,
    output_path: str | Path,
    metrics: list[str] | None = None,
) -> str:
    """Export the ablation results table as a booktabs LaTeX string.

    Parameters
    ----------
    ablation_df:
        DataFrame with at least a ``name`` column (ablation variant) and
        metric columns.
    output_path:
        Destination ``.tex`` file.

    Returns
    -------
    str — the LaTeX table string.
    """
    if metrics is None:
        metrics = [m for m in ["auc", "f1", "mcc", "auprc"] if m in ablation_df.columns]

    keep = [c for c in ["name", "model_type"] + metrics if c in ablation_df.columns]
    table = ablation_df[keep].copy()
    table = bold_best(table, metrics)

    display = {"name": "Variant", "model_type": "Model", "auc": "AUC",
                "f1": "F1", "mcc": "MCC", "auprc": "AUPRC"}
    table = table.rename(columns={k: v for k, v in display.items() if k in table.columns})

    tex = _to_latex_booktabs(
        table,
        caption="Ablation study results.",
        label="tab:ablation",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex, encoding="utf-8")
    _logger.info("Wrote ablation table → %s", output_path)
    return tex


def export_significance_table(
    sig_df: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Export the statistical significance table as a booktabs LaTeX string.

    Adds a ``Stars`` column: ``**`` (p < 0.01), ``*`` (p < 0.05), ``ns``
    (not significant).  The ``p_bonferroni`` column is used when available,
    otherwise ``p_value`` / ``p_raw`` is used.

    Parameters
    ----------
    sig_df:
        DataFrame as returned by :func:`~gbd_ctu.evaluation.stats.full_significance_table`.
    output_path:
        Destination ``.tex`` file.

    Returns
    -------
    str — the LaTeX table string.
    """
    table = sig_df.copy()

    p_col = next(
        (c for c in ["p_bonferroni", "p_value", "p_raw"] if c in table.columns),
        None,
    )

    def _stars(p: float) -> str:
        if np.isnan(p):
            return "—"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    if p_col is not None:
        table["Stars"] = table[p_col].apply(
            lambda v: _stars(float(v)) if pd.notna(v) else "—"
        )

    keep_cols = [c for c in [
        "model_a", "model_b", "metric", "statistic", p_col,
        "effect_size", "effect_size_d", "n_scenarios", "Stars",
    ] if c is not None and c in table.columns]
    table = table[keep_cols]

    display = {
        "model_a": "Model A", "model_b": "Model B",
        "metric": "Metric", "statistic": "W",
        "p_value": "$p$", "p_raw": "$p$", "p_bonferroni": "$p$ (Bonf.)",
        "effect_size": "Effect", "effect_size_d": "Cohen $d$",
        "n_scenarios": "$n$", "Stars": "",
    }
    table = table.rename(columns={k: v for k, v in display.items() if k in table.columns})

    tex = _to_latex_booktabs(
        table,
        caption="Wilcoxon signed-rank test with Bonferroni correction. "
                r"$^{**}p<0.01$, $^{*}p<0.05$, ns: not significant.",
        label="tab:significance",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex, encoding="utf-8")
    _logger.info("Wrote significance table → %s", output_path)
    return tex


def export_dataset_table(
    stats_df: pd.DataFrame,
    output_path: str | Path,
) -> str:
    """Export the dataset statistics table as a booktabs LaTeX string.

    Parameters
    ----------
    stats_df:
        DataFrame as returned by
        :func:`~gbd_ctu.data.ctu13_loader.dataset_statistics_table`.
    output_path:
        Destination ``.tex`` file.

    Returns
    -------
    str — the LaTeX table string.
    """
    display_cols = {
        "scenario_id": "S-ID",
        "family": "Family",
        "total_flows": "Flows",
        "botnet_flows": "Botnet",
        "botnet_frac": "Bot\\%",
        "unique_src_ips": "Src IPs",
        "botnet_host_ips": "Bot IPs",
        "unique_dst_ports": "Ports",
        "mean_duration": "Dur (mean)",
        "max_duration": "Dur (max)",
    }
    keep = [c for c in display_cols if c in stats_df.columns]
    table = stats_df[keep].copy()
    table = table.rename(columns={k: display_cols[k] for k in keep})

    tex = _to_latex_booktabs(
        table,
        caption="CTU-13 per-scenario dataset statistics.",
        label="tab:dataset",
        float_format="%.4f",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex, encoding="utf-8")
    _logger.info("Wrote dataset table → %s", output_path)
    return tex


# ---------------------------------------------------------------------------
# Architecture diagrams
# ---------------------------------------------------------------------------

_ARCH_DOTS: dict[str, str] = {
    "graphsage": """\
digraph GraphSAGE {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor="#aec7e8", fontname="Helvetica", fontsize=10];
    edge [fontsize=9];
    Input   [label="Input\\n(N × in_ch)"];
    SAGE1   [label="SAGEConv 1\\n(in → hidden)\\n+ ReLU + Dropout"];
    SAGE2   [label="SAGEConv 2\\n(hidden → hidden)\\n+ ReLU + Dropout"];
    SAGE3   [label="SAGEConv 3\\n(hidden → hidden)\\n+ ReLU"];
    Linear  [label="Linear\\n(hidden → 2)"];
    Output  [label="Logits\\n(N × 2)", fillcolor="#ffbb78"];
    Input  -> SAGE1 -> SAGE2 -> SAGE3 -> Linear -> Output;
}
""",
    "gat": """\
digraph GAT {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor="#98df8a", fontname="Helvetica", fontsize=10];
    edge [fontsize=9];
    Input  [label="Input\\n(N × in_ch)"];
    GAT1   [label="GATConv 1\\n(in → hidden, heads)\\n+ ELU + Dropout"];
    GAT2   [label="GATConv 2\\n(hidden×heads → hidden)\\n+ ELU"];
    Linear [label="Linear\\n(hidden → 2)"];
    Output [label="Logits\\n(N × 2)", fillcolor="#ffbb78"];
    Input -> GAT1 -> GAT2 -> Linear -> Output;
}
""",
    "hybrid": """\
digraph HybridGNN {
    rankdir=TB;
    node [shape=box, style=filled, fontname="Helvetica", fontsize=10];
    edge [fontsize=9];
    Input    [label="Input\\n(N × in_ch)", fillcolor="#c7c7c7"];
    SAGE1    [label="SAGEConv 1\\n+ ReLU + Dropout", fillcolor="#aec7e8"];
    SAGE2    [label="SAGEConv 2\\n+ ReLU", fillcolor="#aec7e8"];
    GAT1     [label="GATConv 1\\n+ ELU + Dropout", fillcolor="#98df8a"];
    GAT2     [label="GATConv 2\\n+ ELU", fillcolor="#98df8a"];
    Concat   [label="Concat\\n(SAGE + GAT)", fillcolor="#ffbb78", shape=ellipse];
    Linear1  [label="Linear + ReLU + Dropout", fillcolor="#c5b0d5"];
    Linear2  [label="Linear\\n(→ 2)", fillcolor="#c5b0d5"];
    Output   [label="Logits\\n(N × 2)", fillcolor="#ffbb78"];
    Input -> SAGE1 -> SAGE2 -> Concat;
    Input -> GAT1  -> GAT2  -> Concat;
    Concat -> Linear1 -> Linear2 -> Output;
}
""",
    "mlp": """\
digraph MLP {
    rankdir=TB;
    node [shape=box, style=filled, fillcolor="#c5b0d5", fontname="Helvetica", fontsize=10];
    edge [fontsize=9];
    Input  [label="Flow Features\\n(N × 22)", fillcolor="#c7c7c7"];
    L1     [label="Linear(22→128)\\n+ BN + ReLU + Dropout"];
    L2     [label="Linear(128→64)\\n+ BN + ReLU + Dropout"];
    L3     [label="Linear(64→32)\\n+ BN + ReLU"];
    L4     [label="Linear(32→2)"];
    Output [label="Logits\\n(N × 2)", fillcolor="#ffbb78"];
    Input -> L1 -> L2 -> L3 -> L4 -> Output;
}
""",
}


def draw_model_architecture(
    model_type: str,
    output_path: str | Path,
) -> Path:
    """Write a Graphviz ``.dot`` file for *model_type*'s layer graph.

    The ``.dot`` file can be compiled with::

        dot -Tpdf architecture_hybrid.dot -o architecture_hybrid.pdf

    Parameters
    ----------
    model_type:
        One of ``"hybrid"``, ``"gat"``, ``"graphsage"``, ``"mlp"``.
    output_path:
        Destination path (usually ``*.dot``).

    Returns
    -------
    Path — the written file path.

    Raises
    ------
    ValueError
        When *model_type* is not recognised.
    """
    key = model_type.lower().replace("-", "_")
    if key not in _ARCH_DOTS:
        raise ValueError(
            f"Unknown model_type {model_type!r}. "
            f"Choose from: {list(_ARCH_DOTS)}"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_ARCH_DOTS[key], encoding="utf-8")
    _logger.info("Wrote architecture .dot → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def scenario_heatmap_figure(
    metrics_df: pd.DataFrame,
    metric: str = "auc",
    output_dir: str | Path = "paper/figures",
    dpi: int = _DPI,
) -> Path:
    """Save a scenario × model heatmap for *metric*.

    Parameters
    ----------
    metrics_df:
        Long-format DataFrame with columns ``model``, ``scenario`` (or
        ``scenario_id``), and *metric*.
    metric:
        Column name to visualise.
    output_dir:
        Directory where ``heatmap_{metric}.png`` is written.
    dpi:
        Figure resolution.

    Returns
    -------
    Path — the written PNG file.
    """
    _require_mpl()
    apply_paper_style()

    df = metrics_df.copy()
    scenario_col = "scenario" if "scenario" in df.columns else "scenario_id"
    df["model"] = df["model"].str.lower()

    if metric not in df.columns:
        raise ValueError(f"Column {metric!r} not found in metrics_df.")

    pivot = df.pivot_table(
        index=scenario_col, columns="model", values=metric, aggfunc="mean"
    )

    n_scenarios, n_models = pivot.shape
    fig_h = max(4, n_scenarios * 0.35)
    fig_w = max(4, n_models * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = "RdYlGn_r" if metric in LOWER_BETTER else "RdYlGn"
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=30, ha="right")
    ax.set_yticks(range(n_scenarios))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(f"{metric.upper()} per scenario × model")

    for i in range(n_scenarios):
        for j in range(n_models):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / f"heatmap_{metric}.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Wrote heatmap → %s", out)
    return out


def ablation_bar_figure(
    ablation_df: pd.DataFrame,
    metrics: list[str] | None = None,
    output_dir: str | Path = "paper/figures",
    dpi: int = _DPI,
) -> Path:
    """Save a grouped bar chart for ablation variants × metrics.

    Parameters
    ----------
    ablation_df:
        DataFrame with a ``name`` column (ablation variant) and metric columns.
    metrics:
        Metric columns to plot.  Defaults to available from
        ``["auc", "f1", "mcc", "auprc"]``.
    output_dir:
        Directory where ``ablation_bar.png`` is written.

    Returns
    -------
    Path — the written PNG file.
    """
    _require_mpl()
    apply_paper_style()

    if metrics is None:
        metrics = [m for m in ["auc", "f1", "mcc", "auprc"] if m in ablation_df.columns]
    metrics = [m for m in (metrics or []) if m in ablation_df.columns]
    if not metrics:
        raise ValueError("No metric columns found in ablation_df.")

    name_col = "name" if "name" in ablation_df.columns else ablation_df.columns[0]
    df = ablation_df[[name_col] + metrics].copy().set_index(name_col)

    n_variants = len(df)
    n_metrics = len(metrics)
    bar_w = 0.8 / n_metrics
    x = np.arange(n_variants)

    fig, ax = plt.subplots(figsize=(max(6, n_variants * 1.2), 4))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, metric in enumerate(metrics):
        vals = df[metric].values
        ax.bar(
            x + (i - n_metrics / 2 + 0.5) * bar_w,
            vals, bar_w * 0.9,
            label=metric.upper(),
            color=colours[i % len(colours)],
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df.index.tolist(), rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Ablation study")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "ablation_bar.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    _logger.info("Wrote ablation bar chart → %s", out)
    return out


def per_scenario_roc_pr_figure(
    records: list[dict[str, Any]],
    output_dir: str | Path = "paper/figures",
    dpi: int = _DPI,
    max_scenarios: int = 13,
) -> tuple[Path, Path]:
    """Save per-scenario ROC and PR subplot grids.

    Each subplot corresponds to one scenario; curves from different models
    are overlaid.  Returns a 2-tuple ``(roc_path, pr_path)``.

    Parameters
    ----------
    records:
        List of dicts; each must have keys ``model``, ``scenario_id`` (or
        ``scenario``), ``y_true``, and ``y_score``.
    output_dir:
        Directory where the two PNGs are written.
    max_scenarios:
        Cap on the number of subplot columns × rows (default 13 for CTU-13).

    Returns
    -------
    (roc_path, pr_path) — paths to the written PNG files.
    """
    _require_mpl()
    apply_paper_style()

    from sklearn.metrics import roc_curve, precision_recall_curve, auc as _auc

    # Group by scenario
    scenarios: dict[str, list[dict]] = {}
    for rec in records:
        sid = str(rec.get("scenario_id", rec.get("scenario", "?")))
        scenarios.setdefault(sid, []).append(rec)

    sids = sorted(scenarios)[:max_scenarios]
    n = len(sids)
    if n == 0:
        raise ValueError("No records to plot.")

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    def _make_grid(title_prefix: str):
        return plt.subplots(
            nrows, ncols,
            figsize=(ncols * 3.2, nrows * 3.0),
            squeeze=False,
        )

    fig_roc, axes_roc = _make_grid("ROC")
    fig_pr, axes_pr = _make_grid("PR")

    seen_models: set[str] = set()

    for plot_i, sid in enumerate(sids):
        row, col = divmod(plot_i, ncols)
        ax_r = axes_roc[row][col]
        ax_p = axes_pr[row][col]

        ax_r.plot([0, 1], [0, 1], "k--", lw=0.5)
        ax_r.set_title(f"S{sid}", fontsize=9)
        ax_p.set_title(f"S{sid}", fontsize=9)

        for rec in scenarios[sid]:
            model_name = str(rec.get("model", "?")).lower()
            y_true = np.asarray(rec["y_true"], dtype=int)
            y_score = np.asarray(rec["y_score"], dtype=float)
            if np.unique(y_true).shape[0] < 2:
                continue
            colour = PALETTE.get(model_name, "#333333")
            lbl = model_name if model_name not in seen_models else "_nolegend_"
            seen_models.add(model_name)

            fpr_a, tpr_a, _ = roc_curve(y_true, y_score)
            ax_r.plot(fpr_a, tpr_a, color=colour, lw=1.0, alpha=0.8, label=lbl)

            p_a, r_a, _ = precision_recall_curve(y_true, y_score)
            ax_p.plot(r_a, p_a, color=colour, lw=1.0, alpha=0.8, label=lbl)

        ax_r.set(xlim=[0, 1], ylim=[0, 1])
        ax_p.set(xlim=[0, 1], ylim=[0, 1])

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes_roc[r][c].set_visible(False)
        axes_pr[r][c].set_visible(False)

    # Add shared legend from first subplot
    handles, labels = axes_roc[0][0].get_legend_handles_labels()
    real = [(h, l) for h, l in zip(handles, labels) if not l.startswith("_")]
    if real:
        fig_roc.legend(*zip(*real), loc="lower center",
                       ncol=min(6, len(real)), fontsize=8,
                       bbox_to_anchor=(0.5, -0.02))
        fig_pr.legend(*zip(*real), loc="lower center",
                      ncol=min(6, len(real)), fontsize=8,
                      bbox_to_anchor=(0.5, -0.02))

    for fig, xlabel, ylabel, name in [
        (fig_roc, "FPR", "TPR", "per_scenario_roc.png"),
        (fig_pr, "Recall", "Precision", "per_scenario_pr.png"),
    ]:
        fig.text(0.5, -0.01, xlabel, ha="center", fontsize=10)
        fig.text(-0.01, 0.5, ylabel, va="center", rotation="vertical", fontsize=10)
        fig.tight_layout()
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        p = dest / name
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        _logger.info("Wrote %s → %s", name, p)

    return (
        Path(output_dir) / "per_scenario_roc.png",
        Path(output_dir) / "per_scenario_pr.png",
    )
