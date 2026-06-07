"""GBD-CTU leave-one-scenario-out and cross-family generalisation evaluation.

Public API
----------
``leave_one_scenario_out``
    Train on all scenarios except one, evaluate on the held-out scenario's
    test split.  Repeats for every scenario and returns a per-fold DataFrame.

``cross_family_eval``
    Group scenarios by malware family.  For each family, train on all other
    families and evaluate on the held-out family's test splits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from gbd_ctu.evaluation.metrics import classification_metrics

_logger = logging.getLogger(__name__)

_METRIC_COLS = ["auc", "auprc", "mcc", "f1", "fpr", "fpr_at_tpr95"]


def _eval_on_graph(model: Any, graph: Any) -> dict[str, Any] | None:
    """Run inference on *graph*'s test split; return metric dict or None."""
    mask = graph.test_mask.cpu().numpy().astype(bool)
    if not mask.any():
        return None
    with torch.no_grad():
        logits = model(graph)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    labels = graph.y.cpu().numpy()
    return classification_metrics(labels[mask], probs[mask])


def _summary_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        col: {"mean": float(np.nanmean(df[col])), "std": float(np.nanstd(df[col]))}
        for col in _METRIC_COLS
        if col in df.columns
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def leave_one_scenario_out(
    graphs: list,
    model_factory: Callable[[int], Any],
    trainer: Callable[[Any, list], None],
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Leave-one-scenario-out (LOSO) evaluation.

    For each graph ``i`` in *graphs*, trains a fresh model on all other graphs
    then evaluates on ``graphs[i]``'s test split.

    Parameters
    ----------
    graphs:
        List of PyG ``Data`` objects, one per CTU-13 scenario.
    model_factory:
        Callable ``(in_channels: int) -> nn.Module`` that returns an
        untrained model.
    trainer:
        Callable ``(model, train_graphs) -> None`` that trains *model*
        in-place on *train_graphs*.
    output_dir:
        When provided, writes ``loso_results.csv`` and ``loso_summary.json``
        to this directory.

    Returns
    -------
    pd.DataFrame
        One row per held-out scenario with columns
        ``held_out_scenario``, ``scenario``, ``auc``, ``auprc``, ``mcc``,
        ``f1``, ``fpr``, ``fpr_at_tpr95``.
    """
    if torch is None:
        raise ImportError("torch and torch-geometric are required for LOSO evaluation.")

    records: list[dict[str, Any]] = []

    for i, held_out in enumerate(graphs):
        train_graphs = [g for j, g in enumerate(graphs) if j != i]
        scenario_id = int(getattr(held_out, "scenario_id", i))
        scenario_name = getattr(held_out, "scenario", f"scenario-{scenario_id:02d}")

        _logger.info(
            "LOSO fold %d/%d: held-out scenario %s, training on %d graphs",
            i + 1, len(graphs), scenario_id, len(train_graphs),
        )

        if not train_graphs:
            _logger.warning("No training graphs for fold %d; skipping.", i + 1)
            continue

        in_channels = int(held_out.num_node_features)
        model = model_factory(in_channels)
        model.eval()
        trainer(model, train_graphs)
        model.eval()

        metrics = _eval_on_graph(model, held_out)
        if metrics is None:
            _logger.warning("No test nodes in held-out scenario %s; skipping.", scenario_id)
            continue

        record: dict[str, Any] = {
            "held_out_scenario": scenario_id,
            "scenario": scenario_name,
            **{k: metrics[k] for k in _METRIC_COLS if k in metrics},
        }
        records.append(record)
        _logger.info(
            "Fold %d done: AUC=%.4f AUPRC=%.4f MCC=%.4f F1=%.4f",
            i + 1, metrics["auc"], metrics["auprc"], metrics["mcc"], metrics["f1"],
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("held_out_scenario").reset_index(drop=True)

    summary = _summary_stats(df) if not df.empty else {}
    _logger.info(
        "LOSO summary (%d folds): %s",
        len(df),
        "  ".join(f"{k}={v['mean']:.4f}±{v['std']:.4f}" for k, v in summary.items()),
    )

    if output_dir is not None:
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest / "loso_results.csv", index=False)
        (dest / "loso_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    return df


def cross_family_eval(
    graphs: list,
    scenario_family_map: dict[int, str],
    model_factory: Callable[[int], Any],
    trainer: Callable[[Any, list], None],
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Cross-malware-family generalisation evaluation.

    Groups scenarios by malware family (using *scenario_family_map*).  For
    each family trains on all other families and evaluates on every scenario
    belonging to the held-out family.

    Parameters
    ----------
    graphs:
        List of PyG ``Data`` objects.
    scenario_family_map:
        Mapping ``{scenario_id: family_name}``.  Scenarios absent from the
        map are grouped under ``"Unknown"``.
    model_factory:
        Same contract as in :func:`leave_one_scenario_out`.
    trainer:
        Same contract as in :func:`leave_one_scenario_out`.
    output_dir:
        When provided, writes ``cross_family_results.csv`` and
        ``cross_family_summary.csv``.

    Returns
    -------
    pd.DataFrame
        One row per held-out scenario with columns ``held_out_family``,
        ``held_out_scenario``, ``scenario``, plus metric columns.
    """
    if torch is None:
        raise ImportError("torch and torch-geometric are required for cross-family evaluation.")

    family_groups: dict[str, list] = {}
    for graph in graphs:
        sid = int(getattr(graph, "scenario_id", 0))
        family = scenario_family_map.get(sid, "Unknown")
        family_groups.setdefault(family, []).append(graph)

    unknown_count = len(family_groups.get("Unknown", []))
    if unknown_count:
        _logger.warning(
            "%d graph(s) have no family mapping and are grouped under 'Unknown'.",
            unknown_count,
        )

    records: list[dict[str, Any]] = []
    families = sorted(family_groups)

    for held_family in families:
        held_graphs = family_groups[held_family]
        train_graphs = [
            g for fam, gs in family_groups.items() if fam != held_family for g in gs
        ]

        if not train_graphs:
            _logger.warning(
                "No training graphs when holding out family '%s'; skipping.", held_family
            )
            continue

        _logger.info(
            "Cross-family fold: held-out '%s' (%d graphs), training on %d graphs",
            held_family, len(held_graphs), len(train_graphs),
        )

        in_channels = int(held_graphs[0].num_node_features)
        model = model_factory(in_channels)
        trainer(model, train_graphs)
        model.eval()

        for graph in held_graphs:
            sid = int(getattr(graph, "scenario_id", 0))
            metrics = _eval_on_graph(model, graph)
            if metrics is None:
                continue
            records.append({
                "held_out_family": held_family,
                "held_out_scenario": sid,
                "scenario": getattr(graph, "scenario", f"scenario-{sid:02d}"),
                **{k: metrics[k] for k in _METRIC_COLS if k in metrics},
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["held_out_family", "held_out_scenario"]).reset_index(drop=True)

    if output_dir is not None and not df.empty:
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest / "cross_family_results.csv", index=False)

        summary = (
            df.groupby("held_out_family")[[c for c in _METRIC_COLS if c in df.columns]]
            .agg(["mean", "std"])
            .round(4)
        )
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        summary.reset_index().to_csv(dest / "cross_family_summary.csv", index=False)

    return df
