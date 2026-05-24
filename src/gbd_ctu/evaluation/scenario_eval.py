"""GBD-CTU scenario-wise evaluation.

This module evaluates trained GNN checkpoints across CTU-13 scenarios. Inputs
are serialized graphs and a saved checkpoint; outputs are scenario-level metric
tables persisted to CSV and JSON.

Public API
----------
``evaluate_all_scenarios``
    Evaluate any model that accepts a PyG ``Data`` object and returns per-node
    logits.  Returns a ``pd.DataFrame`` and optionally saves it to CSV.

``evaluate_gnn_checkpoint``
    Convenience wrapper that loads a ``*.pt`` checkpoint, rebuilds the model,
    and calls ``evaluate_all_scenarios``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None
    Data = None

from gbd_ctu.data.graph_builder import load_graphs
from gbd_ctu.evaluation.metrics import classification_metrics, metrics_frame
from gbd_ctu.models.gnn import build_gnn_from_config
from gbd_ctu.models.gnn.gat import GATNodeClassifier
from gbd_ctu.models.gnn.graphsage import GraphSAGENodeClassifier
from gbd_ctu.models.gnn.hybrid import GraphSageGATHybridClassifier


MODEL_FAMILIES = {
    "graphsage": GraphSAGENodeClassifier,
    "gat": GATNodeClassifier,
    "hybrid": GraphSageGATHybridClassifier,
}


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for GNN evaluation.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_all_scenarios(
    model: Any,
    graphs: "list[Data]",
    model_name: str = "gnn",
    output_dir: str | Path | None = None,
    split: str = "test",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate *model* on every graph and return a per-scenario metrics table.

    The returned ``pd.DataFrame`` has one row per scenario with columns::

        scenario_id | scenario | auc | f1 | precision | recall | fpr |
        n_botnet_nodes | n_total_nodes

    Parameters
    ----------
    model:
        Any callable (PyTorch ``nn.Module``) that accepts a PyG ``Data`` object
        and returns per-node logits of shape ``(N, 2)``.
    graphs:
        List of PyG ``Data`` objects, one per CTU-13 scenario.
    model_name:
        Used to name the saved CSV file (``results_{model_name}.csv``).
    output_dir:
        If provided, the DataFrame is written to
        ``{output_dir}/results_{model_name}.csv``.
    split:
        Which mask to use: ``"train"``, ``"val"``, or ``"test"``.
    threshold:
        Decision threshold for converting scores to binary predictions.
    """
    _require_torch()
    model.eval()
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for graph in graphs:
            mask_attr = f"{split}_mask"
            mask = getattr(graph, mask_attr, None)
            if mask is None or not mask.any():
                continue

            logits = model(graph)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels = graph.y.cpu().numpy()
            mask_np = mask.cpu().numpy().astype(bool)

            metrics = classification_metrics(labels[mask_np], probs[mask_np],
                                             threshold=threshold)
            scenario_id = int(getattr(graph, "scenario_id", 0))
            n_botnet = int((labels[mask_np] == 1).sum())
            n_total = int(mask_np.sum())

            records.append({
                "scenario_id": scenario_id,
                "scenario": getattr(graph, "scenario", f"scenario-{scenario_id:02d}"),
                "auc": metrics["auc"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "fpr": metrics["fpr"],
                "n_botnet_nodes": n_botnet,
                "n_total_nodes": n_total,
            })

    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame = frame.sort_values("scenario_id").reset_index(drop=True)

    if output_dir is not None:
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)
        csv_path = dest / f"results_{model_name}.csv"
        frame.to_csv(csv_path, index=False)

    return frame


def evaluate_gnn_checkpoint(
    graph_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    model_name: str | None = None,
) -> pd.DataFrame:
    """Evaluate a saved GNN checkpoint on the test split of each scenario graph.

    Parameters
    ----------
    graph_dir:
        Directory of serialised ``.pt`` graph files.
    checkpoint_path:
        Path to the ``*.pt`` checkpoint written by ``train_gnn``.
    output_path:
        CSV destination path.
    model_name:
        Override the model family name used in the CSV (defaults to the
        ``family`` key stored inside the checkpoint).
    """
    _require_torch()
    graphs = load_graphs(graph_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    family = checkpoint.get("family", "hybrid")
    effective_name = model_name or family

    config: dict[str, Any] = {
        "model_type": family,
        family: checkpoint.get("model_kwargs", {}),
    }
    # Remap graphsage kwargs key (hidden_channels → hidden_dim for that family)
    if family == "graphsage" and "hidden_channels" in config.get(family, {}):
        kw = config[family].copy()
        kw["hidden_dim"] = kw.pop("hidden_channels")
        config[family] = kw

    in_channels = checkpoint["model_kwargs"].get("in_channels")
    model = build_gnn_from_config(in_channels=in_channels, config=config)
    model.load_state_dict(checkpoint["model_state"])

    dest = Path(output_path)
    frame = evaluate_all_scenarios(
        model=model,
        graphs=graphs,
        model_name=effective_name,
        output_dir=dest.parent,
        split="test",
    )

    # Also write legacy-compat individual CSV at the exact output_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    dest.with_suffix(".json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    return frame

