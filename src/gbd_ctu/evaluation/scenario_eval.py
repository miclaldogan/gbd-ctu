"""GBD-CTU scenario-wise evaluation.

This module evaluates trained GNN checkpoints across CTU-13 scenarios. Inputs
are serialized graphs and a saved checkpoint; outputs are scenario-level metric
tables persisted to CSV and JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - optional during static inspection
    torch = None

from gbd_ctu.data.graph_builder import load_graphs
from gbd_ctu.evaluation.metrics import classification_metrics, metrics_frame
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


def evaluate_gnn_checkpoint(graph_dir: str | Path, checkpoint_path: str | Path, output_path: str | Path):
    """Evaluate a saved GNN checkpoint on the test split of each scenario graph."""

    _require_torch()
    graphs = load_graphs(graph_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    family = checkpoint.get("family", "hybrid")
    model_class = MODEL_FAMILIES.get(family)
    if model_class is None:
        raise ValueError(f"Unsupported model family in checkpoint: {family}")
    model = model_class(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    records = []
    with torch.no_grad():
        for graph in graphs:
            logits = model(graph)
            probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels = graph.y.cpu().numpy()
            mask = graph.test_mask.cpu().numpy().astype(bool)
            if not mask.any():
                continue
            metrics = classification_metrics(labels[mask], probabilities[mask])
            metrics.update({"model": family, "scenario": graph.scenario, "split": "test"})
            records.append(metrics)

    report = metrics_frame(records)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False)
    destination.with_suffix(".json").write_text(json.dumps(report.to_dict(orient="records"), indent=2), encoding="utf-8")
    return report
