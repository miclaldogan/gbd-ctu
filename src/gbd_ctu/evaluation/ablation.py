"""GBD-CTU ablation study.

Public API
----------
``AblationConfig``
    Dataclass describing one ablation variant (model, graph, loss settings).

``configs_from_yaml``
    Load a list of ``AblationConfig`` objects from the bundled YAML file.

``run_ablation_suite``
    Run all ablation configs, return a wide-format summary DataFrame with
    columns ``name``, ``mean_auc``, ``std_auc``, ``mean_auprc``, ``std_auprc``,
    ``mean_mcc``, ``mean_f1``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover
    _yaml = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from gbd_ctu.evaluation.metrics import classification_metrics

_logger = logging.getLogger(__name__)

_DEFAULT_BUNDLED_YAML = Path(__file__).parent.parent / "configs" / "ablation.yaml"

_NODE_FEATURE_COLUMNS = [
    "flow_count",
    "mean_duration",
    "total_bytes_sent",
    "total_bytes_recv",
    "unique_dst_count",
    "botnet_flow_ratio",
]

_SUMMARY_METRICS = ["auc", "auprc", "mcc", "f1"]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """Configuration for a single ablation variant.

    Parameters
    ----------
    name:
        Unique identifier used as the row label in the summary DataFrame.
    model_kwargs:
        Keys: ``type`` (graphsage|gat|hybrid), ``hidden``, ``heads``,
        ``num_layers``, ``dropout``.
    graph_kwargs:
        Keys: ``use_edge_attr`` (bool), ``node_features`` (list or "all"),
        ``undirected`` (bool), ``self_loops`` (bool).
    trainer_kwargs:
        Keys: ``loss`` (dict with ``type``, ``gamma``, ``alpha``),
        ``epochs``, ``lr``.
    """

    name: str
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    graph_kwargs: dict[str, Any] = field(default_factory=dict)
    trainer_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml_dict(cls, entry: dict[str, Any]) -> "AblationConfig":
        """Build an ``AblationConfig`` from a YAML ablation entry.

        Accepts the short YAML format used in ``ablation.yaml``:
        ``model``, ``graph``, and ``loss`` top-level keys are mapped to the
        corresponding dataclass fields.
        """
        loss = entry.pop("loss", {})
        return cls(
            name=entry["name"],
            model_kwargs=entry.get("model", {}),
            graph_kwargs=entry.get("graph", {}),
            trainer_kwargs={"loss": loss, **entry.get("trainer", {})},
        )


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def configs_from_yaml(path: str | Path | None = None) -> list[AblationConfig]:
    """Load ablation configs from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML file.  Defaults to the bundled
        ``src/gbd_ctu/configs/ablation.yaml``.
    """
    if _yaml is None:
        raise ImportError("pyyaml is required to load ablation configs.")
    yaml_path = Path(path) if path is not None else _DEFAULT_BUNDLED_YAML
    raw = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return [AblationConfig.from_yaml_dict(entry) for entry in raw.get("ablations", [])]


# ---------------------------------------------------------------------------
# Graph transforms
# ---------------------------------------------------------------------------

def _resolve_node_feature_indices(node_features: list[str] | str) -> list[int] | None:
    """Return column indices for the requested node features, or None for all."""
    if node_features == "all" or node_features is None:
        return None
    return [_NODE_FEATURE_COLUMNS.index(f) for f in node_features]


def _apply_graph_transforms(graphs: list, graph_kwargs: dict) -> list:
    """Return a list of (shallow-copied) graphs with ablation transforms applied."""
    import copy

    use_edge_attr: bool = graph_kwargs.get("use_edge_attr", True)
    node_features = graph_kwargs.get("node_features", "all")
    undirected: bool = graph_kwargs.get("undirected", False)
    self_loops: bool = graph_kwargs.get("self_loops", False)

    feature_indices = _resolve_node_feature_indices(node_features)

    result = []
    for g in graphs:
        g2 = copy.copy(g)

        if not use_edge_attr:
            g2.edge_attr = None

        if feature_indices is not None:
            if torch is not None:
                g2.x = g.x[:, feature_indices].clone()
            else:
                g2.x = g.x[:, feature_indices]

        if undirected or self_loops:
            try:
                from torch_geometric.utils import add_self_loops, to_undirected
            except ImportError as exc:
                raise ImportError(
                    "torch_geometric is required for undirected/self_loops transforms."
                ) from exc

            if undirected:
                if g2.edge_attr is not None:
                    ei, ea = to_undirected(
                        g2.edge_index, g2.edge_attr, num_nodes=g2.num_nodes, reduce="mean"
                    )
                    g2.edge_index = ei
                    g2.edge_attr = ea
                else:
                    g2.edge_index = to_undirected(g2.edge_index, num_nodes=g2.num_nodes)

            if self_loops:
                n_edges_before = g2.edge_index.shape[1]
                g2.edge_index, _ = add_self_loops(g2.edge_index, num_nodes=g2.num_nodes)
                if g2.edge_attr is not None:
                    # Pad new self-loop edges with zeros
                    n_new = g2.edge_index.shape[1] - n_edges_before
                    if n_new > 0:
                        pad = torch.zeros(
                            n_new, g2.edge_attr.shape[1], dtype=g2.edge_attr.dtype
                        )
                        g2.edge_attr = torch.cat([g2.edge_attr, pad], dim=0)

        result.append(g2)
    return result


# ---------------------------------------------------------------------------
# Single-config runner
# ---------------------------------------------------------------------------

def _build_criterion(loss_kwargs: dict):
    """Instantiate the loss function from trainer_kwargs['loss']."""
    from gbd_ctu.training.losses import FocalLoss
    import torch.nn as nn

    loss_type = loss_kwargs.get("type", "focal")
    if loss_type == "focal":
        return FocalLoss(
            gamma=float(loss_kwargs.get("gamma", 2.0)),
            alpha=float(loss_kwargs.get("alpha", 0.25)),
        )
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss type '{loss_type}'. Choose 'focal' or 'cross_entropy'.")


def _run_single_config(
    config: AblationConfig,
    graphs: list,
    seed: int,
) -> list[dict[str, Any]]:
    """Train one ablation variant; return per-scenario metric dicts."""
    if torch is None:
        raise ImportError("torch and torch-geometric are required for ablation training.")

    from gbd_ctu.training.trainer_gnn import _build_model, _train_one_epoch

    transformed = _apply_graph_transforms(graphs, config.graph_kwargs)

    in_channels = int(transformed[0].num_node_features)
    mk = config.model_kwargs
    family = mk.get("type", "hybrid")
    model = _build_model(
        family=family,
        in_channels=in_channels,
        hidden_channels=mk.get("hidden", 64),
        out_channels=2,
        heads=mk.get("heads", 4),
        num_layers=mk.get("num_layers", 3),
        dropout=mk.get("dropout", 0.3),
    )

    tk = config.trainer_kwargs
    criterion = _build_criterion(tk.get("loss", {}))
    lr = float(tk.get("lr", 1e-3))
    epochs = int(tk.get("epochs", 20))

    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        _train_one_epoch(
            model, transformed, criterion, optimizer,
            batch_size=512, num_neighbors=[10, 5],
        )

    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for graph in transformed:
            mask = graph.test_mask.cpu().numpy().astype(bool)
            if not mask.any():
                continue
            logits = model(graph)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            labels = graph.y.cpu().numpy()
            records.append(classification_metrics(labels[mask], probs[mask]))

    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_ablation_suite(
    configs: list[AblationConfig],
    graph_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
) -> pd.DataFrame:
    """Run all ablation configs and return a wide-format summary DataFrame.

    For each :class:`AblationConfig`:

    1. Load graphs from *graph_dir*.
    2. Apply graph-level transforms from ``config.graph_kwargs``.
    3. Instantiate model from ``config.model_kwargs``.
    4. Train with loss / LR from ``config.trainer_kwargs``.
    5. Evaluate test split per scenario; record AUC, AUPRC, MCC, F1.

    Parameters
    ----------
    configs:
        List of ablation configurations to run.
    graph_dir:
        Directory of serialized ``.pt`` graph files.
    output_dir:
        Results are written here (``ablation_results.csv``,
        ``ablation_results.json``).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per ablation with columns
        ``name``, ``mean_auc``, ``std_auc``, ``mean_auprc``, ``std_auprc``,
        ``mean_mcc``, ``std_mcc``, ``mean_f1``, ``std_f1``.

    Notes
    -----
    To export results as a LaTeX table::

        from gbd_ctu.evaluation.ablation import run_ablation_suite, configs_from_yaml
        df = run_ablation_suite(configs_from_yaml(), graph_dir="data/processed",
                                output_dir="results/ablation")
        print(df.to_latex(index=False, float_format="%.4f"))
    """
    from gbd_ctu.data.graph_builder import load_graphs

    graphs = load_graphs(graph_dir)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for cfg in configs:
        _logger.info("Running ablation: %s", cfg.name)
        try:
            per_scenario = _run_single_config(cfg, graphs, seed)
        except Exception as exc:
            _logger.error("Ablation '%s' failed: %s", cfg.name, exc)
            rows.append({"name": cfg.name, "error": str(exc)})
            continue

        if not per_scenario:
            _logger.warning("Ablation '%s' produced no per-scenario results.", cfg.name)
            continue

        df_s = pd.DataFrame(per_scenario)
        row: dict[str, Any] = {"name": cfg.name}
        for col in _SUMMARY_METRICS:
            if col in df_s.columns:
                row[f"mean_{col}"] = float(np.nanmean(df_s[col]))
                row[f"std_{col}"] = float(np.nanstd(df_s[col]))
        rows.append(row)
        _logger.info(
            "  %s: AUC=%.4f±%.4f  AUPRC=%.4f±%.4f  MCC=%.4f±%.4f  F1=%.4f±%.4f",
            cfg.name,
            row.get("mean_auc", float("nan")), row.get("std_auc", float("nan")),
            row.get("mean_auprc", float("nan")), row.get("std_auprc", float("nan")),
            row.get("mean_mcc", float("nan")), row.get("std_mcc", float("nan")),
            row.get("mean_f1", float("nan")), row.get("std_f1", float("nan")),
        )

    result = pd.DataFrame(rows)
    result.to_csv(dest / "ablation_results.csv", index=False)
    (dest / "ablation_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    return result
